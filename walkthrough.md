# SplitOculo 完整开发指南

本文档面向参与本项目的研究者与开发者，详细介绍 SplitOculo 的系统架构、训练流程、部署方式以及后续研究方向。内容与 `README-zh.md` 互补，不做重复。

---

## 目录

1. [系统设计概述](#1-系统设计概述)
2. [环境准备](#2-环境准备)
3. [数据准备与特征预计算](#3-数据准备与特征预计算)
4. [基线训练（固定 49 tokens）](#4-基线训练固定-49-tokens)
5. [Information-Aware 自适应传输训练](#5-information-aware-自适应传输训练)
6. [量化感知训练 (QAT)](#6-量化感知训练-qat)
7. [权重拆分与 ONNX 导出](#7-权重拆分与-onnx-导出)
8. [边云部署与推理](#8-边云部署与推理)
9. [模块说明](#9-模块说明)
10. [后续步骤](#10-后续步骤)

---

## 1. 系统设计概述

SplitOculo 在 Qwen2.5-VL-3B 的视觉编码器中间层做切分，形成边端-云端两级推理流水线：

```
边端:
  输入图像 → MobileNetV2(layer3) → [B,96,14,14]
           → StridedTokenProjector/EdgeProjector → [B,49,1280]
           → DimensionBottleneck.encode → [B,49,64]
           → (可选) TokenImportanceScorer → importance logits [B,49]
           → (可选) SoftBudgetedTransmission → 选出 K 个最重要的 token
           → INT8 量化 + base64 编码
           → HTTP POST (~1.5-3.1 KB)

云端:
  接收负载 → dequantize → DimensionBottleneck.decode → [B,K,1280]
           → (可选) SparseTokenUpsampler → 稀疏补全 → [B,49,1280]
           → TransformerUpsampler → [B,256,1280]
           → 注入 Qwen ViT 第 split_layer 层之后 → LLM 生成回答
```

### 两种工作模式

| 模式 | 说明 |
|---|---|
| **基线模式** | 固定传输 49 个 token，统一 INT8 精度，传输量 ~3.1 KB |
| **Information-Aware 模式** | 对 49 个 token 评分，选出 K 个最重要的 token（K 可变），云端稀疏补全后上采样 |

---

## 2. 环境准备

### 依赖安装

```bash
# 创建 conda 环境
conda create -n splitoculo python=3.10 -y
conda activate splitoculo

# 安装依赖
pip install -r requirements.txt

# 确认 PyTorch + CUDA 可用
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

本仓库在以下环境验证通过：

- Python 3.10, PyTorch 2.6.0+cu124, timm 1.0.24
- CUDA 12.4, Windows 11 / Linux
- GPU: 8GB+ 显存推荐（训练）；推理可 CPU

### 仓库结构（关键文件）

```
SplitOculo/
├── core/
│   ├── qwen_extractor.py           # Qwen ViT 特征提取器（支持 layer -1~32）
│   └── utils.py                    # set_seed, get_logger, count_parameters
├── models/
│   ├── strided_projector.py        # StridedTokenProjector: CNN feat → 49 tokens
│   ├── bottleneck.py               # DimensionBottleneck: 1280→64 压缩（含 STE QAT）
│   ├── cloud_upsampler.py          # TransformerUpsampler + SparseTokenUpsampler
│   ├── discriminator.py            # FeatureDiscriminator / PatchDiscriminator
│   ├── importance_scorer.py        # TokenImportanceScorer / TextAwareImportanceScorer
│   └── budgeted_transmission.py    # SoftBudgetedTransmission（软掩码 + top-K）
├── scripts/
│   ├── train_gan.py                # 主训练脚本（warmup + GAN，支持 importance-aware）
│   ├── precompute_qwen_features.py # 离线预计算 Qwen teacher 特征
│   ├── edge_client.py              # 边端推理客户端
│   ├── cloud_server.py             # 云端推理服务器
│   ├── split_checkpoint.py         # 权重拆分为 edge/cloud
│   ├── export_onnx.py              # ONNX 导出
│   └── measure_feature_stats.py    # 特征分布统计
├── checkpoints/                    # 训练产物
├── data/                           # 数据集与预计算特征
└── local_research/                 # 研究笔记
```

---

## 3. 数据准备与特征预计算

### 3.1 COCO 数据集

```bash
# 下载 COCO val2017（快速验证用）
mkdir -p data/coco
wget http://images.cocodataset.org/zips/val2017.zip -P data/coco/
unzip data/coco/val2017.zip -d data/coco/

# 下载 COCO train2017（完整训练用）
wget http://images.cocodataset.org/zips/train2017.zip -P data/coco/
unzip data/coco/train2017.zip -d data/coco/
```

### 3.2 预计算 Qwen 教师特征

训练默认使用**静态模式**，即先离线提取 Qwen ViT 中间层特征到 `.pt` 文件，然后训练时直接加载。这避免了训练期间加载 Qwen 模型的显存开销。

```bash
# Layer 4 特征预计算（推荐切分层）
python scripts/precompute_qwen_features.py \
  --data_dir ./data/coco \
  --output_dir ./data/coco_features_layer4 \
  --layer 4 \
  --split train

python scripts/precompute_qwen_features.py \
  --data_dir ./data/coco \
  --output_dir ./data/coco_features_layer4 \
  --layer 4 \
  --split val
```

> **提示**：如果显存不足以同时加载 Qwen 和训练模型，务必使用静态模式。动态模式 (`--dynamic`) 可实时计算但显存需求翻倍。

---

## 4. 基线训练（固定 49 tokens）

基线模式固定传输全部 49 个 token，使用统一 INT8 量化。

### 4.1 Phase 1: Warmup（MSE 预热）

```bash
python scripts/train_gan.py \
  --features_dir ./data/coco_features_layer4 \
  --data_dir ./data/coco \
  --phase warmup \
  --epochs 20 \
  --batch_size 32 \
  --lr_g 1e-3 \
  --bottleneck_dim 64 \
  --bottleneck_method linear \
  --output_dir ./checkpoints/baseline_layer4
```

Warmup 阶段只用 MSE loss 训练 CNN backbone + projector + bottleneck + upsampler，让模型先学会基本的特征压缩与重建。

### 4.2 Phase 2: GAN 对抗微调

```bash
python scripts/train_gan.py \
  --features_dir ./data/coco_features_layer4 \
  --data_dir ./data/coco \
  --phase gan \
  --warmup_checkpoint ./checkpoints/baseline_layer4/warmup_best.pth \
  --epochs 30 \
  --batch_size 32 \
  --lr_g 1e-4 \
  --bottleneck_dim 64 \
  --bottleneck_method linear \
  --output_dir ./checkpoints/baseline_layer4
```

GAN 阶段在 MSE 基础上引入 adversarial loss，让重建特征更锐利、分布更接近真实 Qwen 特征。

### 4.3 验证基线质量

训练日志会打印每个 epoch 的 `cos_sim`（余弦相似度）。Layer 4 基线通常可达到 ~0.89。

---

## 5. Information-Aware 自适应传输训练

这是本项目的核心研究增量。通过 `--importance_aware` 开关启用。

### 5.1 工作原理

1. **Token 重要性评分** (`models/importance_scorer.py`)：对 49 个 token 各生成一个 importance logit
2. **软掩码选择** (`models/budgeted_transmission.py`)：
   - 训练时：sigmoid(logit / temperature) × token，完全可微
   - 推理时：hard top-K，只传输 K 个 token + 它们的索引
3. **稀疏补全** (`models/cloud_upsampler.py` → `SparseTokenUpsampler`)：
   - 收到的 K 个 token 放到对应位置
   - 缺失位置填入可学习的占位嵌入 + 位置编码 + 存在/缺失指示向量
   - 经 2 层 Transformer 补全后再上采样
4. **预算约束**：budget loss 约束平均传输 token 数趋近目标；entropy loss 鼓励 mask 二值化
5. **温度退火**：训练过程中逐步降低 temperature，使 soft mask 趋向 hard 0/1

### 5.2 训练命令

#### Phase 1: Warmup + Information-Aware

```bash
python scripts/train_gan.py \
  --features_dir ./data/coco_features_layer4 \
  --data_dir ./data/coco \
  --phase warmup \
  --epochs 30 \
  --batch_size 32 \
  --lr_g 1e-3 \
  --bottleneck_dim 64 \
  --bottleneck_method linear \
  --importance_aware \
  --scorer_method mlp \
  --token_budget 24 \
  --min_tokens 8 \
  --budget_temperature 1.0 \
  --anneal_rate 0.01 \
  --lambda_budget 0.1 \
  --lambda_entropy 0.01 \
  --completion_layers 2 \
  --output_dir ./checkpoints/importance_layer4
```

#### Phase 2: GAN + Information-Aware

```bash
python scripts/train_gan.py \
  --features_dir ./data/coco_features_layer4 \
  --data_dir ./data/coco \
  --phase gan \
  --warmup_checkpoint ./checkpoints/importance_layer4/warmup_best.pth \
  --epochs 30 \
  --batch_size 32 \
  --lr_g 1e-4 \
  --bottleneck_dim 64 \
  --bottleneck_method linear \
  --importance_aware \
  --scorer_method mlp \
  --token_budget 24 \
  --min_tokens 8 \
  --lambda_budget 0.1 \
  --lambda_entropy 0.01 \
  --completion_layers 2 \
  --output_dir ./checkpoints/importance_layer4
```

### 5.3 关键超参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--token_budget` | 24 | 期望传输的平均 token 数（49 的约一半） |
| `--min_tokens` | 8 | 推理时最少传输的 token 数 |
| `--budget_temperature` | 1.0 | 初始温度，越大 mask 越软 |
| `--anneal_rate` | 0.01 | 每 epoch 温度衰减量 |
| `--lambda_budget` | 0.1 | budget loss 权重，过大会牺牲质量 |
| `--lambda_entropy` | 0.01 | entropy loss 权重，鼓励二值化 |
| `--scorer_method` | mlp | 评分方法：`mlp`（推荐）或 `attention` |
| `--completion_layers` | 2 | 稀疏补全 Transformer 层数 |

### 5.4 预期效果

- 传输量：从 ~3.1 KB (49 tokens × 64 bytes) 降至 ~1.5 KB (24 tokens × 64 bytes)，约 **50% 带宽节省**
- cos_sim：预期略低于基线（~0.85-0.88），需实际训练验证

---

## 6. 量化感知训练 (QAT)

在瓶颈层 (`DimensionBottleneck`) 中嵌入 STE（Straight-Through Estimator）伪量化，使模型在训练时就适应量化误差。

### 6.1 工作原理

`bottleneck.py` 中的 `fake_quantize()` 方法：
- **前向**：encode 输出经过 fake quantize（clamp + round + scale），模拟 INT8/INT4 精度
- **反向**：STE 跳过不可微的 round 操作，梯度直通

### 6.2 训练命令

在任何训练命令上追加 `--quantize_aware` 即可：

```bash
python scripts/train_gan.py   --features_dir ./data/coco_features_layer4 \
  --data_dir ./data/coco \
  --phase warmup \
  --epochs 20 \
  --bottleneck_dim 64 \
  --importance_aware \
  --quantize_aware \
  --num_bits 8 \
  --output_dir ./checkpoints/qat_importance
```

可用 `--num_bits 4` 测试更激进的低精度量化效果。

---

## 7. 权重拆分与 ONNX 导出

### 7.1 权重拆分

将训练好的单体 checkpoint 拆分为边端和云端两份权重：

```bash
python scripts/split_checkpoint.py \
  --input ./checkpoints/importance_layer4/gan_best.pth \
  --output_dir ./checkpoints/importance_layer4/split/
```

输出：
- `split/edge_weights.pth` — CNN + projector + bottleneck encoder + importance scorer + budgeted transmission
- `split/cloud_weights.pth` — bottleneck decoder + sparse upsampler / upsampler

脚本会打印各部分的参数量。

### 7.2 ONNX 导出

```bash
python scripts/export_onnx.py \
  --checkpoint ./checkpoints/importance_layer4/split/edge_weights.pth \
  --output_dir ./checkpoints/importance_layer4/split/ \
  --bottleneck_dim 64
```

Information-Aware 模式下 ONNX 模型有两个输出：`compressed_features` 和 `importance_logits`，边端运行时可根据 logits 做 top-K 选择。

---

## 8. 边云部署与推理

### 8.1 启动云端服务

```bash
python scripts/cloud_server.py \
  --checkpoint ./checkpoints/importance_layer4/split/cloud_weights.pth \
  --port 8080 \
  --offline
```

`--offline` 表示使用已缓存的 Qwen 模型权重，不从 HuggingFace 下载。

### 8.2 运行边端推理

```bash
# 基线模式
python scripts/edge_client.py \
  --checkpoint ./checkpoints/baseline_layer4/split/edge_weights.pth \
  --image ./test.jpg \
  --server http://CLOUD_IP:8080 \
  --timeout 300

# Information-Aware 模式（自动检测 checkpoint 中的 importance_scorer）
python scripts/edge_client.py \
  --checkpoint ./checkpoints/importance_layer4/split/edge_weights.pth \
  --image ./test.jpg \
  --server http://CLOUD_IP:8080 \
  --timeout 300
```

Information-Aware 模式下，`edge_client.py` 会：
1. 从 checkpoint 自动加载 importance scorer 和 budgeted transmission 模块
2. 对 49 个 token 评分并选出 top-K
3. 只传输选中的 token + 它们的索引
4. 云端接收后调用 SparseTokenUpsampler 补全

### 8.3 传输负载格式

| 字段 | 基线模式 | Information-Aware 模式 |
|---|---|---|
| `features` | base64(int8, [49,64]) | base64(int8, [K,64]) |
| `indices` | 不发送 | JSON array, 长度 K |
| `total_tokens` | 不发送 | 49 |
| `bottleneck_dim` | 64 | 64 |

---

## 9. 模块说明

### 9.1 TokenImportanceScorer (`models/importance_scorer.py`)

两种评分方法：

- **MLP**（推荐）：`[B,49,1280] → Linear → GELU → Linear → [B,49]` — 输出 raw logits，每个 token 独立评分
- **Attention**：用可学习 query 对所有 token 做交叉注意力，考虑全局上下文

### 9.2 TextAwareImportanceScorer (`models/importance_scorer.py`)

在语义评分基础上加入 CNN 分支，从 `[B,96,14,14]` 的 CNN 特征中检测文字密集区域：

```
CNN features → Conv2d(96→32→1) → sigmoid → text_map [B,1,7,7]
            → flatten → [B,49]
            → 与语义 logits 经 fusion layer 加权融合
```

适用于 OCR、文档、图表等文字密集场景，需要 PaddleOCR 预标注数据训练。

### 9.3 SoftBudgetedTransmission (`models/budgeted_transmission.py`)

- **训练**：`mask = sigmoid(logits / temperature)`，soft masking，完全可微
- **推理**：`topk(logits, K)` → 只传输 K 个 token
- **损失**：
  - `budget_loss = (mean(mask) - target_ratio)²` — 约束平均选中数
  - `entropy_loss = -mean(mask * log(mask))` — 鼓励 0/1 二值化
- **温度退火**：每 epoch 降低 temperature，从 soft 渐变为 hard

### 9.4 SparseTokenUpsampler (`models/cloud_upsampler.py`)

云端接收稀疏 token 后的处理流程：

1. 创建 `[B, 49, D]` 全零 slots
2. 收到的 K 个 token 填入对应位置
3. 缺失位置填入 `missing_token_embed`（可学习）
4. 叠加 `slot_position_embed`（49 个位置的位置编码）
5. 叠加 `presence_indicator`（存在/缺失二值指示）
6. 经 2 层 Transformer 补全
7. 输出 `[B, 49, 1280]` → 再经 TransformerUpsampler → `[B, 256, 1280]`

### 9.5 DimensionBottleneck + STE QAT (`models/bottleneck.py`)

- `encode()`: `[B,N,1280] → [B,N,64]`，启用 QAT 时在末尾调用 `fake_quantize()`
- `decode()`: `[B,N,64] → [B,N,1280]`
- `fake_quantize()`: `x_clamp → round(x * scale) / scale`，forward 用量化值，backward STE 直通

---

## 10. 后续步骤

### 立即可做

1. **训练 Information-Aware 模型**：按第 5 节命令训练，对比基线 cos_sim 和传输量
2. **消融实验**：调节 `--token_budget`（16, 24, 32, 40）绘制 Rate-Distortion 曲线
3. **QAT 评估**：对比 `--quantize_aware` 前后的精度损失

### 中期研究

4. **M3: Text-Aware Prior 训练**
   - 用 PaddleOCR 对 COCO/CC3M 图像生成文字区域标注（二值掩码）
   - 用 `--text_aware` 启用 `TextAwareImportanceScorer`
   - 需准备 text region GT 数据，当前训练脚本中 text-aware 分支的 GT 加载需要补充

5. **M4: 自适应混合精度量化**
   - 实现 `models/adaptive_quantizer.py`：根据 importance logits 为每个 token 分配不同位宽（INT4/INT8）
   - 高重要性 token → INT8，低重要性 token → INT4
   - 使用 Gumbel-Softmax 使位宽分配可微

### 评估与论文

6. **端到端 VQA 评估**：在 VQAv2、TextVQA、ChartQA 上对比完整 Qwen 基线
7. **设备延迟测量**：在真实边缘设备（树莓派 / Jetson）上测量 ONNX 推理延迟
8. **Rate-Distortion 分析**：绘制 传输字节数 vs cos_sim / VQA accuracy 曲线
9. **消融论文图表**：对比 MLP vs Attention scorer、不同 budget、有无 QAT 等

### 工程优化

10. **C++ 边端集成**：将 importance scorer ONNX 输出集成到 `cpp_edge_client/`
11. **Electron GUI 更新**：在 GUI 中展示 token 重要性热力图
12. **批量推理**：云端支持 batch 请求以提高吞吐

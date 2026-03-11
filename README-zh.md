# SplitOculo

<p align="center">
  <b>VLM 边端-云端协同视觉特征切分</b>
</p>

<p align="center">
  <a href="README.md">English</a>
</p>

---

## 📖 概述

SplitOculo 实现 **边端-云端协同 VLM 推理**：
- 🖥️ **端侧**: CNN + Projector + Bottleneck → 3 KB 压缩特征
- ☁️ **云端**: 解压 + 上采样 → 256 tokens → Qwen → LLM

---

## 🆕 v2.2 更新: 真实网络分离部署

> [!TIP]
> **v2.2 支持端侧和云端通过 HTTP 真实分离部署！**

### 新特性
- **网络分离**: `cloud_server.py` + `edge_client.py` 实现真实的端云分离
- **静态权重拆分**: 使用 `split_checkpoint.py` 将 AIO 权重拆分为端侧 (~11 MB) 和云端 (~486 MB)
- **瓶颈层压缩**: 61 KB → 3 KB，压缩比 20×
- **离线模式**: `--offline` 标志支持无网络加载 Qwen

---

## 🚀 快速开始

### 📥 下载预训练权重 (可选)

如果你想跳过训练步骤直接使用预训练的拆分权重：

```bash
# 下载拆分后的权重文件
wget https://github.com/Shimmer22/SplitOculo/releases/download/v2.2/edge_weights.pth -O checkpoints/split/edge_weights.pth
wget https://github.com/Shimmer22/SplitOculo/releases/download/v2.2/cloud_weights.pth -O checkpoints/split/cloud_weights.pth
```

或手动下载：
- **[edge_weights.pth](https://github.com/Shimmer22/SplitOculo/releases/download/v2.2/edge_weights.pth)** (11 MB) - 端侧权重
- **[cloud_weights.pth](https://github.com/Shimmer22/SplitOculo/releases/download/v2.2/cloud_weights.pth)** (486 MB) - 云端权重

下载后将文件放置到 `checkpoints/split/` 目录，然后直接跳到 **步骤 5: 网络分离部署**。

---

### 步骤 0: 环境配置

```bash
git clone https://github.com/Shimmer22/SplitOculo.git
cd SplitOculo

conda create -n cnn_vit python=3.10 -y
conda activate cnn_vit

pip install -r requirements.txt
```

### 步骤 1: 下载数据集 (COCO val2017)

```bash
mkdir -p data/coco
wget http://images.cocodataset.org/zips/val2017.zip -P data/coco/
unzip data/coco/val2017.zip -d data/coco/
rm data/coco/val2017.zip

# 划分 train/val (80/20)
python -c "
import os, shutil, random
from pathlib import Path

src = Path('data/coco/val2017')
images = list(src.glob('*.jpg'))
random.seed(42)
random.shuffle(images)

train_dir = Path('data/coco/train')
val_dir = Path('data/coco/val')
train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

split = int(len(images) * 0.8)
for img in images[:split]:
    shutil.copy(img, train_dir / img.name)
for img in images[split:]:
    shutil.copy(img, val_dir / img.name)

print(f'✅ 训练集: {split}, 验证集: {len(images)-split}')
"
```

### 步骤 2: 预计算 Qwen 特征

```bash
# 默认: 对齐到 layer 4 (block 4 输出, 最小练)
python scripts/precompute_qwen_features.py \
    --data_dir ./data/coco --output_dir ./data/coco_features_layer4 \
    --layer 4 --split train

python scripts/precompute_qwen_features.py \
    --data_dir ./data/coco --output_dir ./data/coco_features_layer4 \
    --layer 4 --split val
```

> [!TIP]
> 如需对比不同对齐层级，参考「层级对齐实验」一节。

### 步骤 3: 训练 (启用瓶颈层)

```bash
# Phase 1: Warmup
python scripts/train_gan.py \
    --features_dir ./data/coco_features_layer4 \
    --data_dir ./data/coco \
    --phase warmup \
    --epochs 20 \
    --bottleneck_dim 64 \
    --bottleneck_method linear \
    --output_dir ./checkpoints/gan_bottleneck

# Phase 2: GAN 微调
python scripts/train_gan.py \
    --features_dir ./data/coco_features_layer4 \
    --data_dir ./data/coco \
    --phase gan \
    --warmup_checkpoint ./checkpoints/gan_bottleneck/warmup_best.pth \
    --epochs 30 \
    --bottleneck_dim 64 \
    --output_dir ./checkpoints/gan_bottleneck
```

### 步骤 4: 拆分权重

```bash
python scripts/split_checkpoint.py \
    --input ./checkpoints/gan_bottleneck/gan_best.pth \
    --output_dir ./checkpoints/gan_bottleneck/split/
```

输出：
- `edge_weights.pth` (~11 MB): CNN + Projector + Bottleneck.encoder
- `cloud_weights.pth` (~486 MB): Bottleneck.decoder + Upsampler

### 步骤 5: 网络分离部署

**云端服务器**:
```bash
python scripts/cloud_server.py \
    --checkpoint ./checkpoints/gan_bottleneck/split/cloud_weights.pth \
    --port 8080 \
    --offline
```

**端侧客户端**:
```bash
python scripts/edge_client.py \
    --checkpoint ./checkpoints/gan_bottleneck/split/edge_weights.pth \
    --image ./test.jpg \
    --server http://云端IP:8080 \
    --timeout 300
```

---

## 🧪 层级对齐实验

测试将 CNN+Upsampler 对齐到 Qwen ViT 不同层级的效果。

### 层级语义

| split_layer | 对齐目标 | 推理数据流 | 特征维度 |
|------------|--------|------------|--------|
| `-1` | 原始像素 patches (JPEG 级别) | upsampled → patch_embed → blocks[0:] → merger | 3×pH×pW |
| `0` | patch_embed 输出 | upsampled → blocks[0:] → merger | 1280 |
| `4` | block 4 输出 (默认) | upsampled → blocks[4:] → merger | 1280 |
| `8` | block 8 输出 | upsampled → blocks[8:] → merger | 1280 |
| `16` | block 16 输出 | upsampled → blocks[16:] → merger | 1280 |

### 各层特征分布统计 (COCO, ~200 样本)

| 层级 | mean | std | 备注 |
|------|------|-----|------|
| `-1` (pixel, 1176 dim) | -0.041 | 1.015 | 3×2×14×14, 不归一化 |
| `0` (patch_embed, 1280 dim) | -0.000 | 0.362 | 均已写入代码 |
| `4` (block 4, 1280 dim) | -0.022 | 0.847 | 默认 |
| `8` (block 8, 1280 dim) | -0.021 | 1.066 | 均已写入代码 |
| `16` (block 16, 1280 dim) | -0.030 | 2.255 | 均已写入代码 |

获取待填充的 magic number：
```bash
# 从现有 layer4 目录测量
python scripts/measure_feature_stats.py \
    --features_dir ./data/coco_features_layer4 --split train

# 一次性测量全部 4 个层级 (需加载 Qwen, 约 10 min)
python scripts/measure_feature_stats.py \
    --data_dir ./data/coco --split train --max_files 100 --realtime --all_layers
```

### 预计算各层特征

```bash
# Layer 0: patch_embed 输出
python scripts/precompute_qwen_features.py \
    --data_dir ./data/coco --output_dir ./data/coco_features_layer0 --layer 0 --split train
python scripts/precompute_qwen_features.py \
    --data_dir ./data/coco --output_dir ./data/coco_features_layer0 --layer 0 --split val

# Layer 8
python scripts/precompute_qwen_features.py \
    --data_dir ./data/coco --output_dir ./data/coco_features_layer8 --layer 8 --split train
python scripts/precompute_qwen_features.py \
    --data_dir ./data/coco --output_dir ./data/coco_features_layer8 --layer 8 --split val

# Layer -1 (pixel patches, 无需完整 Qwen 前向)
python scripts/precompute_qwen_features.py \
    --data_dir ./data/coco --output_dir ./data/coco_features_layer-1 --layer -1 --split train
python scripts/precompute_qwen_features.py \
    --data_dir ./data/coco --output_dir ./data/coco_features_layer-1 --layer -1 --split val
```

### 对应训练命令

```bash
# 以 layer 8 为例
python scripts/train_gan.py \
    --features_dir ./data/coco_features_layer8 \
    --data_dir ./data/coco --phase warmup --epochs 20 \
    --output_dir ./checkpoints/layer8

# 以 layer 0 为例
python scripts/train_gan.py \
    --features_dir ./data/coco_features_layer0 \
    --data_dir ./data/coco --phase warmup --epochs 20 \
    --output_dir ./checkpoints/layer0
```

---

## 📐 架构 (v2.2)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 端侧 (EDGE)                               [edge_client.py]              │
│  图像 → MobileNet → Projector → Bottleneck.encode()                     │
│                           ↓                                              │
│              [49 × 64] int8 量化 → base64 编码                          │
└───────────────────────────────────────┬─────────────────────────────────┘
                                        │ HTTP POST (~3 KB payload)
┌───────────────────────────────────────▼─────────────────────────────────┐
│ 云端 (CLOUD)                              [cloud_server.py]             │
│  Flask Server @ :8080                                                    │
│  反量化 → Bottleneck.decode() → Upsampler → Qwen[4:] → LLM             │
│                           ↓                                              │
│              JSON Response: {"response": "图片描述..."}                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### 端侧 vs 云端对比

| | 端侧 | 云端 |
|---|---|---|
| **模型构成** | MobileNetV2 + StridedProjector + Bottleneck.encoder | Bottleneck.decoder + TransformerUpsampler |
| **权重文件** | 11 MB | 486 MB |
| **激活参数量** | 2.87M | 126.63M |
| **额外计算** | - | Qwen ViT [4:32] + Merger + LLM |

---

## 📝 关键参数

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--bottleneck_dim` | 0 | 瓶颈层维度 (推荐 64/128) |
| `--bottleneck_method` | linear | linear / mlp / autoencoder |
| `--lambda_recon` | 0.1 | 瓶颈层重建损失权重 |
| `--upsampler_type` | transformer | transformer / mlp / deconv |
| `--phase` | - | warmup (MSE) / gan (对抗) |

### 部署参数

| 参数 | 说明 |
|------|------|
| `--offline` | 离线模式，不连接 HuggingFace |
| `--qwen_path` | Qwen 模型路径 |
| `--timeout` | 请求超时时间 (秒) |

---

## 📊 传输大小对比

| bottleneck_dim | 传输大小 (int8) | 压缩比 |
|----------------|-----------------|--------|
| 禁用 (1280) | 61 KB | 1× |
| 128 | 6.1 KB | 10× |
| **64** | **3.1 KB** | **20×** |
| 32 | 1.5 KB | 40× |

---

## 📄 许可证

MIT License


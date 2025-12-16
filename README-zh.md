# SplitOculo

<p align="center">
  <b>智能眼镜边端-云端协同视觉特征切分框架</b>
</p>

<p align="center">
  <a href="README.md">English</a>
</p>

---

## 📖 概述

SplitOculo 是一个用于智能眼镜**边端-云端协同计算**的研究框架，支持：

- 📊 **性能评估**：评估不同切分点的特征尺寸和计算量
- 🎓 **特征蒸馏**：训练 CNN 模型逼近 CLIP ViT 输出
- 🔗 **Qwen2.5-VL 对齐**：将 CNN 特征对齐到 Qwen2.5-VL 用于多模态任务

---

## 🏗️ 项目结构

```
SplitOculo/
├── main_benchmark.py       # 性能评估入口
│
├── scripts/                # 训练和推理脚本
│   ├── train_distill.py              # CLIP 蒸馏训练
│   ├── precompute_qwen_features.py   # 预提取 Qwen 特征
│   ├── train_with_precomputed.py     # 使用预计算特征训练
│   ├── infer.py                      # CLIP 对齐推理
│   ├── infer_qwen.py                 # Qwen 对齐推理
│   └── infer_hybrid.py               # 混合 CNN+Qwen 推理
│
├── core/                   # 核心框架
├── models/                 # 模型定义
├── data/                   # 数据工具
├── checkpoints/            # 训练检查点
└── results/                # 评估输出
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. Qwen2.5-VL 对齐（推荐）

**步骤 1：预计算 Qwen 特征（支持中间层）**
```bash
# 提取第 4 层特征（浅层，易学习）
python scripts/precompute_qwen_features.py \
    --data_dir ./data/imagenette2-320 \
    --layer 4 \
    --split train

python scripts/precompute_qwen_features.py \
    --data_dir ./data/imagenette2-320 \
    --layer 4 \
    --split val
```

**步骤 2：使用预计算特征训练**
```bash
python scripts/train_with_precomputed.py \
    --features_dir ./data/qwen_features \
    --target_hidden_size 1280 \
    --epochs 100
```

### 3. 推理

> **⚠️ 重要**：推理时 `--split_layer` 参数必须与预计算特征时的 `--layer` 一致！

```bash
# 仅端侧编码（快速，无需 Qwen）
python scripts/infer_hybrid.py \
    --checkpoint checkpoints/qwen_precomputed/best_model.pth \
    --image photo.jpg

# 完整混合推理（端侧 CNN + 云端 Qwen 深层）
python scripts/infer_hybrid.py \
    --checkpoint checkpoints/qwen_precomputed/best_model.pth \
    --image photo.jpg \
    --split_layer 4 \
    --full_inference
```

---

## 📐 架构

### 当前架构

```
         ┌─────────────────────────────────────────────────────┐
端侧:    │ 图像 → CNN → Projector → 49 tokens (7×7, 1280 维)    │
         └─────────────────────┬───────────────────────────────┘
                               │ ~61 KB (int8) 传输
         ┌─────────────────────▼───────────────────────────────┐
云端:    │ 双线性上采样 49→256 → Qwen[4:] → Merger → LLM        │
         └─────────────────────────────────────────────────────┘
```

### Projector 的作用

**Projector** 是一个轻量级 CNN 模块，负责：
1. 转换 CNN 特征通道 (96ch → 1280 dim)
2. 通过 AvgPool 下采样空间分辨率 (14×14 → 7×7)
3. 转换为 token 序列以兼容 Qwen

**当前上采样是无参数的**（双线性插值），这是性能瓶颈（见下方实验）。

---

## 🔬 消融实验

我们进行了系统性实验来定位语义丢失的瓶颈：

| 实验 | 输入 | 输出 | 结果 |
|------|------|------|------|
| A. Teacher 256 tokens (不下采样) | Qwen Layer 4 直接输出 | "person wearing hat with feather" | ✅ 正确 |
| B. 预计算 .pt 256 tokens | 从训练数据读取 | "fishing net with fish" | ✅ 正确 |
| C. Teacher 256→49→256 (下采样+上采样) | 先下采样再上采样 | "textured fabric, diagonal stripes" | ❌ 错误 |
| D. CNN 49 tokens + 上采样 | CNN 输出 | "pixelated pattern" / "black" | ❌ 错误 |

### 关键发现

1. **流水线正确**：实验 A & B 证明 blocks[4:] + merger + LLM 工作正常
2. **上采样是瓶颈**：实验 C 显示即使 teacher 特征经过 256→49→256 也会丢失语义
3. **49 tokens 不足**：7×7 空间分辨率无法保持图像语义
4. **CNN 质量是次要因素**：0.86 cos_sim 的 CNN 不是主要问题，上采样才是

### 启示

要解决语义丢失，需要：
- **方案 1**：传输 256 tokens（更大带宽，~320 KB int8）
- **方案 2**：在云端训练**可学习上采样器**（推荐）
- **方案 3**：减少 CNN 下采样，直接输出 256 tokens

---

## 📊 训练结果

| 目标 | Val Cos Sim | Val MSE | 状态 |
|------|-------------|---------|------|
| Layer 4 (1280) | **0.86** | 0.63 | ⚠️ 训练正常，上采样瓶颈 |
| Layer 8 (1280) | ~0.77 | 1.07 | ✅ 可学习 |
| Merger (2048) | ~0.00 | ~4.8 | ❌ 太难 |

### 传输大小

| 格式 | 大小 | 说明 |
|------|------|------|
| JPEG 224×224 (Q85) | ~13 KB | 基准 |
| 49 tokens (int8) | ~61 KB | 当前方案，有语义丢失 |
| 256 tokens (int8) | ~320 KB | 无语义丢失 |

---

## 🔮 未来工作：可学习上采样器

一个有前景的方向是在云端训练**可学习上采样器**：

```
         ┌─────────────────────────────────────────────────────┐
端侧:    │ 图像 → CNN → Projector → 49 tokens (7×7, 1280 维)    │
         └─────────────────────┬───────────────────────────────┘
                               │ ~61 KB 传输
         ┌─────────────────────▼───────────────────────────────┐
云端:    │ 可学习上采样 49→256 → Qwen[4:] → Merger → LLM        │
         │     [可训练]                                         │
         └─────────────────────────────────────────────────────┘
```

上采样器可以冻结 Qwen blocks 进行端到端训练，或联合训练。

---

## 📝 关键参数

### 预计算
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--layer` | 8 | ViT 层 (1-32, 或 -1 表示 merger) |
| `--split` | train | train 或 val |

### 训练
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--target_hidden_size` | 1280 | 目标维度 (1280 中间层, 2048 merger) |
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 64 | 批大小 |

### 推理
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--image` | - | 输入图片路径 |
| `--split_layer` | 4 | 必须与训练层一致 |
| `--quantize` | none | none/fp16/int8 |
| `--full_inference` | False | 在云端运行 Qwen 深层 |

---

## 📚 参考文献

- [Qwen2.5-VL 技术报告](https://arxiv.org/abs/2412.00015)
- [MobileVLM V2](https://arxiv.org/abs/2402.03766)
- [CLIP](https://arxiv.org/abs/2103.00020)

---

## 📄 许可证

MIT License

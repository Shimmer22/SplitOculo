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
- 🖥️ **端侧**: CNN + Projector → 49 tokens (61 KB)
- ☁️ **云端**: 可学习上采样器 → 256 tokens → Qwen → LLM

---

## ⚠️ 当前不足

> [!CAUTION]
> **多图训练无法有效泛化。** 单图过拟合可达 cos_sim=0.99 且语义正确，但多图训练仅能达到 cos_sim=0.87，输出错误。

| 模式 | cos_sim | LLM 输出 |
|------|---------|----------|
| **单图过拟合** | 0.99 | ✅ "modern living room with TV, dining table, kitchen..." |
| **多图训练** | 0.87 | ❌ "gradient background transitioning from light brown..." |

### 根本原因分析
- 0.87 的 cos_sim **不足以**让 LLM 理解语义
- CNN 特征与 Qwen ViT 特征本质不同
- 简单的蒸馏无法弥合这个差距

### 可能的解决方案 (TODO)
1. 端到端微调 Qwen blocks
2. 更强表达力的上采样器架构
3. 任务驱动训练 (VQA loss) 而非特征匹配

---

## 🚀 快速开始

### 步骤 0: 环境配置

```bash
git clone https://github.com/Shimmer22/SplitOculo.git
cd SplitOculo

conda create -n cnn_vit python=3.10 -y
conda activate cnn_vit

pip install torch torchvision transformers timm tqdm pillow matplotlib
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
python scripts/precompute_qwen_features.py \
    --data_dir ./data/coco --output_dir ./data/qwen_features \
    --layer 4 --split train --batch_size 4

python scripts/precompute_qwen_features.py \
    --data_dir ./data/coco --output_dir ./data/qwen_features \
    --layer 4 --split val --batch_size 4
```

### 步骤 3: 训练 (或使用过拟合调试)

**正常训练:**
```bash
python scripts/train_with_upsampler.py \
    --features_dir ./data/qwen_features \
    --data_dir ./data/coco \
    --upsampler_method mlp \
    --epochs 100 --batch_size 32 \
    --output_dir ./checkpoints/coco_mlp
```

**单图过拟合调试 (推荐先运行):**
```bash
python scripts/train_with_upsampler.py \
    --features_dir ./data/qwen_features \
    --data_dir ./data/coco \
    --overfit ./data/qwen_features/train/000000.pt \
    --epochs 500
```

### 步骤 4: 推理

```bash
python scripts/infer_hybrid.py \
    --checkpoint ./checkpoints/coco_mlp/best_model.pth \
    --image ./data/coco/train/000000000139.jpg \
    --full_inference
```

---

## 📐 架构

```
┌─────────────────────────────────────────────────────────────┐
│ 端侧 (EDGE)                                                  │
│  图像 → CNN → Projector → 49 tokens (7×7, 1280 维)           │
└────────────────────────┬────────────────────────────────────┘
                         │ ~61 KB int8
┌────────────────────────▼────────────────────────────────────┐
│ 云端 (CLOUD)                                                 │
│  MLP 上采样 → 256 tokens → Qwen[4:] → Merger → LLM           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 关键发现

| 方法 | cos_sim | 有效? |
|------|---------|-------|
| 纯 Bilinear | 0.87 | ❌ |
| deconv + BN | 0.57 | ❌ |
| **MLP (bilinear + mlp)** | 0.99* | ✅* |

*仅限单图过拟合

---

## 📝 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--upsampler_method` | mlp | mlp / deconv / transformer |
| `--overfit` | None | 指定 .pt 文件进行单图过拟合调试 |
| `--transmission_tokens` | 49 | 端侧 tokens (7×7) |
| `--epochs` | 100 | 训练轮数 |

---

## 📄 许可证

MIT License

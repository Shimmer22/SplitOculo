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

## 🆕 v2.0 更新: TransformerUpsampler + GAN 训练

> [!TIP]
> **v2.0 在训练集图片上达到 cos_sim=0.89，LLM 输出正确！**

### 新特性
- **TransformerUpsampler**: 4层 Transformer + Learned Position Embedding (67M 参数)
- **GAN 训练**: 对抗训练生成更锐利的特征
- **FeatureDiscriminator**: Spectral Norm 稳定 GAN 训练

### 训练结果

| 阶段 | cos_sim | val_std | LLM 输出 |
|------|---------|---------|----------|
| Warmup (仅 MSE) | 0.891 | 0.748 | - |
| **GAN 微调** | **0.893** | **0.768** | ✅ 训练集图片正确 |

### 当前不足

> [!CAUTION]
> **泛化差距**: 训练集图片效果好，但**集合外图片仍输出错误**。

| 图片来源 | LLM 输出质量 |
|----------|--------------|
| 训练集 | ✅ 正确 (如 "一只熊"、"厨房场景，有橱柜") |
| 集合外图片 | ❌ 通常不正确或过于笼统 |

### 根本原因
- CNN 特征与 ViT 全局注意力模式本质不同
- 0.89 cos_sim 仍不足以实现鲁棒的跨域泛化

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

### 步骤 3: GAN 训练 (v2.0)

**Phase 1: Warmup (仅 MSE)**
```bash
python scripts/train_gan.py \
    --features_dir ./data/qwen_features \
    --data_dir ./data/coco \
    --phase warmup \
    --epochs 20 \
    --batch_size 16 \
    --output_dir ./checkpoints/gan_layer4
```

**Phase 2: GAN 微调**
```bash
python scripts/train_gan.py \
    --features_dir ./data/qwen_features \
    --data_dir ./data/coco \
    --phase gan \
    --warmup_checkpoint ./checkpoints/gan_layer4/warmup_best.pth \
    --epochs 50 \
    --lambda_mse 10.0 \
    --lambda_adv 0.1 \
    --output_dir ./checkpoints/gan_layer4
```

**启用瓶颈层压缩 (v2.1)** 🆕
```bash
# 使用线性瓶颈层，压缩到 64 维 (~3 KB 传输)
python scripts/train_gan.py \
    --features_dir ./data/qwen_features \
    --data_dir ./data/coco \
    --phase warmup \
    --epochs 20 \
    --bottleneck_dim 64 \
    --bottleneck_method linear \
    --lambda_recon 0.1 \
    --output_dir ./checkpoints/gan_bottleneck
```

### 步骤 4: 推理

```bash
python scripts/infer_hybrid.py \
    --checkpoint ./checkpoints/gan_layer4/gan_best.pth \
    --image ./data/coco/train/000000000285.jpg \
    --full_inference
```

---

## 📐 架构 (v2.1)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 端侧 (EDGE)                                                              │
│  图像 → CNN → Projector → 49 tokens (7×7, 1280 维)                       │
│                    ↓                                                     │
│           Bottleneck.encode() → 49 tokens (7×7, 64 维) 🆕                │
└───────────────────────────────────────┬─────────────────────────────────┘
                                        │ ~3 KB int8 (压缩 20×)
┌───────────────────────────────────────▼─────────────────────────────────┐
│ 云端 (CLOUD)                                                             │
│  Bottleneck.decode() → 49 tokens (1280 维) 🆕                            │
│                    ↓                                                     │
│  TransformerUpsampler → 256 tokens → Qwen[4:] → LLM                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 关键发现

| 方法 | cos_sim | val_std | 有效? |
|------|---------|---------|-------|
| MLP (v1.0) | 0.87 | 0.74 | ❌ |
| **TransformerUpsampler + GAN (v2.0)** | **0.89** | **0.77** | ✅ (训练集) |

---

## 📝 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--upsampler_type` | transformer | transformer / mlp / deconv |
| `--phase` | - | warmup (MSE) / gan (对抗) |
| `--lambda_mse` | 10.0 | MSE 损失权重 (内容) |
| `--lambda_adv` | 0.1 | 对抗损失权重 (风格) |
| `--transformer_layers` | 4 | TransformerUpsampler 层数 |
| `--bottleneck_dim` | 0 | 瓶颈层维度 (0=禁用, 推荐 64/128) 🆕 |
| `--bottleneck_method` | linear | linear / mlp / autoencoder 🆕 |
| `--lambda_recon` | 0.1 | 瓶颈层重建损失权重 🆕 |

---

## 📄 许可证

MIT License

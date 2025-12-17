# SplitOculo

<p align="center">
  <b>Edge-Cloud Collaborative Vision Feature Splitting for VLM</b>
</p>

<p align="center">
  <a href="README-zh.md">中文文档</a>
</p>

---

## 📖 Overview

SplitOculo enables **edge-cloud collaborative VLM inference**:
- 🖥️ **Edge**: CNN + Projector → 49 tokens (61 KB)
- ☁️ **Cloud**: Learned Upsampler → 256 tokens → Qwen blocks → LLM

---

## 🆕 v2.0 Update: TransformerUpsampler + GAN Training

> [!TIP]
> **v2.0 achieves cos_sim=0.89 with correct LLM outputs on training set images!**

### What's New
- **TransformerUpsampler**: 4-layer Transformer with learned positional embedding (67M params)
- **GAN Training**: Adversarial training produces sharper features
- **FeatureDiscriminator**: Spectral-normalized discriminator for stable GAN training

### Training Results

| Phase | cos_sim | val_std | LLM Output |
|-------|---------|---------|------------|
| Warmup (MSE only) | 0.891 | 0.748 | - |
| **GAN Finetuning** | **0.893** | **0.768** | ✅ Correct on training images |

### Current Limitations

> [!CAUTION]
> **Generalization gap**: Works well on training set images, but **out-of-distribution images still produce incorrect outputs**.

| Image Source | LLM Output Quality |
|--------------|-------------------|
| Training set | ✅ Correct (e.g., "a bear", "kitchen scene with cabinets") |
| OOD images | ❌ Often incorrect or generic |

### Root Cause
- CNN features fundamentally differ from ViT global attention patterns
- 0.89 cos_sim is still insufficient for robust cross-domain generalization

---

## 🚀 Quick Start

### Step 0: Setup Environment

```bash
git clone https://github.com/Shimmer22/SplitOculo.git
cd SplitOculo

conda create -n cnn_vit python=3.10 -y
conda activate cnn_vit

pip install torch torchvision transformers timm tqdm pillow matplotlib
```

### Step 1: Download Dataset (COCO val2017)

```bash
mkdir -p data/coco
wget http://images.cocodataset.org/zips/val2017.zip -P data/coco/
unzip data/coco/val2017.zip -d data/coco/
rm data/coco/val2017.zip

# Split into train/val (80/20)
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

print(f'✅ Train: {split}, Val: {len(images)-split}')
"
```

### Step 2: Precompute Qwen Features

```bash
python scripts/precompute_qwen_features.py \
    --data_dir ./data/coco --output_dir ./data/qwen_features \
    --layer 4 --split train --batch_size 4

python scripts/precompute_qwen_features.py \
    --data_dir ./data/coco --output_dir ./data/qwen_features \
    --layer 4 --split val --batch_size 4
```

### Step 3: Train with GAN (v2.0)

**Phase 1: Warmup (MSE only)**
```bash
python scripts/train_gan.py \
    --features_dir ./data/qwen_features \
    --data_dir ./data/coco \
    --phase warmup \
    --epochs 20 \
    --batch_size 16 \
    --output_dir ./checkpoints/gan_layer4
```

**Phase 2: GAN Finetuning**
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

### Step 4: Inference

```bash
python scripts/infer_hybrid.py \
    --checkpoint ./checkpoints/gan_layer4/gan_best.pth \
    --image ./data/coco/train/000000000285.jpg \
    --full_inference
```

---

## 📐 Architecture (v2.0)

```
┌─────────────────────────────────────────────────────────────┐
│ EDGE                                                         │
│  Image → CNN → Projector → 49 tokens (7×7, 1280 dim)        │
└────────────────────────┬────────────────────────────────────┘
                         │ ~61 KB int8
┌────────────────────────▼────────────────────────────────────┐
│ CLOUD                                                        │
│  TransformerUpsampler → 256 tokens → Qwen[4:] → LLM         │
│  (Bilinear + 4-layer Transformer + Learned PosEmbed)        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 Key Findings

| Method | cos_sim | val_std | Works? |
|--------|---------|---------|--------|
| MLP (v1.0) | 0.87 | 0.74 | ❌ |
| **TransformerUpsampler + GAN (v2.0)** | **0.89** | **0.77** | ✅ (training set) |

---

## 📝 Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--upsampler_type` | transformer | transformer / mlp / deconv |
| `--phase` | - | warmup (MSE) / gan (adversarial) |
| `--lambda_mse` | 10.0 | MSE loss weight (content) |
| `--lambda_adv` | 0.1 | Adversarial loss weight (style) |
| `--transformer_layers` | 4 | TransformerUpsampler depth |

---

## 📄 License

MIT License
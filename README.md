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

## ⚠️ Current Limitations

> [!CAUTION]
> **Multi-image training does not generalize well.** Single-image overfitting achieves cos_sim=0.99 with correct semantics, but multi-image training plateaus at cos_sim=0.87 with wrong outputs.

| Mode | cos_sim | LLM Output |
|------|---------|------------|
| **Single-image overfit** | 0.99 | ✅ "modern living room with TV, dining table, kitchen..." |
| **Multi-image train** | 0.87 | ❌ "gradient background transitioning from light brown..." |

### Root Cause Analysis
- 0.87 cos_sim is **insufficient** for semantic understanding
- CNN features fundamentally differ from Qwen ViT features
- Simple distillation cannot bridge this gap

### Potential Solutions (TODO)
1. End-to-end fine-tuning Qwen blocks
2. More expressive upsampler architecture
3. Task-driven training (VQA loss) instead of feature matching

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

### Step 3: Train (or Debug with Overfit)

**Normal training:**
```bash
python scripts/train_with_upsampler.py \
    --features_dir ./data/qwen_features \
    --data_dir ./data/coco \
    --upsampler_method mlp \
    --epochs 100 --batch_size 32 \
    --output_dir ./checkpoints/coco_mlp
```

**Single-image overfit debug (recommended first):**
```bash
python scripts/train_with_upsampler.py \
    --features_dir ./data/qwen_features \
    --data_dir ./data/coco \
    --overfit ./data/qwen_features/train/000000.pt \
    --epochs 500
```

### Step 4: Inference

```bash
python scripts/infer_hybrid.py \
    --checkpoint ./checkpoints/coco_mlp/best_model.pth \
    --image ./data/coco/train/000000000139.jpg \
    --full_inference
```

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ EDGE                                                         │
│  Image → CNN → Projector → 49 tokens (7×7, 1280 dim)        │
└────────────────────────┬────────────────────────────────────┘
                         │ ~61 KB int8
┌────────────────────────▼────────────────────────────────────┐
│ CLOUD                                                        │
│  MLP Upsampler → 256 tokens → Qwen[4:] → Merger → LLM       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 Key Findings

| Method | cos_sim | Works? |
|--------|---------|--------|
| Bilinear only | 0.87 | ❌ |
| deconv + BN | 0.57 | ❌ |
| **MLP (bilinear + mlp)** | 0.99* | ✅* |

*Only with single-image overfitting

---

## 📝 Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--upsampler_method` | mlp | mlp / deconv / transformer |
| `--overfit` | None | Path to .pt file for single-image debug |
| `--transmission_tokens` | 49 | Edge tokens (7×7) |
| `--epochs` | 100 | Training iterations |

---

## 📄 License

MIT License
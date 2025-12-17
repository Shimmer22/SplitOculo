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

## 🚀 Quick Start (Copy-Paste Commands)

### Step 0: Setup Environment

```bash
# Clone repo
git clone https://github.com/Shimmer22/SplitOculo.git
cd SplitOculo

# Create conda env
conda create -n cnn_vit python=3.10 -y
conda activate cnn_vit

# Install dependencies
pip install torch torchvision transformers timm tqdm pillow matplotlib
```

### Step 1: Download Dataset

We recommend **COCO val2017** (5000 diverse images, ~1GB):

```bash
# Create data directory
mkdir -p data/coco

# Download COCO val2017 images
wget http://images.cocodataset.org/zips/val2017.zip -P data/coco/
unzip data/coco/val2017.zip -d data/coco/
rm data/coco/val2017.zip

# Create train/val splits (80/20)
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

print(f'✅ Train: {split} images, Val: {len(images)-split} images')
"
```

### Step 2: Precompute Qwen Features

```bash
# Precompute train features (takes ~30 min on GPU)
python scripts/precompute_qwen_features.py \
    --data_dir ./data/coco \
    --output_dir ./data/qwen_features \
    --layer 4 \
    --split train \
    --batch_size 4

# Precompute val features
python scripts/precompute_qwen_features.py \
    --data_dir ./data/coco \
    --output_dir ./data/qwen_features \
    --layer 4 \
    --split val \
    --batch_size 4
```

### Step 3: Train with MLP Upsampler

```bash
python scripts/train_with_upsampler.py \
    --features_dir ./data/qwen_features \
    --data_dir ./data/coco \
    --transmission_tokens 49 \
    --target_tokens 256 \
    --upsampler_method mlp \
    --epochs 100 \
    --batch_size 32 \
    --output_dir ./checkpoints/coco_mlp
```

### Step 4: Visualize Training

```bash
python scripts/plot_training.py --log ./checkpoints/coco_mlp/train.log
```

### Step 5: Inference

```bash
python scripts/infer_hybrid.py \
    --checkpoint ./checkpoints/coco_mlp/best_model.pth \
    --image ./data/coco/val/000000000139.jpg \
    --full_inference
```

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ EDGE (端侧)                                                  │
├─────────────────────────────────────────────────────────────┤
│  Image → CNN → Projector → 49 tokens (7×7, 1280 dim)        │
└────────────────────────┬────────────────────────────────────┘
                         │ Transmission (~61 KB int8)
┌────────────────────────▼────────────────────────────────────┐
│ CLOUD (云端)                                                 │
├─────────────────────────────────────────────────────────────┤
│  MLP Upsampler → 256 tokens → Qwen[4:] → Merger → LLM       │
│    [Bilinear + MLP]                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Training Results

| Dataset | Epochs | Val cos_sim | Upsampler |
|---------|--------|-------------|-----------|
| Imagenette | 50 | 0.87 | deconv |
| Imagenette | 50 | **TBD** | mlp |
| COCO | 100 | **TBD** | mlp |

---

## 📝 Key Arguments

### train_with_upsampler.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--upsampler_method` | **mlp** | mlp (best) / deconv / transformer |
| `--transmission_tokens` | 49 | Edge tokens (7×7) |
| `--target_tokens` | 256 | Target for Qwen (16×16) |
| `--student_model` | mobilenetv2_100 | CNN backbone |
| `--epochs` | 100 | Training epochs |

### infer_hybrid.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | - | Path to trained model |
| `--image` | - | Input image path |
| `--full_inference` | False | Run complete Qwen inference |

---

## 🔬 Key Findings

| Method | cos_sim | Semantics |
|--------|---------|-----------|
| Bilinear only | 0.87 | ❌ Wrong |
| deconv + BN | 0.57 | ❌ Wrong |
| **MLP (bilinear + mlp)** | **0.999** | ✅ Correct |

**Root cause**: deconv + BatchNorm destroys information. Use `--upsampler_method mlp`.

---

## 📄 License

MIT License
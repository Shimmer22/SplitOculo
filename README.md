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
- 🖥️ **Edge**: CNN + Projector + Bottleneck → 3 KB compressed features
- ☁️ **Cloud**: Decompress + Upsampler → 256 tokens → Qwen → LLM

---

## 🆕 v2.2 Update: Real Network-Split Deployment

> [!TIP]
> **v2.2 supports real edge-cloud deployment via HTTP!**

### What's New
- **Network Split**: `cloud_server.py` + `edge_client.py` for real edge-cloud separation
- **Static Weight Splitting**: Use `split_checkpoint.py` to split AIO weights into edge (~11 MB) and cloud (~486 MB)
- **Bottleneck Compression**: 61 KB → 3 KB, 20× compression ratio
- **Offline Mode**: `--offline` flag for loading Qwen without internet

---

## 🚀 Quick Start

### 📥 Download Pretrained Weights (Optional)

If you want to skip training and use pretrained split weights directly:

```bash
# Download split weights
wget https://github.com/Shimmer22/SplitOculo/releases/download/v2.2/edge_weights.pth -O checkpoints/split/edge_weights.pth
wget https://github.com/Shimmer22/SplitOculo/releases/download/v2.2/cloud_weights.pth -O checkpoints/split/cloud_weights.pth
```

Or manually download:
- **[edge_weights.pth](https://github.com/Shimmer22/SplitOculo/releases/download/v2.2/edge_weights.pth)** (11 MB) - Edge weights
- **[cloud_weights.pth](https://github.com/Shimmer22/SplitOculo/releases/download/v2.2/cloud_weights.pth)** (486 MB) - Cloud weights

Place the downloaded files in `checkpoints/split/` and jump directly to **Step 5: Network-Split Deployment**.

---

### Step 0: Setup Environment

```bash
git clone https://github.com/Shimmer22/SplitOculo.git
cd SplitOculo

conda create -n cnn_vit python=3.10 -y
conda activate cnn_vit

pip install -r requirements.txt
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
# Default: align to layer 4 (after 4 transformer blocks)
python scripts/precompute_qwen_features.py \
    --data_dir ./data/coco --output_dir ./data/coco_features_layer4 \
    --layer 4 --split train

python scripts/precompute_qwen_features.py \
    --data_dir ./data/coco --output_dir ./data/coco_features_layer4 \
    --layer 4 --split val
```

> [!TIP]
> To compare different alignment layers, see the "Layer Alignment Experiments" section.

### Step 3: Train (with Bottleneck)

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

# Phase 2: GAN Finetuning
python scripts/train_gan.py \
    --features_dir ./data/coco_features_layer4 \
    --data_dir ./data/coco \
    --phase gan \
    --warmup_checkpoint ./checkpoints/gan_bottleneck/warmup_best.pth \
    --epochs 30 \
    --bottleneck_dim 64 \
    --output_dir ./checkpoints/gan_bottleneck
```

### Step 4: Split Weights

```bash
python scripts/split_checkpoint.py \
    --input ./checkpoints/gan_bottleneck/gan_best.pth \
    --output_dir ./checkpoints/gan_bottleneck/split/
```

Output:
- `edge_weights.pth` (~11 MB): CNN + Projector + Bottleneck.encoder
- `cloud_weights.pth` (~486 MB): Bottleneck.decoder + Upsampler

### Step 5: Network-Split Deployment

**Cloud Server**:
```bash
python scripts/cloud_server.py \
    --checkpoint ./checkpoints/gan_bottleneck/split/cloud_weights.pth \
    --port 8080 \
    --offline
```

**Edge Client**:
```bash
python scripts/edge_client.py \
    --checkpoint ./checkpoints/gan_bottleneck/split/edge_weights.pth \
    --image ./test.jpg \
    --server http://CLOUD_IP:8080 \
    --timeout 300
```

---

## 🧪 Layer Alignment Experiments

Test the impact of aligning CNN+Upsampler to different Qwen ViT layers.

### Layer Semantics

| split_layer | Alignment Target | Inference Flow | Feature Dim |
|------------|--------|------------|--------|
| `-1` | Raw pixel patches (JPEG level) | upsampled → patch_embed → blocks[0:] → merger | 3×pH×pW |
| `0` | patch_embed output | upsampled → blocks[0:] → merger | 1280 |
| `4` | block 4 output (default) | upsampled → blocks[4:] → merger | 1280 |
| `8` | block 8 output | upsampled → blocks[8:] → merger | 1280 |
| `16` | block 16 output | upsampled → blocks[16:] → merger | 1280 |

### Feature Distribution Stats (COCO, ~200 samples)

| Layer | mean | std | Notes |
|-------|------|-----|-------|
| `-1` (pixel, 1176 dim) | -0.041 | 1.015 | 3×2×14×14, no normalization |
| `0` (patch_embed, 1280 dim) | -0.000 | 0.362 | Written into code |
| `4` (block 4, 1280 dim) | -0.022 | 0.847 | Default |
| `8` (block 8, 1280 dim) | -0.021 | 1.066 | Written into code |
| `16` (block 16, 1280 dim) | -0.030 | 2.255 | Written into code |

Get the missing magic numbers:
```bash
# Measure from existing layer 4 dir
python scripts/measure_feature_stats.py \
    --features_dir ./data/coco_features_layer4 --split train

# Measure all 4 layers at once (loads Qwen, ~10 min)
python scripts/measure_feature_stats.py \
    --data_dir ./data/coco --split train --max_files 100 --realtime --all_layers
```

### Precompute Each Layer

```bash
# Layer 0: patch_embed output
python scripts/precompute_qwen_features.py \
    --data_dir ./data/coco --output_dir ./data/coco_features_layer0 --layer 0 --split train
python scripts/precompute_qwen_features.py \
    --data_dir ./data/coco --output_dir ./data/coco_features_layer0 --layer 0 --split val

# Layer 8
python scripts/precompute_qwen_features.py \
    --data_dir ./data/coco --output_dir ./data/coco_features_layer8 --layer 8 --split train
python scripts/precompute_qwen_features.py \
    --data_dir ./data/coco --output_dir ./data/coco_features_layer8 --layer 8 --split val

# Layer -1 (pixel patches, no full Qwen forward pass needed)
python scripts/precompute_qwen_features.py \
    --data_dir ./data/coco --output_dir ./data/coco_features_layer-1 --layer -1 --split train
python scripts/precompute_qwen_features.py \
    --data_dir ./data/coco --output_dir ./data/coco_features_layer-1 --layer -1 --split val
```

---

## 📐 Architecture (v2.2)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ EDGE                                      [edge_client.py]              │
│  Image → MobileNet → Projector → Bottleneck.encode()                    │
│                           ↓                                              │
│              [49 × 64] int8 quantized → base64 encoded                  │
└───────────────────────────────────────┬─────────────────────────────────┘
                                        │ HTTP POST (~3 KB payload)
┌───────────────────────────────────────▼─────────────────────────────────┐
│ CLOUD                                     [cloud_server.py]             │
│  Flask Server @ :8080                                                    │
│  Dequantize → Bottleneck.decode() → Upsampler → Qwen[4:] → LLM         │
│                           ↓                                              │
│              JSON Response: {"response": "Image description..."}        │
└─────────────────────────────────────────────────────────────────────────┘
```

### Edge vs Cloud Comparison

| | Edge | Cloud |
|---|---|---|
| **Components** | MobileNetV2 + StridedProjector + Bottleneck.encoder | Bottleneck.decoder + TransformerUpsampler |
| **Weight File** | 11 MB | 486 MB |
| **Active Params** | 2.87M | 126.63M |
| **Extra Compute** | - | Qwen ViT [4:32] + Merger + LLM |

---

## 📝 Key Arguments

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--bottleneck_dim` | 0 | Bottleneck dimension (recommended: 64/128) |
| `--bottleneck_method` | linear | linear / mlp / autoencoder |
| `--lambda_recon` | 0.1 | Reconstruction loss weight |
| `--upsampler_type` | transformer | transformer / mlp / deconv |
| `--phase` | - | warmup (MSE) / gan (adversarial) |

### Deployment Arguments

| Argument | Description |
|----------|-------------|
| `--offline` | Offline mode, no HuggingFace connection |
| `--qwen_path` | Qwen model path |
| `--timeout` | Request timeout in seconds |

---

## 📊 Transmission Size Comparison

| bottleneck_dim | Size (int8) | Compression |
|----------------|-------------|-------------|
| Disabled (1280) | 61 KB | 1× |
| 128 | 6.1 KB | 10× |
| **64** | **3.1 KB** | **20×** |
| 32 | 1.5 KB | 40× |

---

## 📄 License

MIT License

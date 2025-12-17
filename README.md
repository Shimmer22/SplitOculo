# SplitOculo

<p align="center">
  <b>Edge-Cloud Collaborative Vision Feature Splitting Framework for Smart Glasses</b>
</p>

<p align="center">
  <a href="README-zh.md">中文文档</a>
</p>

---

## 📖 Overview

SplitOculo is a research framework for **edge-cloud collaborative computing** on smart glasses. It enables:

- 🎓 **Feature Distillation**: Train CNN to approximate Qwen2.5-VL features
- 🔗 **Learnable Upsampler**: Cloud-side upsampling for transmission efficiency
- 📊 **Hybrid Inference**: Edge CNN + Cloud Qwen deep layers

---

## 🏗️ Project Structure

```
SplitOculo/
├── scripts/
│   ├── precompute_qwen_features.py   # Pre-extract Qwen Layer 4 features
│   ├── train_with_upsampler.py       # Train CNN + Projector + Upsampler
│   ├── infer_hybrid.py               # Hybrid edge-cloud inference
│   └── plot_training.py              # Visualize training curves
│
├── models/
│   └── cloud_upsampler.py            # CloudUpsampler module
│
├── core/                   # Core utilities
├── data/                   # Data utilities
└── checkpoints/            # Training outputs
```

---

## 🚀 Quick Start

### 1. Pre-compute Qwen Features (one-time)

```bash
python scripts/precompute_qwen_features.py \
    --data_dir ./data/imagenette2-320 \
    --layer 4 --split train

python scripts/precompute_qwen_features.py \
    --data_dir ./data/imagenette2-320 \
    --layer 4 --split val
```

### 2. Train with Upsampler

```bash
python scripts/train_with_upsampler.py \
    --features_dir ./data/qwen_features \
    --data_dir ./data/imagenette2-320 \
    --transmission_tokens 49 \
    --target_tokens 256 \
    --epochs 100
```

### 3. Inference

```bash
python scripts/infer_hybrid.py \
    --checkpoint checkpoints/upsampler/best_model.pth \
    --image photo.jpg \
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
│  Learned Upsampler → 256 tokens → Qwen[4:] → Merger → LLM   │
│       [Trained]                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Training Results

| Metric | Value |
|--------|-------|
| Epochs | 50 |
| Val Cos Sim | 0.8732 |
| Transmission | 49 tokens (~61 KB) |

### Visualize Training

```bash
python scripts/plot_training.py --log checkpoints/upsampler/train.log
```

---

## 📝 Key Arguments

### train_with_upsampler.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--transmission_tokens` | 49 | Tokens sent from edge (7×7) |
| `--target_tokens` | 256 | Target tokens for Qwen (16×16) |
| `--upsampler_method` | deconv | deconv/pixelshuffle/transformer |
| `--student_model` | mobilenetv2_100 | CNN backbone (timm models) |

### infer_hybrid.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | - | Path to upsampler checkpoint |
| `--image` | - | Input image path |
| `--full_inference` | False | Run complete Qwen inference |

---

## 📄 License

MIT License
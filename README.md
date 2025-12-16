# SplitOculo

<p align="center">
  <b>Edge-Cloud Collaborative Vision Feature Splitting Framework for Smart Glasses</b>
</p>

<p align="center">
  <a href="README-zh.md">中文文档</a>
</p>

---

## 📖 Overview

SplitOculo is a research framework for **edge-cloud collaborative computing** on smart glasses. It supports:

- 📊 **Benchmark**: Evaluate feature size and FLOPs at different split points
- 🎓 **Feature Distillation**: Train CNN models to approximate CLIP ViT outputs
- 🔗 **Qwen2.5-VL Alignment**: Align CNN features with Qwen2.5-VL for multimodal tasks

---

## 🏗️ Project Structure

```
SplitOculo/
├── main_benchmark.py       # Benchmark entry point
│
├── scripts/                # Training & Inference scripts
│   ├── train_distill.py              # CLIP distillation training
│   ├── precompute_qwen_features.py   # Pre-extract Qwen features
│   ├── train_with_precomputed.py     # Train with Qwen features
│   ├── infer.py                      # CLIP-aligned inference
│   ├── infer_qwen.py                 # Qwen-aligned inference
│   └── infer_hybrid.py               # Hybrid CNN+Qwen inference
│
├── core/                   # Core framework
├── models/                 # Model definitions
├── data/                   # Data utilities
├── checkpoints/            # Training checkpoints
└── results/                # Benchmark outputs
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Qwen2.5-VL Alignment (Recommended)

**Step 1: Pre-compute Qwen features (supports intermediate layers)**
```bash
# Extract Layer 8 features (shallow, easier to learn)
python scripts/precompute_qwen_features.py \
    --data_dir ./data/imagenette2-320 \
    --layer 8 \
    --split train

python scripts/precompute_qwen_features.py \
    --data_dir ./data/imagenette2-320 \
    --layer 8 \
    --split val
```

**Step 2: Train with precomputed features**
```bash
python scripts/train_with_precomputed.py \
    --features_dir ./data/qwen_features \
    --target_hidden_size 1280 \
    --epochs 100
```

### 3. Inference

```bash
# Edge-side encoding only (fast, no Qwen needed)
python scripts/infer_hybrid.py \
    --checkpoint checkpoints/qwen_precomputed/best_model.pth \
    --image photo.jpg

# With int8 quantization for smaller transmission
python scripts/infer_hybrid.py \
    --checkpoint checkpoints/qwen_precomputed/best_model.pth \
    --image photo.jpg \
    --quantize int8

# Full hybrid inference (edge CNN + cloud Qwen deep layers)
python scripts/infer_hybrid.py \
    --checkpoint checkpoints/qwen_precomputed/best_model.pth \
    --image photo.jpg \
    --full_inference
```

---

## 📐 Architecture

### Shallow Layer Alignment (Layer 8)

```
Qwen ViT Layer 8 (1280 dim) ────────┐
    [Offline Precomputed]            │
                                     │──→  MSE + Cosine Loss
CNN (96ch) ──→ Projector (1280 dim) ─┘
    [Trainable]
```

### Hybrid Inference Pipeline

```
         ┌─────────────────────────────────────────────────────┐
Edge:    │ Image → CNN → Projector → features (1280 dim, int8) │
         └─────────────────────┬───────────────────────────────┘
                               │ ~61 KB transmission
         ┌─────────────────────▼───────────────────────────────┐
Cloud:   │ features → Qwen Blocks[8:] → Merger → LLM → Response │
         └─────────────────────────────────────────────────────┘
```

---

## 📊 Training Results

| Target | Val Cos Sim | Val MSE | Learnable? |
|--------|-------------|---------|------------|
| Layer 8 (1280) | **0.77** | 1.07 | ✅ Yes |
| Merger (2048) | ~0.00 | ~4.8 | ❌ No |

---

## 📝 Key Arguments

### Precompute
| Argument | Default | Description |
|----------|---------|-------------|
| `--layer` | 8 | ViT layer (1-32, or -1 for merger) |
| `--split` | train | train or val |
| `--resume` | False | Resume from checkpoint |

### Training
| Argument | Default | Description |
|----------|---------|-------------|
| `--target_hidden_size` | 1280 | Target dim (1280 for layer, 2048 for merger) |
| `--epochs` | 100 | Training epochs |
| `--batch_size` | 64 | Batch size |

### Inference
| Argument | Default | Description |
|----------|---------|-------------|
| `--image` | - | Path to input image |
| `--quantize` | int8 | none/fp16/int8 |
| `--full_inference` | False | Run Qwen deep layers on cloud |

---

## 📚 References

- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2412.00015)
- [MobileVLM V2](https://arxiv.org/abs/2402.03766)
- [CLIP](https://arxiv.org/abs/2103.00020)

---

## 📄 License

MIT License
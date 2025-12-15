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
│   └── infer_qwen.py                 # Qwen-aligned inference
│
├── core/                   # Core framework
│   ├── framework.py        # BaseSplitModel, ExperimentRunner
│   └── utils.py            # FLOPs counter, seed, logger
│
├── models/                 # Model definitions
│   ├── mobilenet_v2.py
│   ├── mobile_vit.py
│   ├── levit.py
│   └── adapters.py         # Feature alignment adapters
│
├── data/                   # Data utilities
│   └── dataset.py          # ImageNet/Dummy loaders
│
├── checkpoints/            # Training checkpoints
└── results/                # Benchmark outputs
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Benchmark

```bash
python main_benchmark.py
```

### 3. CLIP Feature Distillation

```bash
# Quick test with dummy data
python scripts/train_distill.py --dummy --epochs 5

# Full training with ImageNet
python scripts/train_distill.py \
    --data_dir /path/to/imagenet \
    --epochs 100 --batch_size 64
```

### 4. Qwen2.5-VL Alignment (Recommended for VLM)

**Step 1: Pre-compute Qwen features (one-time, supports resume)**
```bash
# Extract features for all images
python scripts/precompute_qwen_features.py \
    --data_dir ./data/imagenette2-320 \
    --output_dir ./data/qwen_features \
    --split train

python scripts/precompute_qwen_features.py \
    --data_dir ./data/imagenette2-320 \
    --output_dir ./data/qwen_features \
    --split val

# Resume from checkpoint if interrupted
python scripts/precompute_qwen_features.py --resume ...
```

**Step 2: Train with precomputed features (10-50x faster)**
```bash
python scripts/train_with_precomputed.py \
    --features_dir ./data/qwen_features \
    --data_dir ./data/imagenette2-320 \
    --epochs 100 --batch_size 64
```

### 5. Inference

```bash
# CLIP-aligned inference
python scripts/infer.py --checkpoint checkpoints/best_model.pth --image photo.jpg

# Qwen-aligned inference (edge-side visual encoding)
python scripts/infer_qwen.py --checkpoint checkpoints/qwen_precomputed/best_model.pth --image photo.jpg
```

---

## 📐 Architecture

### CLIP Distillation

```
Teacher (CLIP ViT)  ────────┐
    [Frozen]                │
                            │──→  MSE + Cosine Loss
Student (MobileNetV2) ──→ Adapter ──┘
    [Trainable]
```

### Qwen2.5-VL Alignment

```
Qwen ViT+Merger (2048 dim) ─────────┐
    [Offline Precomputed]           │
                                    │──→  MSE + Cosine Loss
CNN (96ch) ──→ LLMProjector (2048) ─┘
    [Trainable]
```

---

## 📊 Supported Models

| Model | Params | GFLOPs | Usage |
|-------|--------|--------|-------|
| MobileNetV2 | 1.8M | 0.56 | Student backbone |
| CLIP ViT-L/14 | 304M | 81.0 | CLIP teacher |
| Qwen2.5-VL 3B | 3B | - | VLM teacher |

---

## 📝 Key Arguments

### Precompute Features
| Argument | Default | Description |
|----------|---------|-------------|
| `--max_samples` | None | Limit samples (for testing) |
| `--resume` | False | Resume from checkpoint |
| `--split` | train | Which split to process |

### Training
| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 100 | Training epochs |
| `--batch_size` | 64 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--cos_weight` | 0.5 | Cosine loss weight |
| `--llm_hidden_size` | 2048 | Qwen LLM hidden size |

---

## 📚 References

- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2412.00015)
- [MobileVLM V2](https://arxiv.org/abs/2402.03766)
- [CLIP](https://arxiv.org/abs/2103.00020)

---

## 📄 License

MIT License
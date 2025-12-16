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
# Extract Layer 4 features (shallow, easier to learn)
python scripts/precompute_qwen_features.py \
    --data_dir ./data/imagenette2-320 \
    --layer 4 \
    --split train

python scripts/precompute_qwen_features.py \
    --data_dir ./data/imagenette2-320 \
    --layer 4 \
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

> **⚠️ Important**: The `--split_layer` parameter in inference must match the `--layer` used during feature precomputation!

```bash
# Edge-side encoding only (fast, no Qwen needed)
python scripts/infer_hybrid.py \
    --checkpoint checkpoints/qwen_precomputed/best_model.pth \
    --image photo.jpg

# Full hybrid inference (edge CNN + cloud Qwen deep layers)
python scripts/infer_hybrid.py \
    --checkpoint checkpoints/qwen_precomputed/best_model.pth \
    --image photo.jpg \
    --split_layer 4 \
    --full_inference

# With int8 quantization for smaller transmission
python scripts/infer_hybrid.py \
    --checkpoint checkpoints/qwen_precomputed/best_model.pth \
    --image photo.jpg \
    --split_layer 4 \
    --quantize int8 \
    --full_inference
```

---

## 📐 Architecture

### Shallow Layer Alignment (Layer 4)

```
Qwen ViT Layer 4 (1280 dim) ────────┐
    [Offline Precomputed]            │
                                     │──→  MSE + Cosine Loss
CNN (96ch) ──→ Projector (1280 dim) ─┘
    [Trainable]
```

### Hybrid Inference Pipeline

```
         ┌─────────────────────────────────────────────────────┐
Edge:    │ Image → CNN → Projector → 49 tokens (7×7, 1280 dim) │
         └─────────────────────┬───────────────────────────────┘
                               │ ~61 KB (int8) transmission
         ┌─────────────────────▼───────────────────────────────┐
Cloud:   │ Upsample 49→256 → Scale → Qwen[4:] → Merger → LLM   │
         └─────────────────────────────────────────────────────┘
```

---

## 📊 Training Results

| Target | Val Cos Sim | Val MSE | Status |
|--------|-------------|---------|--------|
| Layer 4 (1280) | **0.86** | 0.63 | ⚠️ Pipeline works, semantic quality needs improvement |
| Layer 8 (1280) | ~0.77 | 1.07 | ✅ Learnable |
| Merger (2048) | ~0.00 | ~4.8 | ❌ Too Hard |

### Quality Analysis

- **cos_sim=0.86**: Pipeline works, but **semantic information is lost**
- **Required**: cos_sim > 0.95 for semantic correctness, or end-to-end fine-tuning
- **Ground truth test**: Qwen Layer 4 features produce correct output, CNN features don't

### Transmission Size Analysis

| Format | Size | Notes |
|--------|------|-------|
| JPEG 224×224 (Q85) | ~13 KB | Baseline |
| Edge features (int8) | ~61 KB | 49 tokens × 1280 dim |
| Edge features (fp16) | ~123 KB | Higher precision |

> Note: Current int8 features are larger than JPEG. Future work: add entropy coding or reduce token count.

---

## ⚠️ Known Issues

### Feature Quality Gap
Current training achieves 0.86 cosine similarity, which is **insufficient for semantic correctness**. The pipeline is verified to work correctly - when using Qwen's actual Layer 4 features, the LLM produces correct responses.

**To improve quality, consider:**
- Train with more diverse/larger datasets
- Use deeper or wider CNN architecture
- Try end-to-end fine-tuning of Qwen blocks[4:]
- Target cos_sim > 0.95

### Layer Mismatch
The `--split_layer` parameter in `infer_hybrid.py` must match the `--layer` used during `precompute_qwen_features.py`. A mismatch will cause incorrect outputs.

### Feature Scale Mismatch
The CNN Projector outputs features with a different scale than Qwen's intermediate layers. The inference script automatically scales features to match Qwen's expected distribution (std≈0.83, mean≈-0.017).

### Token Upsampling (Cloud-side)
CNN outputs 49 tokens (7×7 grid), but Qwen expects 256 tokens (16×16). Bilinear upsampling is performed **on the cloud side** to save transmission bandwidth.

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
| `--split_layer` | 4 | Must match training layer |
| `--quantize` | none | none/fp16/int8 |
| `--full_inference` | False | Run Qwen deep layers on cloud |

---

## 📚 References

- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2412.00015)
- [MobileVLM V2](https://arxiv.org/abs/2402.03766)
- [CLIP](https://arxiv.org/abs/2103.00020)

---

## 📄 License

MIT License
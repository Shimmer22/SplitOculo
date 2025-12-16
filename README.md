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
```

---

## 📐 Architecture

### Current Architecture

```
         ┌─────────────────────────────────────────────────────┐
Edge:    │ Image → CNN → Projector → 49 tokens (7×7, 1280 dim) │
         └─────────────────────┬───────────────────────────────┘
                               │ ~61 KB (int8) transmission
         ┌─────────────────────▼───────────────────────────────┐
Cloud:   │ Bilinear Upsample 49→256 → Qwen[4:] → Merger → LLM  │
         └─────────────────────────────────────────────────────┘
```

### Projector Role

The **Projector** is a lightweight CNN module that:
1. Transforms CNN feature channels (96ch → 1280 dim)
2. Downsamples spatial resolution (14×14 → 7×7) via AvgPool
3. Converts to token sequence for Qwen compatibility

**Current upsampling is parameter-free** (bilinear interpolation), which is the bottleneck (see experiments below).

---

## 🔬 Ablation Experiments

We conducted systematic experiments to identify the semantic loss bottleneck:

| Experiment | Input | Output | Result |
|------------|-------|--------|--------|
| A. Teacher 256 tokens (no downsample) | Qwen Layer 4 directly | "person wearing hat with feather" | ✅ Correct |
| B. Precomputed .pt 256 tokens | From training data | "fishing net with fish" | ✅ Correct |
| C. Teacher 256→49→256 (down+upsample) | Downsampled then upsampled | "textured fabric, diagonal stripes" | ❌ Wrong |
| D. CNN 49 tokens + upsample | CNN output | "pixelated pattern" / "black" | ❌ Wrong |

### Key Findings

1. **Pipeline is correct**: Experiments A & B prove blocks[4:] + merger + LLM work correctly
2. **Upsampling is the bottleneck**: Experiment C shows even teacher features lose semantics after 256→49→256
3. **49 tokens insufficient**: 7×7 spatial resolution cannot preserve image semantics
4. **CNN quality secondary**: The 0.86 cos_sim CNN isn't the main issue; upsampling is

### Implication

To fix semantic loss, we need:
- **Option 1**: Transmit 256 tokens (larger bandwidth, ~320 KB int8)
- **Option 2**: Train a **learnable upsampler** on cloud side (recommended)
- **Option 3**: Reduce CNN downsampling to output 256 tokens directly

---

## 📊 Training Results

| Target | Val Cos Sim | Val MSE | Status |
|--------|-------------|---------|--------|
| Layer 4 (1280) | **0.86** | 0.63 | ⚠️ Training OK, upsampling bottleneck |
| Layer 8 (1280) | ~0.77 | 1.07 | ✅ Learnable |
| Merger (2048) | ~0.00 | ~4.8 | ❌ Too Hard |

### Transmission Size

| Format | Size | Notes |
|--------|------|-------|
| JPEG 224×224 (Q85) | ~13 KB | Baseline |
| 49 tokens (int8) | ~61 KB | Current, semantic loss |
| 256 tokens (int8) | ~320 KB | No semantic loss |

---

## 🔮 Future Work: Learnable Upsampler

A promising direction is training a **learnable upsampler** on the cloud side:

```
         ┌─────────────────────────────────────────────────────┐
Edge:    │ Image → CNN → Projector → 49 tokens (7×7, 1280 dim) │
         └─────────────────────┬───────────────────────────────┘
                               │ ~61 KB transmission
         ┌─────────────────────▼───────────────────────────────┐
Cloud:   │ Learned Upsampler 49→256 → Qwen[4:] → Merger → LLM  │
         │     [Trainable]                                      │
         └─────────────────────────────────────────────────────┘
```

The upsampler can be trained end-to-end with frozen Qwen blocks or jointly.

---

## 📝 Key Arguments

### Precompute
| Argument | Default | Description |
|----------|---------|-------------|
| `--layer` | 8 | ViT layer (1-32, or -1 for merger) |
| `--split` | train | train or val |

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
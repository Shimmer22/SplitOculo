# SplitOculo

<p align="center">
  <b>Edge-Cloud Collaborative Vision Feature Splitting Framework for Smart Glasses</b>
</p>

<p align="center">
  <a href="README-zh.md">中文文档</a>
</p>

---

## 📖 Overview

SplitOculo is a research framework for **edge-cloud collaborative computing** on smart glasses. It helps find optimal feature split points for computation offloading and supports training lightweight CNN models to mimic ViT feature extractors through **feature distillation**.

### Key Features

- 📊 **Benchmark**: Evaluate feature size and FLOPs at different split points
- 🎓 **Feature Distillation**: Train CNN models to approximate CLIP ViT outputs
- 📈 **Visualization**: Auto-generate Size-Compute tradeoff and Pareto analysis plots

---

## 🏗️ Project Structure

```
SplitOculo/
├── main_benchmark.py       # Benchmark entry point
├── train_distill.py        # Distillation training entry
├── infer.py                # Inference with trained model
├── requirements.txt
│
├── core/                   # Core framework
│   ├── framework.py        # BaseSplitModel, ExperimentRunner
│   └── utils.py            # FLOPs counter, seed, logger
│
├── models/                 # Model definitions
│   ├── mobilenet_v2.py
│   ├── mobile_vit.py
│   ├── levit.py
│   ├── mobile_vlm.py
│   └── adapters.py         # Feature alignment adapters
│
├── data/                   # Data utilities
│   └── dataset.py          # ImageNet/Dummy loaders
│
├── results/                # Benchmark outputs
└── checkpoints/            # Training checkpoints
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

Example output:
```
================================================================================
                        Experiment Report
================================================================================

📌 MobileNetV2
   Total Params: 1.81M
   Total GFLOPs: 0.559
----------------------------------------------------------------------
       Split_Point Resolution  Channels  Size_KB  Cumulative_GFLOPs  Is_Viable
  Stride 4 (Local)      56x56        24    73.50              0.140      False
    Stride 8 (Mid)      28x28        32    24.50              0.279       True  ✅
   Stride 16 (Rec)      14x14        96    18.38              0.419       True  ✅
Stride 32 (Global)        7x7       320    15.31              0.559       True  ✅

================================================================================
✅ Viable split points (Size < 30KB): 6 / 15
🏆 Optimal split: MobileNetV2 - Stride 8 (Mid)
   Size: 24.50 KB, GFLOPs: 0.279
================================================================================
```

### 3. Feature Distillation Training

**Quick test with dummy data:**
```bash
python train_distill.py --dummy --epochs 5
```

**Full training with ImageNet:**
```bash
python train_distill.py \
    --data_dir /path/to/imagenet \
    --teacher_model vit_large_patch14_clip_224 \
    --student_model mobilenetv2_100 \
    --epochs 100 \
    --batch_size 64
```

### 4. Inference with Trained Model

```bash
# Test with dummy data
python infer.py --checkpoint checkpoints/best_model.pth --dummy

# Run on real image
python infer.py --checkpoint checkpoints/best_model.pth --image path/to/image.jpg

# Save features to file
python infer.py --checkpoint checkpoints/best_model.pth --image photo.jpg --output features.pt
```

Output:
```
🔧 Device: cuda
✅ Loaded checkpoint: checkpoints/best_model.pth
✅ Feature shape: torch.Size([1, 1024, 16, 16])
   Feature range: [-1.6851, 1.5349]
   Feature mean: 0.0167
```

---

## 📐 Distillation Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Teacher       │     │   Student       │
│  (CLIP ViT)     │     │  (MobileNetV2)  │
│   [Frozen]      │     │   [Trainable]   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         │ feat_t                │ feat_s
         │ (1024, 16, 16)        │ (96, 14, 14)
         │                       │
         │               ┌───────┴───────┐
         │               │   Adapter     │
         │               │   (1x1 Conv)  │
         │               └───────┬───────┘
         │                       │ adapted
         │                       │ (1024, 16, 16)
         │                       │
         └───────────┬───────────┘
                     │
              ┌──────┴──────┐
              │    Loss     │
              │  MSE + Cos  │
              └─────────────┘
```

**Loss Functions:**
- **MSE Loss**: Mean squared error of feature values
- **Cosine Loss**: Cosine similarity of feature directions

---

## 🔧 Adding New Models

Simply create a new file with the `@register_model` decorator:

```python
# models/my_model.py
from core.framework import BaseSplitModel, register_model

@register_model
class MyModel(BaseSplitModel):
    def load_model(self):
        # Load your model
        pass
    
    def get_features_at_splits(self, x):
        # Return features at each split point
        return [feat1, feat2, feat3]
```

Models are automatically discovered and registered.

---

## 📊 Supported Models

| Model | Params | GFLOPs | Source |
|-------|--------|--------|--------|
| MobileNetV2 | 1.8M | 0.56 | timm |
| MobileViT | 4.9M | 2.83 | timm |
| LeViT | 18.9M | 2.10 | timm |
| MobileVLM V2 | 209M | 62.7 | Simulated |

---

## 📝 Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--teacher_model` | `vit_large_patch14_clip_224` | Teacher model name |
| `--student_model` | `mobilenetv2_100` | Student model name |
| `--teacher_layer` | 3 | Teacher feature layer |
| `--student_layer` | 3 | Student feature layer |
| `--epochs` | 100 | Training epochs |
| `--batch_size` | 64 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--cos_weight` | 0.5 | Cosine loss weight |

---

## 📚 References

- [MobileVLM V2: Faster and Stronger Baseline for Vision Language Model](https://arxiv.org/abs/2402.03766)
- [CLIP: Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020)
- [timm: PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)

---

## 📄 License

MIT License
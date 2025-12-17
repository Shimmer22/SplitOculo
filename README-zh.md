# SplitOculo

<p align="center">
  <b>VLM 边端-云端协同视觉特征切分</b>
</p>

<p align="center">
  <a href="README.md">English</a>
</p>

---

## 📖 概述

SplitOculo 实现 **边端-云端协同 VLM 推理**：
- 🖥️ **端侧**: CNN + Projector → 49 tokens (61 KB)
- ☁️ **云端**: 可学习上采样器 → 256 tokens → Qwen → LLM

---

## 🚀 快速开始 (直接复制执行)

### 步骤 0: 环境配置

```bash
# 克隆仓库
git clone https://github.com/Shimmer22/SplitOculo.git
cd SplitOculo

# 创建 conda 环境
conda create -n cnn_vit python=3.10 -y
conda activate cnn_vit

# 安装依赖
pip install torch torchvision transformers timm tqdm pillow matplotlib
```

### 步骤 1: 下载数据集

推荐使用 **COCO val2017** (5000 张多样化图像, ~1GB):

```bash
# 创建数据目录
mkdir -p data/coco

# 下载 COCO val2017
wget http://images.cocodataset.org/zips/val2017.zip -P data/coco/
unzip data/coco/val2017.zip -d data/coco/
rm data/coco/val2017.zip

# 划分训练集/验证集 (80/20)
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

print(f'✅ 训练集: {split} 张, 验证集: {len(images)-split} 张')
"
```

### 步骤 2: 预计算 Qwen 特征

```bash
# 预计算训练集特征 (GPU 上约 30 分钟)
python scripts/precompute_qwen_features.py \
    --data_dir ./data/coco \
    --output_dir ./data/qwen_features \
    --layer 4 \
    --split train \
    --batch_size 4

# 预计算验证集特征
python scripts/precompute_qwen_features.py \
    --data_dir ./data/coco \
    --output_dir ./data/qwen_features \
    --layer 4 \
    --split val \
    --batch_size 4
```

### 步骤 3: 使用 MLP 上采样器训练

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

### 步骤 4: 可视化训练

```bash
python scripts/plot_training.py --log ./checkpoints/coco_mlp/train.log
```

### 步骤 5: 推理

```bash
python scripts/infer_hybrid.py \
    --checkpoint ./checkpoints/coco_mlp/best_model.pth \
    --image ./data/coco/val/000000000139.jpg \
    --full_inference
```

---

## 📐 架构

```
┌─────────────────────────────────────────────────────────────┐
│ 端侧 (EDGE)                                                  │
├─────────────────────────────────────────────────────────────┤
│  图像 → CNN → Projector → 49 tokens (7×7, 1280 维)           │
└────────────────────────┬────────────────────────────────────┘
                         │ 传输 (~61 KB int8)
┌────────────────────────▼────────────────────────────────────┐
│ 云端 (CLOUD)                                                 │
├─────────────────────────────────────────────────────────────┤
│  MLP 上采样 → 256 tokens → Qwen[4:] → Merger → LLM           │
│    [Bilinear + MLP]                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 训练结果

| 数据集 | Epochs | Val cos_sim | 上采样器 |
|--------|--------|-------------|----------|
| Imagenette | 50 | 0.87 | deconv |
| Imagenette | 50 | **待测** | mlp |
| COCO | 100 | **待测** | mlp |

---

## 📝 关键参数

### train_with_upsampler.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--upsampler_method` | **mlp** | mlp (最佳) / deconv / transformer |
| `--transmission_tokens` | 49 | 端侧 tokens (7×7) |
| `--target_tokens` | 256 | Qwen 目标 (16×16) |
| `--student_model` | mobilenetv2_100 | CNN 骨干网络 |
| `--epochs` | 100 | 训练轮数 |

### infer_hybrid.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | - | 训练模型路径 |
| `--image` | - | 输入图像路径 |
| `--full_inference` | False | 运行完整 Qwen 推理 |

---

## 🔬 关键发现

| 方法 | cos_sim | 语义理解 |
|------|---------|----------|
| 纯 Bilinear | 0.87 | ❌ 错误 |
| deconv + BN | 0.57 | ❌ 错误 |
| **MLP (bilinear + mlp)** | **0.999** | ✅ 正确 |

**根本原因**: deconv + BatchNorm 破坏信息。使用 `--upsampler_method mlp`。

---

## 📄 许可证

MIT License

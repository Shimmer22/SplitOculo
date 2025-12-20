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
- 🖥️ **端侧**: CNN + Projector + Bottleneck → 3 KB 压缩特征
- ☁️ **云端**: 解压 + 上采样 → 256 tokens → Qwen → LLM

---

## 🆕 v2.2 更新: 真实网络分离部署

> [!TIP]
> **v2.2 支持端侧和云端通过 HTTP 真实分离部署！**

### 新特性
- **网络分离**: `cloud_server.py` + `edge_client.py` 实现真实的端云分离
- **静态权重拆分**: 使用 `split_checkpoint.py` 将 AIO 权重拆分为端侧 (~11 MB) 和云端 (~486 MB)
- **瓶颈层压缩**: 61 KB → 3 KB，压缩比 20×
- **离线模式**: `--offline` 标志支持无网络加载 Qwen

---

## 🚀 快速开始

### 步骤 0: 环境配置

```bash
git clone https://github.com/Shimmer22/SplitOculo.git
cd SplitOculo

conda create -n cnn_vit python=3.10 -y
conda activate cnn_vit

pip install -r requirements.txt
```

### 步骤 1: 下载数据集 (COCO val2017)

```bash
mkdir -p data/coco
wget http://images.cocodataset.org/zips/val2017.zip -P data/coco/
unzip data/coco/val2017.zip -d data/coco/
rm data/coco/val2017.zip

# 划分 train/val (80/20)
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

print(f'✅ 训练集: {split}, 验证集: {len(images)-split}')
"
```

### 步骤 2: 预计算 Qwen 特征

```bash
python scripts/precompute_qwen_features.py \
    --data_dir ./data/coco --output_dir ./data/qwen_features \
    --layer 4 --split train --batch_size 4

python scripts/precompute_qwen_features.py \
    --data_dir ./data/coco --output_dir ./data/qwen_features \
    --layer 4 --split val --batch_size 4
```

### 步骤 3: 训练 (启用瓶颈层)

```bash
# Phase 1: Warmup
python scripts/train_gan.py \
    --features_dir ./data/qwen_features \
    --data_dir ./data/coco \
    --phase warmup \
    --epochs 20 \
    --bottleneck_dim 64 \
    --bottleneck_method linear \
    --output_dir ./checkpoints/gan_bottleneck

# Phase 2: GAN 微调
python scripts/train_gan.py \
    --features_dir ./data/qwen_features \
    --data_dir ./data/coco \
    --phase gan \
    --warmup_checkpoint ./checkpoints/gan_bottleneck/warmup_best.pth \
    --epochs 30 \
    --bottleneck_dim 64 \
    --output_dir ./checkpoints/gan_bottleneck
```

### 步骤 4: 拆分权重

```bash
python scripts/split_checkpoint.py \
    --input ./checkpoints/gan_bottleneck/gan_best.pth \
    --output_dir ./checkpoints/gan_bottleneck/split/
```

输出：
- `edge_weights.pth` (~11 MB): CNN + Projector + Bottleneck.encoder
- `cloud_weights.pth` (~486 MB): Bottleneck.decoder + Upsampler

### 步骤 5: 网络分离部署

**云端服务器**:
```bash
python scripts/cloud_server.py \
    --checkpoint ./checkpoints/gan_bottleneck/split/cloud_weights.pth \
    --port 8080 \
    --offline
```

**端侧客户端**:
```bash
python scripts/edge_client.py \
    --checkpoint ./checkpoints/gan_bottleneck/split/edge_weights.pth \
    --image ./test.jpg \
    --server http://云端IP:8080 \
    --timeout 300
```

---

## 📐 架构 (v2.2)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 端侧 (EDGE)                               [edge_client.py]              │
│  图像 → MobileNet → Projector → Bottleneck.encode()                     │
│                           ↓                                              │
│              [49 × 64] int8 量化 → base64 编码                          │
└───────────────────────────────────────┬─────────────────────────────────┘
                                        │ HTTP POST (~3 KB payload)
┌───────────────────────────────────────▼─────────────────────────────────┐
│ 云端 (CLOUD)                              [cloud_server.py]             │
│  Flask Server @ :8080                                                    │
│  反量化 → Bottleneck.decode() → Upsampler → Qwen[4:] → LLM             │
│                           ↓                                              │
│              JSON Response: {"response": "图片描述..."}                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📝 关键参数

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--bottleneck_dim` | 0 | 瓶颈层维度 (推荐 64/128) |
| `--bottleneck_method` | linear | linear / mlp / autoencoder |
| `--lambda_recon` | 0.1 | 瓶颈层重建损失权重 |
| `--upsampler_type` | transformer | transformer / mlp / deconv |
| `--phase` | - | warmup (MSE) / gan (对抗) |

### 部署参数

| 参数 | 说明 |
|------|------|
| `--offline` | 离线模式，不连接 HuggingFace |
| `--qwen_path` | Qwen 模型路径 |
| `--timeout` | 请求超时时间 (秒) |

---

## 📊 传输大小对比

| bottleneck_dim | 传输大小 (int8) | 压缩比 |
|----------------|-----------------|--------|
| 禁用 (1280) | 61 KB | 1× |
| 128 | 6.1 KB | 10× |
| **64** | **3.1 KB** | **20×** |
| 32 | 1.5 KB | 40× |

---

## 📄 许可证

MIT License


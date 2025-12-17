# SplitOculo

<p align="center">
  <b>智能眼镜边端-云端协同视觉特征切分框架</b>
</p>

<p align="center">
  <a href="README.md">English</a>
</p>

---

## 📖 概述

SplitOculo 是一个用于智能眼镜**边端-云端协同计算**的研究框架，支持：

- 🎓 **特征蒸馏**：训练 CNN 逼近 Qwen2.5-VL 特征
- 🔗 **可学习上采样器**：云端上采样以提升传输效率
- 📊 **混合推理**：端侧 CNN + 云端 Qwen 深层

---

## 🏗️ 项目结构

```
SplitOculo/
├── scripts/
│   ├── precompute_qwen_features.py   # 预提取 Qwen Layer 4 特征
│   ├── train_with_upsampler.py       # 训练 CNN + Projector + Upsampler
│   ├── infer_hybrid.py               # 混合端云推理
│   └── plot_training.py              # 可视化训练曲线
│
├── models/
│   └── cloud_upsampler.py            # CloudUpsampler 模块
│
├── core/                   # 核心工具
├── data/                   # 数据工具
└── checkpoints/            # 训练输出
```

---

## 🚀 快速开始

### 1. 预计算 Qwen 特征（一次性）

```bash
python scripts/precompute_qwen_features.py \
    --data_dir ./data/imagenette2-320 \
    --layer 4 --split train

python scripts/precompute_qwen_features.py \
    --data_dir ./data/imagenette2-320 \
    --layer 4 --split val
```

### 2. 训练上采样器

```bash
python scripts/train_with_upsampler.py \
    --features_dir ./data/qwen_features \
    --data_dir ./data/imagenette2-320 \
    --transmission_tokens 49 \
    --target_tokens 256 \
    --epochs 100
```

### 3. 推理

```bash
python scripts/infer_hybrid.py \
    --checkpoint checkpoints/upsampler/best_model.pth \
    --image photo.jpg \
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
│  可学习上采样 → 256 tokens → Qwen[4:] → Merger → LLM         │
│       [已训练]                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 训练结果

| 指标 | 值 |
|------|-----|
| Epochs | 50 |
| Val Cos Sim | 0.8732 |
| 传输大小 | 49 tokens (~61 KB) |

### 可视化训练

```bash
python scripts/plot_training.py --log checkpoints/upsampler/train.log
```

---

## 📝 关键参数

### train_with_upsampler.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--transmission_tokens` | 49 | 端侧传输 tokens (7×7) |
| `--target_tokens` | 256 | Qwen 目标 tokens (16×16) |
| `--upsampler_method` | deconv | deconv/pixelshuffle/transformer |
| `--student_model` | mobilenetv2_100 | CNN 骨干网络 (timm 模型) |

### infer_hybrid.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | - | 上采样器检查点路径 |
| `--image` | - | 输入图像路径 |
| `--full_inference` | False | 运行完整 Qwen 推理 |

---

## 📄 许可证

MIT License

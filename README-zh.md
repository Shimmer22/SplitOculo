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

- 📊 **性能评估**：评估不同切分点的特征尺寸和计算量
- 🎓 **特征蒸馏**：训练 CNN 模型逼近 CLIP ViT 输出
- 🔗 **Qwen2.5-VL 对齐**：将 CNN 特征对齐到 Qwen2.5-VL 用于多模态任务

---

## 🏗️ 项目结构

```
SplitOculo/
├── main_benchmark.py       # 性能评估入口
│
├── scripts/                # 训练和推理脚本
│   ├── train_distill.py              # CLIP 蒸馏训练
│   ├── precompute_qwen_features.py   # 预提取 Qwen 特征
│   ├── train_with_precomputed.py     # 使用预计算特征训练
│   ├── infer.py                      # CLIP 对齐推理
│   ├── infer_qwen.py                 # Qwen 对齐推理
│   └── infer_hybrid.py               # 混合 CNN+Qwen 推理
│
├── core/                   # 核心框架
├── models/                 # 模型定义
├── data/                   # 数据工具
├── checkpoints/            # 训练检查点
└── results/                # 评估输出
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. Qwen2.5-VL 对齐（推荐）

**步骤 1：预计算 Qwen 特征（支持中间层）**
```bash
# 提取第 8 层特征（浅层，易学习）
python scripts/precompute_qwen_features.py \
    --data_dir ./data/imagenette2-320 \
    --layer 8 \
    --split train

python scripts/precompute_qwen_features.py \
    --data_dir ./data/imagenette2-320 \
    --layer 8 \
    --split val
```

**步骤 2：使用预计算特征训练**
```bash
python scripts/train_with_precomputed.py \
    --features_dir ./data/qwen_features \
    --target_hidden_size 1280 \
    --epochs 100
```

### 3. 推理

```bash
# 仅端侧编码（快速，无需 Qwen）
python scripts/infer_hybrid.py \
    --checkpoint checkpoints/qwen_precomputed/best_model.pth \
    --image photo.jpg

# 使用 int8 量化减少传输大小
python scripts/infer_hybrid.py \
    --checkpoint checkpoints/qwen_precomputed/best_model.pth \
    --image photo.jpg \
    --quantize int8

# 完整混合推理（端侧 CNN + 云端 Qwen 深层）
python scripts/infer_hybrid.py \
    --checkpoint checkpoints/qwen_precomputed/best_model.pth \
    --image photo.jpg \
    --full_inference
```

---

## 📐 架构

### 浅层对齐（Layer 8）

```
Qwen ViT 第8层 (1280 维) ────────────┐
    [离线预计算]                      │
                                     │──→  MSE + Cosine Loss
CNN (96通道) ──→ Projector (1280 维) ─┘
    [可训练]
```

### 混合推理流水线

```
         ┌─────────────────────────────────────────────────────┐
端侧:    │ 图像 → CNN → Projector → 特征 (1280 维, int8)        │
         └─────────────────────┬───────────────────────────────┘
                               │ ~61 KB 传输
         ┌─────────────────────▼───────────────────────────────┐
云端:    │ 特征 → Qwen Blocks[8:] → Merger → LLM → 回复         │
         └─────────────────────────────────────────────────────┘
```

---

## 📊 训练结果

| 目标 | Val Cos Sim | Val MSE | 可学习？ |
|------|-------------|---------|----------|
| Layer 8 (1280) | **0.77** | 1.07 | ✅ 是 |
| Merger (2048) | ~0.00 | ~4.8 | ❌ 否 |

---

## 📝 关键参数

### 预计算
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--layer` | 8 | ViT 层 (1-32, 或 -1 表示 merger) |
| `--split` | train | train 或 val |
| `--resume` | False | 断点续传 |

### 训练
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--target_hidden_size` | 1280 | 目标维度 (1280 中间层, 2048 merger) |
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 64 | 批大小 |

### 推理
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--image` | - | 输入图片路径 |
| `--quantize` | int8 | none/fp16/int8 |
| `--full_inference` | False | 在云端运行 Qwen 深层 |

---

## 📚 参考文献

- [Qwen2.5-VL 技术报告](https://arxiv.org/abs/2412.00015)
- [MobileVLM V2](https://arxiv.org/abs/2402.03766)
- [CLIP](https://arxiv.org/abs/2103.00020)

---

## 📄 许可证

MIT License

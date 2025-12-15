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
│   └── infer_qwen.py                 # Qwen 对齐推理
│
├── core/                   # 核心框架
│   ├── framework.py        # BaseSplitModel, ExperimentRunner
│   └── utils.py            # FLOPs 计算, 随机种子, 日志
│
├── models/                 # 模型定义
│   ├── mobilenet_v2.py
│   ├── mobile_vit.py
│   ├── levit.py
│   └── adapters.py         # 特征对齐适配器
│
├── data/                   # 数据工具
│   └── dataset.py          # ImageNet/虚拟数据加载器
│
├── checkpoints/            # 训练检查点
└── results/                # 评估输出
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行性能评估

```bash
python main_benchmark.py
```

### 3. CLIP 特征蒸馏

```bash
# 使用假数据快速测试
python scripts/train_distill.py --dummy --epochs 5

# 使用 ImageNet 完整训练
python scripts/train_distill.py \
    --data_dir /path/to/imagenet \
    --epochs 100 --batch_size 64
```

### 4. Qwen2.5-VL 对齐（推荐用于 VLM）

**步骤 1：预计算 Qwen 特征（一次性，支持断点续传）**
```bash
# 提取所有图片的特征
python scripts/precompute_qwen_features.py \
    --data_dir ./data/imagenette2-320 \
    --output_dir ./data/qwen_features \
    --split train

python scripts/precompute_qwen_features.py \
    --data_dir ./data/imagenette2-320 \
    --output_dir ./data/qwen_features \
    --split val

# 断点续传
python scripts/precompute_qwen_features.py --resume ...
```

**步骤 2：使用预计算特征训练（快 10-50 倍）**
```bash
python scripts/train_with_precomputed.py \
    --features_dir ./data/qwen_features \
    --data_dir ./data/imagenette2-320 \
    --epochs 100 --batch_size 64
```

### 5. 推理

```bash
# CLIP 对齐推理
python scripts/infer.py --checkpoint checkpoints/best_model.pth --image photo.jpg

# Qwen 对齐推理（端侧视觉编码）
python scripts/infer_qwen.py --checkpoint checkpoints/qwen_precomputed/best_model.pth --image photo.jpg
```

---

## 📐 架构

### CLIP 蒸馏

```
Teacher (CLIP ViT)  ────────┐
    [冻结]                   │
                            │──→  MSE + Cosine Loss
Student (MobileNetV2) ──→ Adapter ──┘
    [可训练]
```

### Qwen2.5-VL 对齐

```
Qwen ViT+Merger (2048 维) ──────────┐
    [离线预计算]                     │
                                    │──→  MSE + Cosine Loss
CNN (96通道) ──→ LLMProjector (2048) ─┘
    [可训练]
```

---

## 📊 支持的模型

| 模型 | 参数量 | GFLOPs | 用途 |
|------|--------|--------|------|
| MobileNetV2 | 1.8M | 0.56 | Student 骨干网络 |
| CLIP ViT-L/14 | 304M | 81.0 | CLIP Teacher |
| Qwen2.5-VL 3B | 3B | - | VLM Teacher |

---

## 📝 关键参数

### 预计算特征
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_samples` | None | 限制样本数（测试用） |
| `--resume` | False | 断点续传 |
| `--split` | train | 处理哪个数据集分割 |

### 训练
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 64 | 批大小 |
| `--lr` | 1e-4 | 学习率 |
| `--cos_weight` | 0.5 | Cosine 损失权重 |
| `--llm_hidden_size` | 2048 | Qwen LLM 隐藏维度 |

---

## 📚 参考文献

- [Qwen2.5-VL 技术报告](https://arxiv.org/abs/2412.00015)
- [MobileVLM V2](https://arxiv.org/abs/2402.03766)
- [CLIP](https://arxiv.org/abs/2103.00020)

---

## 📄 许可证

MIT License

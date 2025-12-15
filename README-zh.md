# SplitOculo

<p align="center">
  <b>面向智能眼镜的端云协同视觉特征切分框架</b>
  <br>
  <i>Edge-Cloud Collaborative Vision Feature Splitting for Smart Glasses</i>
</p>

<p align="center">
  <a href="README.md">English</a>
</p>

---

## 📖 项目简介

本项目用于研究**端侧计算卸载**场景下的最优特征切分点选择，并支持将轻量级 CNN 模型通过**特征蒸馏**训练成类 ViT 的特征提取器。

### 核心功能

- 📊 **Benchmark**: 评估不同模型在各切分点的特征大小 (Size) 和计算量 (FLOPs)
- 🎓 **特征蒸馏**: 训练 CNN 模型逼近 CLIP ViT 的输出特征
- 📈 **可视化**: 自动生成 Size-Compute 权衡图和帕累托分析

---

## 🏗️ 项目结构

```
cnn_vit/
├── main_benchmark.py       # 性能评估入口
├── train_distill.py        # 蒸馏训练入口
├── infer.py                # 推理脚本
├── requirements.txt        # 依赖库
│
├── core/                   # 核心框架
│   ├── framework.py        # BaseSplitModel, ExperimentRunner
│   └── utils.py            # FLOPs计算, 随机种子, Logger
│
├── models/                 # 模型定义
│   ├── mobilenet_v2.py     # MobileNetV2
│   ├── mobile_vit.py       # MobileViT
│   ├── levit.py            # LeViT
│   ├── mobile_vlm.py       # MobileVLM V2 (模拟)
│   └── adapters.py         # 特征对齐适配器
│
├── data/                   # 数据处理
│   └── dataset.py          # ImageNet/Dummy 数据加载
│
├── results/                # Benchmark 输出
└── checkpoints/            # 训练检查点
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行 Benchmark

```bash
python main_benchmark.py
```

输出示例：
```
================================================================================
                        实验结果报告
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
✅ 可行切分点 (Size < 30KB): 6 / 15
🏆 最优切分点: MobileNetV2 - Stride 8 (Mid)
   Size: 24.50 KB, GFLOPs: 0.279
================================================================================
```

### 3. 特征蒸馏训练

**快速测试 (使用假数据):**
```bash
python train_distill.py --dummy --epochs 5
```

**完整训练 (使用 ImageNet):**
```bash
python train_distill.py \
    --data_dir /path/to/imagenet \
    --teacher_model vit_large_patch14_clip_224 \
    --student_model mobilenetv2_100 \
    --student_layer 3 \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-4
```

### 4. 使用训练好的模型推理

```bash
# 快速测试
python infer.py --checkpoint checkpoints/best_model.pth --dummy

# 对真实图像推理
python infer.py --checkpoint checkpoints/best_model.pth --image path/to/image.jpg

# 保存特征到文件
python infer.py --checkpoint checkpoints/best_model.pth --image photo.jpg --output features.pt
```

输出：
```
🔧 Device: cuda
✅ 已加载检查点: checkpoints/best_model.pth
✅ 特征形状: torch.Size([1, 1024, 16, 16])
   特征范围: [-1.6851, 1.5349]
   特征均值: 0.0167
```

---

## 📐 蒸馏训练原理

```
┌─────────────────┐     ┌─────────────────┐
│   Teacher       │     │   Student       │
│  (CLIP ViT)     │     │  (MobileNetV2)  │
│   [冻结]        │     │   [可训练]      │
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
              │  Loss       │
              │  MSE + Cos  │
              └─────────────┘
```

**损失函数:**
- **MSE Loss**: 特征值的均方误差
- **Cosine Loss**: 特征方向的余弦相似度

---

## 🔧 添加新模型

只需创建新文件并使用 `@register_model` 装饰器：

```python
# models/my_model.py
from core.framework import BaseSplitModel, register_model

@register_model
class MyModel(BaseSplitModel):
    def load_model(self):
        # 加载模型
        pass
    
    def get_features_at_splits(self, x):
        # 返回各切分点特征
        return [feat1, feat2, feat3]
```

模型会被自动发现和注册，无需修改其他文件。

---

## 📊 支持的模型

| 模型 | 参数量 | GFLOPs | 来源 |
|------|--------|--------|------|
| MobileNetV2 | 1.8M | 0.56 | timm |
| MobileViT | 4.9M | 2.83 | timm |
| LeViT | 18.9M | 2.10 | timm |
| MobileVLM V2 | 209M | 62.7 | 模拟实现 |

---

## 📝 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--teacher_model` | `vit_large_patch14_clip_224` | Teacher 模型名 |
| `--student_model` | `mobilenetv2_100` | Student 模型名 |
| `--teacher_layer` | 3 | Teacher 特征提取层 |
| `--student_layer` | 3 | Student 特征提取层 |
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 64 | 批次大小 |
| `--lr` | 1e-4 | 学习率 |
| `--cos_weight` | 0.5 | Cosine Loss 权重 |

---

## 📚 参考文献

- [MobileVLM V2: Faster and Stronger Baseline for Vision Language Model](https://arxiv.org/abs/2402.03766)
- [CLIP: Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020)
- [timm: PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)

---

## 📄 License

MIT License

"""
SplitOculo v2.0 GAN Training Script

训练流程:
- Phase 1 (Warmup): 只用 MSE Loss 预热，让 Upsampler 学会基本的上采样
- Phase 2 (GAN): Generator vs Discriminator 对抗训练，让特征更 sharp

支持两种模式:
- Static (默认): 从预计算的 .pt 文件加载特征，训练快，显存低
- Dynamic: 训练时实时计算 Qwen 特征，无需预计算，但显存需求高

Usage:
    # Static 模式 (默认) - Phase 1: Warmup
    python scripts/train_gan.py \
        --features_dir ./data/qwen_features \
        --data_dir ./data/coco \
        --phase warmup \
        --epochs 20

    # Static 模式 - Phase 2: GAN finetuning
    python scripts/train_gan.py \
        --features_dir ./data/qwen_features \
        --data_dir ./data/coco \
        --phase gan \
        --warmup_checkpoint ./checkpoints/gan/warmup_best.pth \
        --epochs 50

    # Dynamic 模式 (无需预计算) - Phase 1: Warmup
    python scripts/train_gan.py \
        --dynamic \
        --data_dir ./data/coco \
        --qwen_model Qwen/Qwen2.5-VL-3B-Instruct \
        --qwen_layer 4 \
        --phase warmup \
        --batch_size 8 \
        --epochs 20
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from core.qwen_extractor import QwenFeatureExtractor
from core.utils import count_parameters, get_logger, set_seed
from models.bottleneck import DimensionBottleneck
from models.budgeted_transmission import SoftBudgetedTransmission
from models.cloud_upsampler import (
    CloudUpsampler,
    SparseTokenUpsampler,
    TransformerUpsampler,
)
from models.discriminator import FeatureDiscriminator
from models.importance_scorer import TextAwareImportanceScorer, TokenImportanceScorer

# PoolingTokenProjector is the inline EdgeProjector class (pooling_projector.py was removed)
# The 'pooling' projector_type now uses EdgeProjector defined below
from models.strided_projector import StridedTokenProjector


def parse_bool_flag(value):
    """Parse bool-like CLI values for flags that can be explicitly enabled/disabled."""
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


class EdgeProjector(nn.Module):
    """端侧 Projector: CNN 特征 → 传输 tokens"""

    def __init__(
        self, in_channels, hidden_size=1280, hidden_channels=512, transmission_tokens=49
    ):
        super().__init__()

        self.transmission_size = int(math.sqrt(transmission_tokens))
        assert self.transmission_size**2 == transmission_tokens

        self.pw_conv1 = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act1 = nn.GELU()

        self.pool = nn.AdaptiveAvgPool2d(
            (self.transmission_size, self.transmission_size)
        )

        self.dw_conv = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
            groups=hidden_channels,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.act2 = nn.GELU()

        self.pw_conv2 = nn.Conv2d(
            hidden_channels, hidden_size, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(hidden_size)

    def forward(self, x):
        x = self.act1(self.bn1(self.pw_conv1(x)))
        x = self.pool(x)
        residual = x
        x = self.act2(self.bn2(self.dw_conv(x)))
        x = x + residual
        x = self.bn3(self.pw_conv2(x))
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x


class PrecomputedFeatureDataset(Dataset):
    """加载预计算的 Qwen 特征和对应的原始图像"""

    def __init__(self, features_dir, images_dir, split="train", image_size=224):
        self.features_dir = Path(features_dir) / split
        self.images_dir = Path(images_dir) / split

        metadata_path = self.features_dir / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"找不到元数据: {metadata_path}")

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.feature_files = sorted(self.features_dir.glob("*.pt"))
        print(f"Found {len(self.feature_files)} precomputed features")

        from torchvision import transforms

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    image_size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        feature_data = torch.load(self.feature_files[idx], weights_only=False)
        teacher_features = feature_data["features"]
        img_path = feature_data["path"]

        from PIL import Image

        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        return img_tensor, teacher_features


class DynamicFeatureDataset(Dataset):
    """动态模式: 训练时实时计算 Qwen 特征

    无需预计算，但需要更多显存和计算时间
    """

    def __init__(self, images_dir, extractor, split="train", image_size=224):
        """
        Args:
            images_dir: 图像目录
            extractor: QwenFeatureExtractor 实例
            split: train 或 val
            image_size: 输入图像大小
        """
        self.images_dir = Path(images_dir) / split
        self.extractor = extractor

        # 扫描图像文件
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.image_files = sorted(
            [
                f
                for f in self.images_dir.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
        )
        print(f"Found {len(self.image_files)} images in {self.images_dir}")

        from torchvision import transforms

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    image_size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Qwen 预处理 (只做 resize + crop,不做 normalize)
        self.qwen_transform = transforms.Compose(
            [
                transforms.Resize(
                    image_size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(image_size),
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        from PIL import Image

        pil_img = Image.open(img_path).convert("RGB")

        # CNN 输入
        img_tensor = self.transform(pil_img)

        # Qwen 特征 (实时计算)
        qwen_img = self.qwen_transform(pil_img)
        teacher_features = self.extractor.extract_features(qwen_img)

        return img_tensor, teacher_features


def collate_fn(batch):
    """处理不同长度的特征"""
    images = torch.stack([item[0] for item in batch])
    features = [item[1] for item in batch]
    min_tokens = min(f.shape[0] for f in features)
    aligned_features = torch.stack([f[:min_tokens] for f in features])
    return images, aligned_features


class GANTrainer:
    """SplitOculo v2.0 GAN 训练器"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.logger = get_logger("gan_train", log_file=f"{args.output_dir}/train.log")

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._load_models()
        self._setup_optimizers()

        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        # AMP (Automatic Mixed Precision)
        self.use_amp = getattr(args, "amp", False) and self.device.type == "cuda"
        if self.use_amp:
            self.scaler_G = torch.amp.GradScaler("cuda")
            self.scaler_D = torch.amp.GradScaler("cuda")
        else:
            self.scaler_G = None
            self.scaler_D = None

        # 记录参数量
        self._log_run_config()
        self._log_model_params()

    def _log_run_config(self):
        """Log full CLI args so newly added arguments are always visible in train.log."""
        self.logger.info("Run config:")
        for key in sorted(vars(self.args)):
            self.logger.info(f"  {key}: {getattr(self.args, key)}")

        self.logger.info(
            f"Runtime: device={self.device.type}, amp_requested={self.args.amp}, amp_enabled={self.use_amp}"
        )

    def _log_model_params(self):
        """Log parameter counts from actual optimizer groups and module-level breakdown."""
        g_params = sum(
            p.numel()
            for group in self.opt_G.param_groups
            for p in group["params"]
            if p.requires_grad
        )
        d_params = sum(
            p.numel()
            for group in self.opt_D.param_groups
            for p in group["params"]
            if p.requires_grad
        )

        self.logger.info(f"Generator params: {g_params / 1e6:.2f}M")
        self.logger.info(f"Discriminator params: {d_params / 1e6:.2f}M")

        module_keys = [
            "student",
            "projector",
            "bottleneck",
            "upsampler",
            "importance_scorer",
            "budgeted_tx",
            "sparse_upsampler",
            "discriminator",
        ]
        self.logger.info("Model components:")
        for key in module_keys:
            module = getattr(self, key, None)
            if module is None:
                self.logger.info(f"  {key}: disabled")
                continue
            self.logger.info(f"  {key}: {count_parameters(module) / 1e6:.2f}M")

        if self.bottleneck is not None:
            size_kb = self.bottleneck.get_transmission_size_kb(
                self.args.transmission_tokens
            )
            self.logger.info(f"Transmission size: {size_kb:.2f} KB")

    def _update_hard_topk_ratio(self, epoch):
        """Progressively increase train-time hard top-k ratio to reduce train/infer gap."""
        if self.budgeted_tx is None:
            return
        if not getattr(self.args, "topk_train_schedule", False):
            self.budgeted_tx.set_train_hard_ratio(0.0)
            return

        start = max(1, int(getattr(self.args, "topk_start_epoch", 1)))
        end = int(getattr(self.args, "topk_end_epoch", self.args.epochs))
        end = max(start, end)
        if epoch <= start:
            ratio = 0.0
        elif epoch >= end:
            ratio = 1.0
        else:
            ratio = (epoch - start) / float(end - start)

        self.budgeted_tx.set_train_hard_ratio(ratio)
        self.logger.info(f"  Hard-topk ratio: {ratio:.3f}")

    def _load_models(self):
        """加载模型"""
        args = self.args

        # CNN backbone
        self.logger.info(f"Loading student: {args.student_model}")
        self.student = timm.create_model(
            args.student_model,
            pretrained=True,
            features_only=True,
            out_indices=[args.student_layer],
        ).to(self.device)

        # 获取输出通道数
        with torch.no_grad():
            dummy = torch.randn(1, 3, args.image_size, args.image_size).to(self.device)
            student_feat = self.student(dummy)[-1]
            student_channels = student_feat.shape[1]

        self.logger.info(f"Student output: {student_channels} channels")

        # 端侧 Projector
        if args.projector_type == "strided":
            self.projector = StridedTokenProjector(
                in_channels=student_channels,
                hidden_size=args.target_hidden_size,
                hidden_channels=args.projector_hidden,
                transmission_tokens=args.transmission_tokens,
            ).to(self.device)
            self.logger.info("Using strided token projector")
        else:
            self.projector = EdgeProjector(
                in_channels=student_channels,
                hidden_size=args.target_hidden_size,
                hidden_channels=args.projector_hidden,
                transmission_tokens=args.transmission_tokens,
            ).to(self.device)
            self.logger.info("Using pooling (edge) projector")

        # 云端 Upsampler (使用 TransformerUpsampler)
        if args.upsampler_type == "transformer":
            self.upsampler = TransformerUpsampler(
                hidden_size=args.target_hidden_size,
                input_tokens=args.transmission_tokens,
                target_tokens=args.target_tokens,
                num_layers=args.transformer_layers,
                initial_upsample=args.initial_upsample,
            ).to(self.device)
            self.logger.info(
                f"Using TransformerUpsampler with {args.transformer_layers} layers ({args.initial_upsample})"
            )
        else:
            self.upsampler = CloudUpsampler(
                hidden_size=args.target_hidden_size,
                input_tokens=args.transmission_tokens,
                target_tokens=args.target_tokens,
                method=args.upsampler_type,
            ).to(self.device)
            self.logger.info(f"Using CloudUpsampler ({args.upsampler_type})")

        # 瓶颈层 (可选)
        if args.bottleneck_dim > 0:
            self.bottleneck = DimensionBottleneck(
                hidden_size=args.target_hidden_size,
                bottleneck_dim=args.bottleneck_dim,
                method=args.bottleneck_method,
                quantize_aware=getattr(args, "quantize_aware", False),
                num_bits=getattr(args, "num_bits", 8),
            ).to(self.device)
            self.logger.info(
                f"Using Bottleneck: {args.target_hidden_size} → {args.bottleneck_dim} → {args.target_hidden_size}"
            )
        else:
            self.bottleneck = None
            self.logger.info("No bottleneck (full dimension transmission)")

        # Importance scorer (for information-aware transmission)
        if args.importance_aware:
            if args.text_aware:
                self.importance_scorer = TextAwareImportanceScorer(
                    cnn_channels=student_channels,
                    hidden_size=args.target_hidden_size,
                    spatial_size=student_feat.shape[2],  # 14 for MobileNetV2
                    token_grid_size=int(args.transmission_tokens**0.5),
                ).to(self.device)
                self.logger.info(
                    f"Using TextAwareImportanceScorer (CNN {student_channels}ch + semantic)"
                )
            else:
                self.importance_scorer = TokenImportanceScorer(
                    hidden_size=args.target_hidden_size,
                    method=args.scorer_method,
                ).to(self.device)
                self.logger.info(
                    f"Using TokenImportanceScorer (method={args.scorer_method})"
                )

            # Budgeted transmission
            self.budgeted_tx = SoftBudgetedTransmission(
                max_tokens=args.transmission_tokens,
                target_budget=args.token_budget,
                initial_temperature=args.budget_temperature,
                min_temperature=0.1,
                anneal_rate=args.anneal_rate,
                min_tokens=args.min_tokens,
            ).to(self.device)
            self.logger.info(
                f"Using SoftBudgetedTransmission (budget={args.token_budget}, temp={args.budget_temperature})"
            )

            # Use SparseTokenUpsampler instead of regular upsampler
            self.sparse_upsampler = SparseTokenUpsampler(
                hidden_size=args.target_hidden_size,
                max_tokens=args.transmission_tokens,
                target_tokens=args.target_tokens,
                num_completion_layers=args.completion_layers,
                num_upsample_layers=args.transformer_layers,
                initial_upsample=args.initial_upsample,
            ).to(self.device)
            self.logger.info(
                f"Using SparseTokenUpsampler (completion_layers={args.completion_layers})"
            )
        else:
            self.importance_scorer = None
            self.budgeted_tx = None
            self.sparse_upsampler = None

        # Discriminator
        self.discriminator = FeatureDiscriminator(
            hidden_size=args.target_hidden_size, num_tokens=args.target_tokens
        ).to(self.device)

        self.logger.info(
            f"Upsampler: {args.transmission_tokens} → {args.target_tokens} tokens"
        )

    def _setup_optimizers(self):
        """设置优化器"""
        args = self.args

        # Generator 参数
        g_params = (
            list(self.student.parameters())
            + list(self.projector.parameters())
            + list(self.upsampler.parameters())
        )

        # 添加瓶颈层参数
        if self.bottleneck is not None:
            g_params += list(self.bottleneck.parameters())

        # Add importance-aware module parameters
        if self.importance_scorer is not None:
            g_params += list(self.importance_scorer.parameters())
        if self.sparse_upsampler is not None:
            g_params += list(self.sparse_upsampler.parameters())

        # Generator 优化器
        self.opt_G = optim.AdamW(
            g_params, lr=args.lr_g, weight_decay=args.weight_decay, betas=(0.5, 0.9)
        )

        # Discriminator 优化器 (学习率通常低一些)
        self.opt_D = optim.AdamW(
            self.discriminator.parameters(),
            lr=args.lr_d,
            weight_decay=args.weight_decay,
            betas=(0.5, 0.9),
        )

        # 学习率调度器
        self.scheduler_G = optim.lr_scheduler.CosineAnnealingLR(
            self.opt_G, T_max=args.epochs
        )
        self.scheduler_D = optim.lr_scheduler.CosineAnnealingLR(
            self.opt_D, T_max=args.epochs
        )

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.student.load_state_dict(ckpt["student_state_dict"])
        self.projector.load_state_dict(ckpt["projector_state_dict"])
        self.upsampler.load_state_dict(ckpt["upsampler_state_dict"])

        if "bottleneck_state_dict" in ckpt and self.bottleneck is not None:
            self.bottleneck.load_state_dict(ckpt["bottleneck_state_dict"])
            self.logger.info("Loaded bottleneck weights")

        if (
            "importance_scorer_state_dict" in ckpt
            and self.importance_scorer is not None
        ):
            self.importance_scorer.load_state_dict(ckpt["importance_scorer_state_dict"])
            self.logger.info("Loaded importance scorer weights")

        if "budgeted_tx_state_dict" in ckpt and self.budgeted_tx is not None:
            self.budgeted_tx.load_state_dict(ckpt["budgeted_tx_state_dict"])
            self.logger.info("Loaded budgeted transmission weights")

        if "sparse_upsampler_state_dict" in ckpt and self.sparse_upsampler is not None:
            self.sparse_upsampler.load_state_dict(ckpt["sparse_upsampler_state_dict"])
            self.logger.info("Loaded sparse upsampler weights")

        if "discriminator_state_dict" in ckpt:
            self.discriminator.load_state_dict(ckpt["discriminator_state_dict"])

        self.logger.info(f"Loaded checkpoint (epoch {ckpt.get('epoch', 'unknown')})")
        return ckpt.get("epoch", 0)

    def _align_tokens(self, tokens, target_num):
        """对齐 token 数量"""
        b, n, c = tokens.shape
        h = w = int(n**0.5)
        tokens = tokens.view(b, h, w, c).permute(0, 3, 1, 2)
        target_h = target_w = int(target_num**0.5)
        tokens = F.adaptive_avg_pool2d(tokens, (target_h, target_w))
        tokens = tokens.permute(0, 2, 3, 1).view(b, -1, c)
        return tokens

    def train_epoch_warmup(self, dataloader, epoch):
        """Warmup 训练: 只用 MSE Loss"""
        self.student.train()
        self.projector.train()
        self.upsampler.train()
        if self.importance_scorer is not None:
            self.importance_scorer.train()
        if self.budgeted_tx is not None:
            self.budgeted_tx.train()
        if self.sparse_upsampler is not None:
            self.sparse_upsampler.train()

        total_loss = 0
        total_cos_sim = 0
        total_budget_loss = 0
        total_entropy_loss = 0
        total_effective_k = 0

        pbar = tqdm(dataloader, desc=f"Warmup Epoch {epoch}")
        for images, teacher_tokens in pbar:
            images = images.to(self.device)
            teacher_tokens = teacher_tokens.to(self.device).float()

            self.opt_G.zero_grad()

            # Forward with AMP
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                student_feat = self.student(images)[-1]
                edge_tokens = self.projector(student_feat)

                if self.importance_scorer is not None:
                    # Information-aware path
                    # Score importance
                    if self.args.text_aware and isinstance(
                        self.importance_scorer, TextAwareImportanceScorer
                    ):
                        importance_logits, imp_details = self.importance_scorer(
                            student_feat, edge_tokens
                        )
                    else:
                        importance_logits = self.importance_scorer(edge_tokens)
                        imp_details = {}

                    # Budgeted transmission (soft masking in training)
                    masked_tokens, mask, budget_loss, entropy_loss = self.budgeted_tx(
                        edge_tokens, importance_logits
                    )

                    # Bottleneck compression
                    if self.bottleneck is not None:
                        decompressed, compressed = self.bottleneck(masked_tokens)
                    else:
                        decompressed = masked_tokens
                        compressed = None

                    # Sparse upsampler (in training mode, receives soft-masked full sequence)
                    # During training, use forward_dense since we have all 49 tokens (just soft-masked)
                    output_tokens = self.sparse_upsampler.forward_dense(decompressed)
                else:
                    # Original path (no importance scoring)
                    if self.bottleneck is not None:
                        decompressed, compressed = self.bottleneck(edge_tokens)
                        output_tokens = self.upsampler(decompressed)
                    else:
                        output_tokens = self.upsampler(edge_tokens)
                        compressed = None
                        decompressed = None
                    budget_loss = torch.tensor(0.0, device=self.device)
                    entropy_loss = torch.tensor(0.0, device=self.device)
                    mask = None

                # Align token counts
                if output_tokens.shape[1] != teacher_tokens.shape[1]:
                    teacher_tokens = self._align_tokens(
                        teacher_tokens, output_tokens.shape[1]
                    )

                # MSE Loss
                loss = self.mse_loss(output_tokens, teacher_tokens)

                # Bottleneck reconstruction loss
                if self.bottleneck is not None and self.args.lambda_recon > 0:
                    if self.importance_scorer is not None:
                        loss_recon = self.mse_loss(decompressed, masked_tokens.detach())
                    else:
                        loss_recon = self.mse_loss(decompressed, edge_tokens.detach())
                    loss = loss + self.args.lambda_recon * loss_recon

                # Budget & entropy losses (importance-aware mode only)
                if self.importance_scorer is not None:
                    loss = loss + self.args.lambda_budget * budget_loss
                    loss = loss + self.args.lambda_entropy * entropy_loss

            # Backward with AMP
            if self.use_amp:
                self.scaler_G.scale(loss).backward()
                self.scaler_G.step(self.opt_G)
                self.scaler_G.update()
            else:
                loss.backward()
                self.opt_G.step()

            # 计算 cos_sim
            with torch.no_grad():
                cos_sim = F.cosine_similarity(
                    output_tokens.reshape(-1, self.args.target_hidden_size),
                    teacher_tokens.reshape(-1, self.args.target_hidden_size),
                    dim=-1,
                ).mean()

            total_loss += loss.item()
            total_cos_sim += cos_sim.item()
            total_budget_loss += budget_loss.item()
            total_entropy_loss += entropy_loss.item()
            if mask is not None:
                total_effective_k += mask.sum(dim=1).mean().item()

            postfix = {"loss": f"{loss.item():.4f}", "cos_sim": f"{cos_sim.item():.4f}"}
            if mask is not None:
                postfix["eff_k"] = f"{mask.sum(dim=1).mean().item():.1f}"
            pbar.set_postfix(postfix)

        n = len(dataloader)
        metrics = {"loss": total_loss / n, "cos_sim": total_cos_sim / n}
        if self.importance_scorer is not None:
            metrics["budget_loss"] = total_budget_loss / n
            metrics["entropy_loss"] = total_entropy_loss / n
            metrics["effective_k"] = total_effective_k / n
            metrics["temperature"] = self.budgeted_tx.current_temperature
        return metrics

    def train_epoch_gan(self, dataloader, epoch):
        """GAN 训练: Generator vs Discriminator"""
        self.student.train()
        self.projector.train()
        self.upsampler.train()
        self.discriminator.train()
        if self.importance_scorer is not None:
            self.importance_scorer.train()
        if self.budgeted_tx is not None:
            self.budgeted_tx.train()
        if self.sparse_upsampler is not None:
            self.sparse_upsampler.train()

        total_loss_g = 0
        total_loss_d = 0
        total_cos_sim = 0
        total_mse = 0
        total_budget_loss = 0
        total_entropy_loss = 0
        total_effective_k = 0

        pbar = tqdm(dataloader, desc=f"GAN Epoch {epoch}")
        for images, teacher_tokens in pbar:
            images = images.to(self.device)
            teacher_tokens = teacher_tokens.to(self.device).float()

            # 对齐 token 数量
            if teacher_tokens.shape[1] != self.args.target_tokens:
                teacher_tokens = self._align_tokens(
                    teacher_tokens, self.args.target_tokens
                )

            # ===================
            # 1. Train Discriminator
            # ===================
            self.opt_D.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                # 真实样本
                pred_real = self.discriminator(teacher_tokens)
                label_real = torch.ones_like(pred_real)
                loss_d_real = self.bce_loss(pred_real, label_real)

                # 假样本 (detach 防止梯度传回 G)
                with torch.no_grad():
                    student_feat = self.student(images)[-1]
                    edge_tokens = self.projector(student_feat)

                    if self.importance_scorer is not None:
                        # Information-aware path for D
                        if self.args.text_aware and isinstance(
                            self.importance_scorer, TextAwareImportanceScorer
                        ):
                            importance_logits, _ = self.importance_scorer(
                                student_feat, edge_tokens
                            )
                        else:
                            importance_logits = self.importance_scorer(edge_tokens)
                        masked_tokens, _, _, _ = self.budgeted_tx(
                            edge_tokens, importance_logits
                        )
                        if self.bottleneck is not None:
                            decompressed, _ = self.bottleneck(masked_tokens)
                        else:
                            decompressed = masked_tokens
                        fake_tokens_d = self.sparse_upsampler.forward_dense(
                            decompressed
                        )
                    else:
                        if self.bottleneck is not None:
                            decompressed, _ = self.bottleneck(edge_tokens)
                            edge_tokens = decompressed
                        fake_tokens_d = self.upsampler(edge_tokens)

                pred_fake = self.discriminator(fake_tokens_d.detach())
                label_fake = torch.zeros_like(pred_fake)
                loss_d_fake = self.bce_loss(pred_fake, label_fake)

                loss_d = (loss_d_real + loss_d_fake) / 2

            if self.use_amp:
                self.scaler_D.scale(loss_d).backward()
                self.scaler_D.step(self.opt_D)
                self.scaler_D.update()
            else:
                loss_d.backward()
                self.opt_D.step()

            # ===================
            # 2. Train Generator
            # ===================
            self.opt_G.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                # 重新生成 (带梯度)
                student_feat = self.student(images)[-1]
                edge_tokens_raw = self.projector(student_feat)

                if self.importance_scorer is not None:
                    # Information-aware path
                    if self.args.text_aware and isinstance(
                        self.importance_scorer, TextAwareImportanceScorer
                    ):
                        importance_logits, imp_details = self.importance_scorer(
                            student_feat, edge_tokens_raw
                        )
                    else:
                        importance_logits = self.importance_scorer(edge_tokens_raw)
                        imp_details = {}

                    masked_tokens, mask, budget_loss, entropy_loss = self.budgeted_tx(
                        edge_tokens_raw, importance_logits
                    )

                    if self.bottleneck is not None:
                        decompressed, compressed = self.bottleneck(masked_tokens)
                    else:
                        decompressed = masked_tokens
                        compressed = None

                    fake_tokens = self.sparse_upsampler.forward_dense(decompressed)
                else:
                    # Original path
                    if self.bottleneck is not None:
                        decompressed, compressed = self.bottleneck(edge_tokens_raw)
                        fake_tokens = self.upsampler(decompressed)
                    else:
                        fake_tokens = self.upsampler(edge_tokens_raw)
                        decompressed = None
                    budget_loss = torch.tensor(0.0, device=self.device)
                    entropy_loss = torch.tensor(0.0, device=self.device)
                    mask = None

                # 内容损失 (MSE)
                loss_mse = self.mse_loss(fake_tokens, teacher_tokens)

                # 对抗损失 (GAN) - 骗过 D
                pred_fake = self.discriminator(fake_tokens)
                loss_adv = self.bce_loss(pred_fake, torch.ones_like(pred_fake))

                # 重建损失
                loss_recon = 0
                if self.bottleneck is not None and self.args.lambda_recon > 0:
                    if self.importance_scorer is not None:
                        loss_recon = self.mse_loss(decompressed, masked_tokens.detach())
                    else:
                        loss_recon = self.mse_loss(
                            decompressed, edge_tokens_raw.detach()
                        )

                # 组合损失
                loss_g = (
                    loss_mse * self.args.lambda_mse
                    + loss_adv * self.args.lambda_adv
                    + loss_recon * self.args.lambda_recon
                )

                # Budget & entropy losses (importance-aware mode only)
                if self.importance_scorer is not None:
                    loss_g = loss_g + self.args.lambda_budget * budget_loss
                    loss_g = loss_g + self.args.lambda_entropy * entropy_loss

            if self.use_amp:
                self.scaler_G.scale(loss_g).backward()
                self.scaler_G.step(self.opt_G)
                self.scaler_G.update()
            else:
                loss_g.backward()
                self.opt_G.step()

            # 计算 cos_sim
            with torch.no_grad():
                cos_sim = F.cosine_similarity(
                    fake_tokens.reshape(-1, self.args.target_hidden_size),
                    teacher_tokens.reshape(-1, self.args.target_hidden_size),
                    dim=-1,
                ).mean()

            total_loss_g += loss_g.item()
            total_loss_d += loss_d.item()
            total_cos_sim += cos_sim.item()
            total_mse += loss_mse.item()
            total_budget_loss += budget_loss.item()
            total_entropy_loss += entropy_loss.item()
            if mask is not None:
                total_effective_k += mask.sum(dim=1).mean().item()

            postfix = {
                "G": f"{loss_g.item():.3f}",
                "D": f"{loss_d.item():.3f}",
                "cos": f"{cos_sim.item():.3f}",
            }
            if mask is not None:
                postfix["eff_k"] = f"{mask.sum(dim=1).mean().item():.1f}"
            pbar.set_postfix(postfix)

        n = len(dataloader)
        metrics = {
            "loss_g": total_loss_g / n,
            "loss_d": total_loss_d / n,
            "cos_sim": total_cos_sim / n,
            "mse": total_mse / n,
        }
        if self.importance_scorer is not None:
            metrics["budget_loss"] = total_budget_loss / n
            metrics["entropy_loss"] = total_entropy_loss / n
            metrics["effective_k"] = total_effective_k / n
            metrics["temperature"] = self.budgeted_tx.current_temperature
        return metrics

    @torch.no_grad()
    def validate(self, dataloader):
        """验证"""
        self.student.eval()
        self.projector.eval()
        self.upsampler.eval()
        if self.importance_scorer is not None:
            self.importance_scorer.eval()
        if self.budgeted_tx is not None:
            self.budgeted_tx.eval()
        if self.sparse_upsampler is not None:
            self.sparse_upsampler.eval()

        total_mse = 0
        total_cos_sim = 0
        total_std = 0
        total_effective_k = 0

        for images, teacher_tokens in tqdm(dataloader, desc="Validating"):
            images = images.to(self.device)
            teacher_tokens = teacher_tokens.to(self.device).float()

            student_feat = self.student(images)[-1]
            edge_tokens = self.projector(student_feat)

            if self.importance_scorer is not None:
                # Information-aware path (eval mode: hard top-K selection)
                if self.args.text_aware and isinstance(
                    self.importance_scorer, TextAwareImportanceScorer
                ):
                    importance_logits, _ = self.importance_scorer(
                        student_feat, edge_tokens
                    )
                else:
                    importance_logits = self.importance_scorer(edge_tokens)

                # Hard top-K selection in eval mode
                selected_tokens, indices, _, _ = self.budgeted_tx(
                    edge_tokens, importance_logits
                )

                # Bottleneck
                if self.bottleneck is not None:
                    decompressed, _ = self.bottleneck(selected_tokens)
                else:
                    decompressed = selected_tokens

                # Sparse upsampler with indices
                output_tokens = self.sparse_upsampler(decompressed, indices)
                total_effective_k += indices.shape[1]
            else:
                # Original path
                if self.bottleneck is not None:
                    decompressed, _ = self.bottleneck(edge_tokens)
                    output_tokens = self.upsampler(decompressed)
                else:
                    output_tokens = self.upsampler(edge_tokens)

            if output_tokens.shape[1] != teacher_tokens.shape[1]:
                teacher_tokens = self._align_tokens(
                    teacher_tokens, output_tokens.shape[1]
                )

            mse = self.mse_loss(output_tokens, teacher_tokens)
            total_mse += mse.item()

            cos_sim = F.cosine_similarity(
                output_tokens.reshape(-1, self.args.target_hidden_size),
                teacher_tokens.reshape(-1, self.args.target_hidden_size),
                dim=-1,
            ).mean()
            total_cos_sim += cos_sim.item()

            # 特征方差 (越大越好，说明特征越 "sharp")
            total_std += output_tokens.std().item()

        n = len(dataloader)
        metrics = {
            "val_mse": total_mse / n,
            "val_cos_sim": total_cos_sim / n,
            "val_std": total_std / n,
        }
        if self.importance_scorer is not None:
            metrics["val_effective_k"] = total_effective_k / n
        return metrics

    def save_checkpoint(self, epoch, metrics, is_best=False, prefix=""):
        """保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "student_state_dict": self.student.state_dict(),
            "projector_state_dict": self.projector.state_dict(),
            "upsampler_state_dict": self.upsampler.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "opt_G_state_dict": self.opt_G.state_dict(),
            "opt_D_state_dict": self.opt_D.state_dict(),
            "metrics": metrics,
            "args": vars(self.args),
        }

        # 保存瓶颈层权重
        if self.bottleneck is not None:
            checkpoint["bottleneck_state_dict"] = self.bottleneck.state_dict()

        # 保存 importance-aware 模块权重
        if self.importance_scorer is not None:
            checkpoint["importance_scorer_state_dict"] = (
                self.importance_scorer.state_dict()
            )
        if self.budgeted_tx is not None:
            checkpoint["budgeted_tx_state_dict"] = self.budgeted_tx.state_dict()
        if self.sparse_upsampler is not None:
            checkpoint["sparse_upsampler_state_dict"] = (
                self.sparse_upsampler.state_dict()
            )

        torch.save(checkpoint, self.output_dir / f"{prefix}latest.pth")

        if is_best:
            torch.save(checkpoint, self.output_dir / f"{prefix}best.pth")
            self.logger.info(
                f"[Saved best model] cos_sim: {metrics['val_cos_sim']:.4f}"
            )

    def train_warmup(self, train_loader, val_loader):
        """Phase 1: Warmup 训练"""
        best_cos_sim = 0

        self.logger.info("=" * 60)
        self.logger.info("Phase 1: Warmup Training (MSE only)")
        self.logger.info("=" * 60)

        for epoch in range(1, self.args.epochs + 1):
            self._update_hard_topk_ratio(epoch)
            train_metrics = self.train_epoch_warmup(train_loader, epoch)
            val_metrics = self.validate(val_loader)

            self.scheduler_G.step()

            # Temperature annealing for budgeted transmission
            if self.budgeted_tx is not None:
                self.budgeted_tx.anneal_temperature()
                self.logger.info(
                    f"  Temperature: {self.budgeted_tx.current_temperature:.4f}"
                )

            metrics = {**train_metrics, **val_metrics}

            self.logger.info(
                f"Epoch {epoch}: "
                f"loss={metrics['loss']:.4f}, cos_sim={metrics['cos_sim']:.4f}, "
                f"val_cos_sim={metrics['val_cos_sim']:.4f}, val_std={metrics['val_std']:.3f}"
            )

            is_best = metrics["val_cos_sim"] > best_cos_sim
            if is_best:
                best_cos_sim = metrics["val_cos_sim"]

            if epoch % self.args.save_freq == 0 or is_best:
                self.save_checkpoint(epoch, metrics, is_best, prefix="warmup_")

        self.logger.info("=" * 60)
        self.logger.info(f"Warmup 完成! Best cos_sim: {best_cos_sim:.4f}")
        self.logger.info("=" * 60)

    def train_gan(self, train_loader, val_loader):
        """Phase 2: GAN 训练"""
        best_cos_sim = 0

        self.logger.info("=" * 60)
        self.logger.info("Phase 2: GAN Training")
        self.logger.info(f"λ_MSE={self.args.lambda_mse}, λ_ADV={self.args.lambda_adv}")
        self.logger.info("=" * 60)

        for epoch in range(1, self.args.epochs + 1):
            self._update_hard_topk_ratio(epoch)
            train_metrics = self.train_epoch_gan(train_loader, epoch)
            val_metrics = self.validate(val_loader)

            self.scheduler_G.step()
            self.scheduler_D.step()

            # Temperature annealing for budgeted transmission
            if self.budgeted_tx is not None:
                self.budgeted_tx.anneal_temperature()
                self.logger.info(
                    f"  Temperature: {self.budgeted_tx.current_temperature:.4f}"
                )

            metrics = {**train_metrics, **val_metrics}

            self.logger.info(
                f"Epoch {epoch}: "
                f"G={metrics['loss_g']:.4f}, D={metrics['loss_d']:.4f}, "
                f"mse={metrics['mse']:.4f}, val_cos_sim={metrics['val_cos_sim']:.4f}, "
                f"val_std={metrics['val_std']:.3f}"
            )

            is_best = metrics["val_cos_sim"] > best_cos_sim
            if is_best:
                best_cos_sim = metrics["val_cos_sim"]

            if epoch % self.args.save_freq == 0 or is_best:
                self.save_checkpoint(epoch, metrics, is_best, prefix="gan_")

        self.logger.info("=" * 60)
        self.logger.info(f"GAN 训练完成! Best cos_sim: {best_cos_sim:.4f}")
        self.logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="SplitOculo v2.0 GAN Training")

    # 数据参数
    parser.add_argument(
        "--features_dir", type=str, default=None, help="预计算特征目录 (静态模式必需)"
    )
    parser.add_argument("--data_dir", type=str, required=True, help="原始图像目录")
    parser.add_argument("--image_size", type=int, default=224)

    # Student 模型参数
    parser.add_argument("--student_model", type=str, default="mobilenetv2_100")
    parser.add_argument("--student_layer", type=int, default=3)

    # Projector 参数
    parser.add_argument("--target_hidden_size", type=int, default=1280)
    parser.add_argument("--projector_hidden", type=int, default=512)

    # Token 参数
    parser.add_argument("--transmission_tokens", type=int, default=49)
    parser.add_argument("--target_tokens", type=int, default=256)

    # Architecture variants
    parser.add_argument(
        "--projector_type", type=str, default="strided", choices=["pooling", "strided"]
    )
    parser.add_argument(
        "--initial_upsample",
        type=str,
        default="pixelshuffle",
        choices=["bilinear", "pixelshuffle"],
    )

    # Upsampler 参数
    parser.add_argument(
        "--upsampler_type",
        type=str,
        default="transformer",
        choices=["transformer", "mlp", "deconv"],
    )
    parser.add_argument("--transformer_layers", type=int, default=4)

    # 训练阶段
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=["warmup", "gan"],
        help="训练阶段: warmup (MSE only) 或 gan (adversarial)",
    )
    parser.add_argument(
        "--warmup_checkpoint",
        type=str,
        default=None,
        help="GAN 阶段加载的 warmup checkpoint",
    )
    # 训练参数
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_g", type=float, default=1e-4, help="Generator 学习率")
    parser.add_argument(
        "--lr_d", type=float, default=4e-5, help="Discriminator 学习率 (通常低于 G)"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=8)

    # GAN 损失权重
    parser.add_argument(
        "--lambda_mse", type=float, default=10.0, help="MSE 损失权重 (内容)"
    )
    parser.add_argument(
        "--lambda_adv", type=float, default=0.1, help="对抗损失权重 (样式)"
    )
    parser.add_argument(
        "--lambda_recon", type=float, default=0.1, help="瓶颈层重建损失权重"
    )

    # 瓶颈层参数
    parser.add_argument(
        "--bottleneck_dim", type=int, default=64, help="瓶颈层维度 (0 = 禁用瓶颈层)"
    )
    parser.add_argument(
        "--bottleneck_method",
        type=str,
        default="linear",
        choices=["linear", "mlp", "autoencoder"],
        help="瓶颈层方法",
    )

    # Information-aware transmission
    parser.add_argument(
        "--importance_aware",
        action="store_true",
        help="Enable information-aware token transmission",
    )
    parser.add_argument(
        "--text_aware",
        action="store_true",
        help="Enable text-region-aware importance scoring",
    )
    parser.add_argument(
        "--scorer_method",
        type=str,
        default="mlp",
        choices=["mlp", "attention"],
        help="Importance scorer method",
    )
    parser.add_argument(
        "--token_budget",
        type=int,
        default=24,
        help="Target number of tokens to transmit (budget)",
    )
    parser.add_argument(
        "--min_tokens",
        type=int,
        default=8,
        help="Minimum tokens to transmit during inference",
    )
    parser.add_argument(
        "--budget_temperature",
        type=float,
        default=1.0,
        help="Initial temperature for soft budget masking",
    )
    parser.add_argument(
        "--anneal_rate",
        type=float,
        default=0.01,
        help="Temperature annealing rate per epoch",
    )
    parser.add_argument(
        "--lambda_budget", type=float, default=0.1, help="Budget constraint loss weight"
    )
    parser.add_argument(
        "--lambda_entropy",
        type=float,
        default=0.01,
        help="Entropy regularization loss weight",
    )
    parser.add_argument(
        "--completion_layers",
        type=int,
        default=2,
        help="Number of Transformer layers for sparse token completion",
    )
    parser.add_argument(
        "--topk_train_schedule",
        action="store_true",
        help="Enable progressive hard top-k in training for importance-aware mode",
    )
    parser.add_argument(
        "--topk_start_epoch",
        type=int,
        default=1,
        help="Epoch to start hard top-k blending (phase-local)",
    )
    parser.add_argument(
        "--topk_end_epoch",
        type=int,
        default=10,
        help="Epoch to reach fully hard top-k blending (phase-local)",
    )

    # Quantization-aware training
    parser.add_argument(
        "--quantize_aware",
        action="store_true",
        help="Enable STE-based quantization-aware training",
    )
    parser.add_argument(
        "--num_bits",
        type=int,
        default=8,
        help="Number of bits for fake quantization (QAT)",
    )

    # 动态模式参数
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="启用动态模式: 训练时实时计算 Qwen 特征 (无需预计算)",
    )
    parser.add_argument(
        "--qwen_model",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Qwen 模型名称或路径 (动态模式)",
    )
    parser.add_argument(
        "--qwen_layer", type=int, default=4, help="Qwen ViT 提取层 (1-32, 默认 4)"
    )

    # 其他
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output_dir", type=str, default="./checkpoints/gan")
    parser.add_argument("--save_freq", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--amp",
        type=parse_bool_flag,
        nargs="?",
        const=True,
        default=True,
        help="启用 AMP 混合精度训练 (加速 1.5-2x)",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    # 加载数据
    if args.dynamic:
        # 动态模式: 实时计算 Qwen 特征
        print(f"[Dynamic Mode] qwen_model={args.qwen_model}, layer={args.qwen_layer}")

        # 创建 Qwen 特征提取器
        extractor = QwenFeatureExtractor(
            model_name=args.qwen_model,
            device=args.device,
            extract_layer=args.qwen_layer,
        ).load()

        train_dataset = DynamicFeatureDataset(
            images_dir=args.data_dir,
            extractor=extractor,
            split="train",
            image_size=args.image_size,
        )

        val_dataset = DynamicFeatureDataset(
            images_dir=args.data_dir,
            extractor=extractor,
            split="val",
            image_size=args.image_size,
        )

        # 动态模式不使用多进程 (Qwen 模型不可 pickle)
        num_workers = 0
    else:
        # 静态模式: 从预计算文件加载
        if not args.features_dir:
            raise ValueError(
                "静态模式需要 --features_dir 参数，或使用 --dynamic 启用动态模式"
            )

        train_dataset = PrecomputedFeatureDataset(
            features_dir=args.features_dir,
            images_dir=args.data_dir,
            split="train",
            image_size=args.image_size,
        )

        val_dataset = PrecomputedFeatureDataset(
            features_dir=args.features_dir,
            images_dir=args.data_dir,
            split="val",
            image_size=args.image_size,
        )

        num_workers = args.num_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # 创建训练器
    trainer = GANTrainer(args)
    trainer.logger.info(
        f"Data: mode={'dynamic' if args.dynamic else 'static'}, "
        f"train_samples={len(train_dataset)}, val_samples={len(val_dataset)}, "
        f"batch_size={args.batch_size}, num_workers={num_workers}"
    )

    # 根据阶段训练
    if args.phase == "warmup":
        trainer.train_warmup(train_loader, val_loader)
    elif args.phase == "gan":
        if args.warmup_checkpoint:
            trainer.load_checkpoint(args.warmup_checkpoint)
        trainer.train_gan(train_loader, val_loader)


if __name__ == "__main__":
    main()

"""
直接对齐 Qwen2.5-VL 的训练脚本

目标：训练 CNN + Projector 直接替代 Qwen2.5-VL 的 ViT + Merger
Teacher: Qwen2.5-VL 的视觉编码器输出 (2048 dim)
Student: MobileNetV2 + LLMProjector (96 -> 2048 dim)
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import timm

from core.utils import set_seed, get_logger, count_parameters
from data.dataset import get_dummy_loader, get_imagenet_loader


class LLMProjector(nn.Module):
    """
    将 CNN 特征投影到 LLM 隐藏空间
    
    设计参考 MobileVLM V2 的 LDPv2，但目标维度改为 Qwen2.5-VL
    """
    def __init__(self, in_channels, llm_hidden_size=2048, 
                 hidden_channels=512, downsample_ratio=2):
        super().__init__()
        
        # Point-wise conv: 通道变换
        self.pw_conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act1 = nn.GELU()
        
        # Spatial downsample (模拟 Qwen 的 2x2 merger)
        self.avg_pool = nn.AvgPool2d(kernel_size=downsample_ratio, stride=downsample_ratio)
        
        # Depth-wise conv + PEG (位置编码)
        self.dw_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                                  padding=1, groups=hidden_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.act2 = nn.GELU()
        
        # 投影到 LLM hidden size
        self.pw_conv2 = nn.Conv2d(hidden_channels, llm_hidden_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(llm_hidden_size)
        
    def forward(self, x):
        """
        Args:
            x: (B, in_channels, H, W) CNN 特征
        Returns:
            (B, num_tokens, llm_hidden_size) 可直接送入 LLM 的 tokens
        """
        x = self.act1(self.bn1(self.pw_conv1(x)))
        x = self.avg_pool(x)
        
        # PEG with residual
        residual = x
        x = self.act2(self.bn2(self.dw_conv(x)))
        x = x + residual
        
        x = self.bn3(self.pw_conv2(x))
        
        # 转换为 token 序列: (B, C, H, W) -> (B, H*W, C)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, num_tokens, llm_hidden_size)
        
        return x


class QwenVisionTeacher(nn.Module):
    """
    Qwen2.5-VL 视觉编码器包装器
    
    提取 ViT + Merger 的输出作为蒸馏目标
    """
    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct", device='cuda', image_size=224):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.image_size = image_size
        self.model = None
        self.processor = None
        
    def load(self):
        """延迟加载模型（需要较大显存）"""
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        print(f"📥 Loading Qwen2.5-VL vision encoder from {self.model_name}...")
        
        # 加载完整模型，使用低精度节省显存
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        # 预计算固定尺寸图像的 grid_thw
        # Qwen2.5-VL 使用 patch_size=14, merge_size=2
        # 对于 224x224 图像: (224/14) = 16 patches per dimension
        # merge 后: 16/2 = 8 
        self.patch_size = 14
        self.merge_size = 2
        
        print("✅ Qwen2.5-VL loaded successfully")
        return self
    
    def _compute_grid_thw(self, batch_size, height, width):
        """
        计算 Qwen 所需的 grid_thw tensor
        
        对于静态图像: t=1, h=height/patch_size, w=width/patch_size
        """
        # 计算 patch 网格大小
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        
        # grid_thw: (num_images, 3) - 每个图像的 (t, h, w)
        # 对于静态图像 t=1
        grid_thw = torch.tensor([[1, grid_h, grid_w]] * batch_size, dtype=torch.long)
        
        return grid_thw
    
    def _preprocess_images(self, images):
        """
        将标准化的 ImageNet tensor 转换为 Qwen 格式
        
        Args:
            images: (B, 3, H, W) normalized tensor (ImageNet normalization)
        Returns:
            pixel_values: Qwen 格式的 pixel values
            grid_thw: grid 信息
        """
        from PIL import Image
        import numpy as np
        
        B, C, H, W = images.shape
        device = images.device
        
        # 反归一化 (ImageNet -> [0, 1])
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        images_denorm = images * std + mean
        images_denorm = images_denorm.clamp(0, 1)
        
        # 转换为 PIL 图像列表
        pil_images = []
        for i in range(B):
            img_np = (images_denorm[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img_np))
        
        # 使用 Qwen processor 处理
        # 构造 Qwen 需要的消息格式
        messages_list = []
        for img in pil_images:
            messages_list.append([{
                "role": "user",
                "content": [{"type": "image", "image": img}]
            }])
        
        # 批量处理
        all_pixel_values = []
        all_grid_thw = []
        
        for messages in messages_list:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=[text],
                images=[messages[0]["content"][0]["image"]],
                return_tensors="pt",
                padding=True
            )
            all_pixel_values.append(inputs["pixel_values"])
            all_grid_thw.append(inputs["image_grid_thw"])
        
        # 合并 batch
        pixel_values = torch.cat(all_pixel_values, dim=0)
        grid_thw = torch.cat(all_grid_thw, dim=0)
        
        return pixel_values.to(device), grid_thw.to(device)
    
    @torch.no_grad()
    def extract_vision_features(self, images, _unused=None):
        """
        提取视觉特征（Merger 输出，维度 2048）
        
        Args:
            images: (B, 3, H, W) ImageNet normalized images
        Returns:
            (B, fixed_num_tokens, 2048) visual tokens
        """
        from PIL import Image
        import numpy as np
        
        B, C, H, W = images.shape
        device = images.device
        
        # 反归一化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        images_denorm = images * std + mean
        images_denorm = images_denorm.clamp(0, 1)
        
        all_features = []
        
        # 逐个处理图像（避免 Qwen 的动态 token 数量问题）
        for i in range(B):
            # 转换为 PIL
            img_np = (images_denorm[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            
            # 使用 processor
            messages = [{
                "role": "user",
                "content": [{"type": "image", "image": pil_img}]
            }]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=[text],
                images=[pil_img],
                return_tensors="pt",
                padding=True
            )
            
            pixel_values = inputs["pixel_values"].to(device)
            grid_thw = inputs["image_grid_thw"].to(device)
            
            # 提取视觉特征
            hidden_states = self.model.visual(
                pixel_values.to(self.model.visual.patch_embed.proj.weight.dtype),
                grid_thw=grid_thw
            )
            # hidden_states: (num_tokens, hidden_size)
            
            all_features.append(hidden_states)
        
        # 对齐 token 数量（取最小的）
        min_tokens = min(f.shape[0] for f in all_features)
        
        # 截断或池化到相同长度
        aligned_features = []
        for feat in all_features:
            if feat.shape[0] > min_tokens:
                # 使用平均池化减少 tokens
                feat = feat[:min_tokens]  # 简单截断
            aligned_features.append(feat.unsqueeze(0))  # (1, num_tokens, hidden_size)
        
        # 合并 batch
        batch_features = torch.cat(aligned_features, dim=0)  # (B, num_tokens, hidden_size)
        
        return batch_features


class SimulatedQwenTeacher(nn.Module):
    """
    模拟 Qwen2.5-VL 视觉编码器 (用于无法加载完整模型时)
    
    结构:
    - ViT-like encoder: 1280 hidden
    - Patch Merger: 2x2 spatial merge + 1280->2048 projection
    """
    def __init__(self, image_size=224, patch_size=14, hidden_size=1280, 
                 llm_hidden_size=2048, merge_ratio=2):
        super().__init__()
        
        self.patch_size = patch_size
        self.merge_ratio = merge_ratio
        num_patches = (image_size // patch_size) ** 2
        
        # 使用 timm 的 ViT 作为 backbone
        self.vit = timm.create_model(
            'vit_large_patch14_clip_224',  # CLIP ViT-L/14
            pretrained=True,
            num_classes=0,  # 去掉分类头
        )
        
        # 获取 ViT 输出维度
        self.vit_hidden = self.vit.embed_dim  # 1024 for ViT-L
        
        # 模拟 Qwen 的 Merger: 2x2 空间合并 + 投影
        self.merger_proj = nn.Linear(self.vit_hidden * (merge_ratio ** 2), llm_hidden_size)
        self.merger_ln = nn.LayerNorm(llm_hidden_size)
        
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) images
        Returns:
            (B, num_tokens, 2048) visual tokens (模拟 Qwen Merger 输出)
        """
        B = x.shape[0]
        
        # ViT forward (获取 patch tokens，不要 CLS)
        # timm ViT 的 forward_features 返回 (B, num_patches+1, embed_dim)
        tokens = self.vit.forward_features(x)
        tokens = tokens[:, 1:, :]  # 去掉 CLS token, (B, 196, 1024)
        
        # 重塑为 2D grid
        h = w = int(tokens.shape[1] ** 0.5)  # 14x14
        tokens = tokens.view(B, h, w, -1)  # (B, 14, 14, 1024)
        
        # 2x2 Patch Merge (模拟 Qwen 的 merger)
        # (B, 14, 14, 1024) -> (B, 7, 7, 4096) -> (B, 7, 7, 2048)
        new_h, new_w = h // self.merge_ratio, w // self.merge_ratio
        
        # 重组 patches
        tokens = tokens.view(B, new_h, self.merge_ratio, new_w, self.merge_ratio, -1)
        tokens = tokens.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, 7, 7, 2, 2, 1024)
        tokens = tokens.view(B, new_h, new_w, -1)  # (B, 7, 7, 4096)
        
        # 投影到 LLM hidden size
        tokens = self.merger_proj(tokens)  # (B, 7, 7, 2048)
        tokens = self.merger_ln(tokens)
        
        # 展平为 token 序列
        tokens = tokens.view(B, -1, tokens.shape[-1])  # (B, 49, 2048)
        
        return tokens


class QwenAlignmentTrainer:
    """
    直接对齐 Qwen2.5-VL 的训练器
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.logger = get_logger('qwen_align', log_file=f'{args.output_dir}/train.log')
        
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载模型
        self._load_teacher()
        self._load_student()
        
        # 优化器 (只训练 student 和 projector)
        self.optimizer = optim.AdamW(
            list(self.student.parameters()) + list(self.projector.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs
        )
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.cos_loss = nn.CosineEmbeddingLoss()
        
        self.logger.info(f"Teacher type: {'Real Qwen' if args.use_real_qwen else 'Simulated'}")
        self.logger.info(f"Student params: {count_parameters(self.student) / 1e6:.2f}M")
        self.logger.info(f"Projector params: {count_parameters(self.projector) / 1e6:.2f}M")
    
    def _load_teacher(self):
        """加载 Teacher (Qwen2.5-VL 视觉编码器)"""
        if self.args.use_real_qwen:
            self.teacher = QwenVisionTeacher(
                model_name=self.args.qwen_model,
                device=self.device
            ).load()
        else:
            self.logger.info("Using simulated Qwen teacher (CLIP ViT + Merger)")
            self.teacher = SimulatedQwenTeacher(
                image_size=self.args.image_size,
                llm_hidden_size=self.args.llm_hidden_size
            ).to(self.device)
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False
    
    def _load_student(self):
        """加载 Student (CNN + Projector)"""
        self.logger.info(f"Loading student: {self.args.student_model}")
        
        # CNN backbone
        self.student = timm.create_model(
            self.args.student_model,
            pretrained=True,
            features_only=True,
            out_indices=[self.args.student_layer]
        ).to(self.device)
        
        # 获取 CNN 输出通道数
        with torch.no_grad():
            dummy = torch.randn(1, 3, self.args.image_size, self.args.image_size).to(self.device)
            student_feat = self.student(dummy)[-1]
            self.student_channels = student_feat.shape[1]
            self.student_size = student_feat.shape[2]
        
        self.logger.info(f"Student output: {self.student_channels} channels, {self.student_size}x{self.student_size}")
        
        # LLM Projector
        self.projector = LLMProjector(
            in_channels=self.student_channels,
            llm_hidden_size=self.args.llm_hidden_size,
            hidden_channels=self.args.projector_hidden,
            downsample_ratio=self.args.downsample_ratio
        ).to(self.device)
        
        # 计算输出 token 数
        output_size = self.student_size // self.args.downsample_ratio
        self.num_tokens = output_size * output_size
        self.logger.info(f"Output tokens: {self.num_tokens} ({output_size}x{output_size})")
    
    def train_epoch(self, dataloader, epoch):
        """训练一个 epoch"""
        self.student.train()
        self.projector.train()
        
        total_loss = 0
        total_mse = 0
        total_cos = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)
            
            # Teacher forward
            with torch.no_grad():
                if self.args.use_real_qwen:
                    # 使用真实 Qwen
                    teacher_tokens = self.teacher.extract_vision_features(images, None)
                else:
                    # 使用模拟的 Teacher
                    teacher_tokens = self.teacher(images)  # (B, num_tokens, 2048)
            
            # Student forward
            student_feat = self.student(images)[-1]  # (B, C, H, W)
            student_tokens = self.projector(student_feat)  # (B, num_tokens, 2048)
            
            # 对齐 token 数量 (如果不同)
            if student_tokens.shape[1] != teacher_tokens.shape[1]:
                # 简单平均池化对齐
                student_tokens = self._align_tokens(student_tokens, teacher_tokens.shape[1])
            
            # 计算损失
            mse = self.mse_loss(student_tokens, teacher_tokens.float())
            
            # Cosine loss (per-token)
            b, n, c = student_tokens.shape
            student_flat = student_tokens.contiguous().view(b * n, c)
            teacher_flat = teacher_tokens.contiguous().view(b * n, c).float()
            target = torch.ones(b * n).to(self.device)
            cos = self.cos_loss(student_flat, teacher_flat, target)
            
            loss = mse + self.args.cos_weight * cos
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse.item()
            total_cos += cos.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mse': f'{mse.item():.4f}',
                'cos': f'{cos.item():.4f}'
            })
        
        n = len(dataloader)
        return {
            'loss': total_loss / n,
            'mse': total_mse / n,
            'cos': total_cos / n
        }
    
    def _align_tokens(self, tokens, target_num):
        """对齐 token 数量"""
        b, n, c = tokens.shape
        h = w = int(n ** 0.5)
        
        tokens = tokens.view(b, h, w, c).permute(0, 3, 1, 2)  # (B, C, H, W)
        
        target_h = target_w = int(target_num ** 0.5)
        tokens = nn.functional.adaptive_avg_pool2d(tokens, (target_h, target_w))
        
        tokens = tokens.permute(0, 2, 3, 1).view(b, -1, c)  # (B, target_num, C)
        return tokens
    
    @torch.no_grad()
    def validate(self, dataloader):
        """验证"""
        self.student.eval()
        self.projector.eval()
        
        total_mse = 0
        total_cos_sim = 0
        
        for images, _ in tqdm(dataloader, desc="Validating"):
            images = images.to(self.device)
            
            if self.args.use_real_qwen:
                teacher_tokens = self.teacher.extract_vision_features(images, None)
            else:
                teacher_tokens = self.teacher(images)
            
            student_feat = self.student(images)[-1]
            student_tokens = self.projector(student_feat)
            
            if student_tokens.shape[1] != teacher_tokens.shape[1]:
                student_tokens = self._align_tokens(student_tokens, teacher_tokens.shape[1])
            
            mse = self.mse_loss(student_tokens, teacher_tokens.float())
            
            # Cosine similarity
            cos_sim = nn.functional.cosine_similarity(
                student_tokens, teacher_tokens.float(), dim=-1
            ).mean()
            
            total_mse += mse.item()
            total_cos_sim += cos_sim.item()
        
        n = len(dataloader)
        return {
            'val_mse': total_mse / n,
            'val_cos_sim': total_cos_sim / n
        }
    
    def train(self, train_loader, val_loader=None):
        """完整训练循环"""
        best_loss = float('inf')
        
        for epoch in range(1, self.args.epochs + 1):
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Epoch {epoch}/{self.args.epochs}")
            self.logger.info(f"LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
            train_metrics = self.train_epoch(train_loader, epoch)
            self.logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                           f"MSE: {train_metrics['mse']:.4f}, "
                           f"Cos: {train_metrics['cos']:.4f}")
            
            if val_loader:
                val_metrics = self.validate(val_loader)
                self.logger.info(f"Val - MSE: {val_metrics['val_mse']:.4f}, "
                               f"Cos Sim: {val_metrics['val_cos_sim']:.4f}")
            
            if train_metrics['loss'] < best_loss:
                best_loss = train_metrics['loss']
                self.save_checkpoint(epoch, is_best=True)
            
            if epoch % self.args.save_freq == 0:
                self.save_checkpoint(epoch)
            
            self.scheduler.step()
        
        self.logger.info("Training completed!")
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'student_state_dict': self.student.state_dict(),
            'projector_state_dict': self.projector.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'args': self.args
        }
        
        if is_best:
            path = self.output_dir / 'best_qwen_aligned.pth'
        else:
            path = self.output_dir / f'qwen_checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")


def main():
    parser = argparse.ArgumentParser(description='Direct Qwen2.5-VL Alignment Training')
    
    # Qwen 参数
    parser.add_argument('--use_real_qwen', action='store_true',
                        help='Use real Qwen2.5-VL (requires ~8GB VRAM)')
    parser.add_argument('--qwen_model', type=str, default='Qwen/Qwen2.5-VL-3B-Instruct',
                        help='Qwen model name')
    parser.add_argument('--llm_hidden_size', type=int, default=2048,
                        help='Qwen LLM hidden size (2048 for 3B model)')
    
    # Student 参数
    parser.add_argument('--student_model', type=str, default='mobilenetv2_100',
                        help='Student CNN model')
    parser.add_argument('--student_layer', type=int, default=3,
                        help='Student layer to extract features from')
    parser.add_argument('--projector_hidden', type=int, default=512,
                        help='Projector hidden channels')
    parser.add_argument('--downsample_ratio', type=int, default=2,
                        help='Spatial downsample ratio in projector')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--cos_weight', type=float, default=0.5,
                        help='Weight for cosine similarity loss')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--dummy', action='store_true',
                        help='Use dummy data for testing')
    
    # 其他参数
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/qwen_aligned')
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # 创建数据加载器
    if args.dummy:
        train_loader = get_dummy_loader(
            batch_size=args.batch_size,
            num_samples=1000,
            image_size=args.image_size
        )
        val_loader = get_dummy_loader(
            batch_size=args.batch_size,
            num_samples=200,
            image_size=args.image_size
        )
    elif args.data_dir:
        train_loader = get_imagenet_loader(
            args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            is_train=True,
            num_workers=args.num_workers
        )
        val_loader = get_imagenet_loader(
            args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            is_train=False,
            num_workers=args.num_workers
        )
    else:
        print("⚠️ No data specified. Use --data_dir or --dummy")
        return
    
    trainer = QwenAlignmentTrainer(args)
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()

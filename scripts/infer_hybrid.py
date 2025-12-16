"""
混合推理脚本：CNN 替换 Qwen ViT 浅层，剩余深层仍用 Qwen

端云协同场景:
- 端侧: Image → CNN → Projector → features (1280 dim, 可量化)
- 云端: features → Remaining Qwen Blocks → Merger → LLM → response

Usage:
    python scripts/infer_hybrid.py --checkpoint checkpoints/best_model.pth --image photo.jpg
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms


class LLMProjector(nn.Module):
    """将 CNN 特征投影到 ViT 隐藏空间 (1280 dim)"""
    def __init__(self, in_channels, hidden_size=1280, 
                 hidden_channels=512, downsample_ratio=2):
        super().__init__()
        
        self.pw_conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act1 = nn.GELU()
        self.avg_pool = nn.AvgPool2d(kernel_size=downsample_ratio, stride=downsample_ratio)
        self.dw_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                                  padding=1, groups=hidden_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.act2 = nn.GELU()
        self.pw_conv2 = nn.Conv2d(hidden_channels, hidden_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(hidden_size)
        
    def forward(self, x):
        x = self.act1(self.bn1(self.pw_conv1(x)))
        x = self.avg_pool(x)
        residual = x
        x = self.act2(self.bn2(self.dw_conv(x)))
        x = x + residual
        x = self.bn3(self.pw_conv2(x))
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, num_tokens, hidden_size)
        return x


class EdgeVisualEncoder(nn.Module):
    """
    端侧视觉编码器
    
    CNN (MobileNetV2) + Projector → features (1280 dim)
    输出与 Qwen ViT 中间层兼容
    """
    def __init__(self, student_model='mobilenetv2_100', student_layer=3,
                 student_channels=96, hidden_size=1280,
                 projector_hidden=512, downsample_ratio=2):
        super().__init__()
        
        self.student = timm.create_model(
            student_model,
            pretrained=False,
            features_only=True,
            out_indices=[student_layer]
        )
        
        self.projector = LLMProjector(
            in_channels=student_channels,
            hidden_size=hidden_size,
            hidden_channels=projector_hidden,
            downsample_ratio=downsample_ratio
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input images
        Returns:
            (B, num_tokens, 1280) features compatible with Qwen ViT
        """
        feat = self.student(x)[-1]
        tokens = self.projector(feat)
        return tokens
    
    def load_checkpoint(self, checkpoint_path, device='cpu'):
        """加载训练好的权重"""
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.student.load_state_dict(ckpt['student_state_dict'])
        self.projector.load_state_dict(ckpt['projector_state_dict'])
        print(f"✅ 已加载检查点: {checkpoint_path}")
        return self


class HybridQwenVLM:
    """
    混合视觉语言模型
    
    CNN + Projector 替换 Qwen ViT 前 N 层
    剩余层 + Merger 仍使用 Qwen
    
    架构:
        Image → CNN → Projector (1280) → Qwen Blocks[N:] → Merger (2048) → LLM
    """
    def __init__(self, 
                 edge_checkpoint=None,
                 qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
                 split_layer=8,
                 device='cuda'):
        self.device = device
        self.qwen_model_name = qwen_model_name
        self.split_layer = split_layer
        
        # 端侧编码器
        self.edge_encoder = EdgeVisualEncoder(
            student_model='mobilenetv2_100',
            student_layer=3,
            student_channels=96,
            hidden_size=1280
        )
        
        if edge_checkpoint:
            self.edge_encoder.load_checkpoint(edge_checkpoint, device)
        
        self.edge_encoder = self.edge_encoder.to(device)
        self.edge_encoder.eval()
        
        # Qwen 模型 (延迟加载)
        self.qwen_model = None
        self.processor = None
    
    def load_qwen(self):
        """加载 Qwen 模型 (云端)"""
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        print(f"📥 Loading Qwen from {self.qwen_model_name}...")
        
        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.qwen_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.qwen_model_name,
            trust_remote_code=True
        )
        
        # 冻结参数
        for param in self.qwen_model.parameters():
            param.requires_grad = False
        self.qwen_model.eval()
        
        print(f"✅ Qwen loaded")
        print(f"📍 Will use Qwen blocks from layer {self.split_layer} onwards")
        
        return self
    
    @torch.no_grad()
    def encode_image_edge(self, image_tensor):
        """
        端侧编码
        
        Args:
            image_tensor: (B, 3, H, W) normalized tensor
        Returns:
            features: (B, num_tokens, 1280)
        """
        return self.edge_encoder(image_tensor)
    
    @torch.no_grad() 
    def complete_visual_encoding(self, edge_features, grid_thw=None):
        """
        云端完成剩余视觉编码
        
        Args:
            edge_features: (B, num_tokens, 1280) 端侧输出 (来自 CNN, 通常是 7x7=49 tokens)
            grid_thw: grid 信息 (可选)
        Returns:
            visual_tokens: (B, merged_tokens, 2048) 可送入 LLM
        """
        import torch.nn.functional as F
        
        if self.qwen_model is None:
            raise RuntimeError("请先调用 load_qwen()")
        
        visual = self.qwen_model.visual
        B = edge_features.shape[0]
        num_tokens = edge_features.shape[1]
        cnn_h = cnn_w = int(num_tokens ** 0.5)  # 7x7 = 49
        
        # Qwen ViT 使用 224/14 = 16x16 = 256 tokens
        target_h = target_w = 16
        target_tokens = target_h * target_w  # 256
        
        # 上采样 CNN 特征以匹配 Qwen 的 token 网格
        # (B, 49, 1280) -> (B, 1280, 7, 7) -> upsample -> (B, 1280, 16, 16) -> (B, 256, 1280)
        edge_features_2d = edge_features.view(B, cnn_h, cnn_w, -1).permute(0, 3, 1, 2)  # (B, 1280, 7, 7)
        edge_features_upsampled = F.interpolate(
            edge_features_2d, 
            size=(target_h, target_w), 
            mode='bilinear', 
            align_corners=False
        )  # (B, 1280, 16, 16)
        edge_features = edge_features_upsampled.permute(0, 2, 3, 1).view(B, target_tokens, -1)  # (B, 256, 1280)
        
        print(f"   上采样: {cnn_h}x{cnn_w} -> {target_h}x{target_w} tokens")
        
        # === Feature scaling to match Qwen Layer 4 distribution ===
        # CNN output: mean≈0, std≈0.06
        # Qwen Layer 4: mean≈-0.017, std≈0.83
        # Scale CNN output to match Qwen's expected distribution
        cnn_std = edge_features.std()
        target_std = 0.83  # Qwen Layer 4 approximate std
        target_mean = -0.017  # Qwen Layer 4 approximate mean
        
        if cnn_std > 0:
            edge_features = (edge_features - edge_features.mean()) / cnn_std * target_std + target_mean
            print(f"   特征缩放: std {cnn_std:.3f} -> {target_std}, mean -> {target_mean}")
        
        # 处理 edge_features 形状: (B, num_tokens, hidden) -> (total_tokens, hidden)
        edge_features = edge_features.view(-1, edge_features.shape[-1])  # (256, 1280)
        
        # 使用固定的 grid_thw
        grid_thw = torch.tensor([[1, target_h, target_w]] * B, dtype=torch.long).to(self.device)
        
        # Rotary position embedding
        rotary_pos_emb = visual.rot_pos_emb(grid_thw)
        
        # Window indexing
        window_index, cu_window_seqlens = visual.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=edge_features.device,
            dtype=torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        
        # 重排 hidden_states 以匹配 window indexing
        seq_len = edge_features.shape[0]
        hidden_states = edge_features.view(seq_len // visual.spatial_merge_unit, visual.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        
        # 重排 rotary_pos_emb 并创建 position_embeddings
        rotary_pos_emb = rotary_pos_emb.view(seq_len // visual.spatial_merge_unit, visual.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.view(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        
        # cu_seqlens
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2],
            grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        
        # 转换数据类型
        hidden_states = hidden_states.to(visual.blocks[0].attn.qkv.weight.dtype)
        
        # 继续执行剩余的 blocks
        for i, block in enumerate(visual.blocks):
            if i < self.split_layer:
                continue  # 跳过前 N 层 (已由 CNN 替代)
            
            # 选择正确的 cu_seqlens
            if i in visual.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
            )
        
        # 通过 merger
        # 需要先反转 window indexing
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states.view(seq_len // visual.spatial_merge_unit, visual.spatial_merge_unit, -1)
        hidden_states = hidden_states[reverse_indices, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        
        visual_tokens = visual.merger(hidden_states)
        
        # 恢复 batch 维度
        visual_tokens = visual_tokens.view(B, -1, visual_tokens.shape[-1])
        
        return visual_tokens  # (B, merged_tokens, 2048)
    
    def quantize_features(self, features, method='int8'):
        """
        量化特征以减少传输大小
        
        Args:
            features: (B, num_tokens, hidden_size) float tensor
            method: 'int8', 'fp16'
        Returns:
            quantized_data, metadata for dequantization
        """
        if method == 'int8':
            # 简单的 min-max int8 量化
            min_val = features.min()
            max_val = features.max()
            scale = (max_val - min_val) / 255
            quantized = ((features - min_val) / scale).round().to(torch.uint8)
            return quantized, {'min': min_val.item(), 'scale': scale.item()}
        elif method == 'fp16':
            return features.half(), {}
        else:
            return features, {}
    
    def dequantize_features(self, quantized, metadata, method='int8'):
        """反量化"""
        if method == 'int8':
            return quantized.float() * metadata['scale'] + metadata['min']
        elif method == 'fp16':
            return quantized.float()
        else:
            return quantized


def get_image_transform(image_size=224):
    """图像预处理"""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def main():
    parser = argparse.ArgumentParser(description='Hybrid CNN-Qwen Inference')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/qwen_precomputed/best_model.pth')
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--dummy', action='store_true')
    parser.add_argument('--split_layer', type=int, default=4,
                        help='Which layer CNN replaces (1-32, must match training layer)')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--quantize', type=str, default='none',
                        choices=['none', 'fp16', 'int8'],
                        help='Quantization method for transmission (none recommended for debugging)')
    parser.add_argument('--full_inference', action='store_true',
                        help='Run complete inference including Qwen deep layers')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Text prompt for LLM (default: 描述这张图片中的内容。)')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"🔧 Device: {device}")
    print(f"📍 Split layer: {args.split_layer}")
    
    # 加载混合模型
    hybrid = HybridQwenVLM(
        edge_checkpoint=args.checkpoint if Path(args.checkpoint).exists() else None,
        split_layer=args.split_layer,
        device=device
    )
    
    # 准备输入
    transform = get_image_transform(224)
    
    if args.dummy:
        print("📷 使用假数据测试...")
        image = torch.randn(1, 3, 224, 224).to(device)
    elif args.image:
        print(f"📷 加载图像: {args.image}")
        img = Image.open(args.image).convert('RGB')
        image = transform(img).unsqueeze(0).to(device)
    else:
        print("❌ 请指定 --image 或 --dummy")
        return
    
    # 端侧编码
    print("\n🖥️  端侧 (Edge) 编码...")
    edge_features = hybrid.encode_image_edge(image)
    print(f"   输出形状: {edge_features.shape}")
    print(f"   维度: {edge_features.shape[-1]} (应为 1280)")
    
    # 量化分析 (端侧特征大小，上采样前)
    print(f"\n📡 传输大小分析 ({args.quantize} 量化):")
    quantized, metadata = hybrid.quantize_features(edge_features, method=args.quantize)
    
    if args.quantize == 'int8':
        bytes_per_val = 1
    elif args.quantize == 'fp16':
        bytes_per_val = 2
    else:
        bytes_per_val = 4
    
    # 端侧特征大小 (上采样在云端执行)
    transmission_bytes = quantized.numel() * bytes_per_val
    num_tokens = edge_features.shape[1]  # 49 tokens (7x7)
    
    # 测量实际 JPEG 大小 (resize 到 224x224 后)
    import io
    if args.image and not args.dummy:
        img_resized = Image.open(args.image).convert('RGB')
        img_resized = img_resized.resize((224, 224), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img_resized.save(buffer, format='JPEG', quality=85)
        jpeg_bytes = buffer.tell()
        print(f"   原始 JPEG (224×224, Q85): {jpeg_bytes / 1024:.2f} KB")
    else:
        jpeg_bytes = 30 * 1024  # fallback
        print(f"   原始 JPEG 224×224: ~{jpeg_bytes / 1024:.0f} KB (估计)")
    
    print(f"   端侧特征 ({num_tokens} tokens, {args.quantize}): {transmission_bytes / 1024:.2f} KB")
    if transmission_bytes < jpeg_bytes:
        print(f"   压缩比: {jpeg_bytes / transmission_bytes:.1f}x")
    else:
        print(f"   大于原图: {transmission_bytes / jpeg_bytes:.1f}x")
    
    # 完整推理 (可选)
    if args.full_inference:
        print("\n☁️  云端 (Cloud) 完成推理...")
        hybrid.load_qwen()
        
        # 反量化
        dequantized = hybrid.dequantize_features(quantized, metadata, method=args.quantize)
        
        # 完成视觉编码
        visual_tokens = hybrid.complete_visual_encoding(dequantized)
        print(f"   视觉 tokens: {visual_tokens.shape}")
        print(f"   维度: {visual_tokens.shape[-1]} (应为 2048)")
        
        # 生成文本回复
        prompt = args.prompt if args.prompt else "描述这张图片中的内容。"
        print(f"\n💬 生成回复 (prompt: {prompt})")
        
        response = generate_with_visual_tokens(
            hybrid.qwen_model,
            hybrid.processor,
            visual_tokens,
            prompt,
            device
        )
        print(f"\n🤖 回复:\n{response}")
        
        return response
    
    return edge_features


def generate_with_visual_tokens(model, processor, visual_tokens, prompt, device):
    """
    使用视觉 tokens 生成文本回复
    
    正确的注入方式：替换 <|image_pad|> 占位符，而不是简单拼接
    """
    num_visual_tokens = visual_tokens.shape[1]  # 例如 64 或 165
    
    # 构造包含图像占位符的消息
    # 使用 <|vision_start|><|image_pad|>×N<|vision_end|> 格式
    image_placeholder = "<|vision_start|>" + "<|image_pad|>" * num_visual_tokens + "<|vision_end|>"
    
    messages = [
        {"role": "user", "content": image_placeholder + prompt}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize
    text_inputs = processor.tokenizer(text, return_tensors="pt", padding=True)
    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs["attention_mask"].to(device)
    
    # 获取文本 embeddings
    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(input_ids)
    
    # 找到 image_pad token 的位置并替换
    image_token_id = model.config.image_token_id  # 151655
    image_mask = (input_ids == image_token_id)
    
    # 确保 visual_tokens 数量匹配
    num_placeholders = image_mask.sum().item()
    if num_placeholders != num_visual_tokens:
        print(f"   ⚠️ Token 数量不匹配: 占位符 {num_placeholders} vs 视觉 {num_visual_tokens}")
        # 调整 visual_tokens 大小
        if num_visual_tokens < num_placeholders:
            # 重复最后一个 token
            pad = visual_tokens[:, -1:, :].repeat(1, num_placeholders - num_visual_tokens, 1)
            visual_tokens = torch.cat([visual_tokens, pad], dim=1)
        else:
            # 截断
            visual_tokens = visual_tokens[:, :num_placeholders, :]
    
    # 替换 image_pad embeddings 为 visual_tokens
    visual_tokens = visual_tokens.to(inputs_embeds.dtype)
    
    # 找到所有 image_pad 位置的索引
    batch_indices, token_indices = torch.where(image_mask)
    
    # 替换
    for i, (b, t) in enumerate(zip(batch_indices, token_indices)):
        inputs_embeds[b, t] = visual_tokens[b, i % visual_tokens.shape[1]]
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=False,  # 使用 greedy 更稳定
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    
    # 解码
    response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 只返回回复部分（去掉 prompt）
    # 查找 assistant 标记后的内容
    if "assistant" in response.lower():
        response = response.split("assistant")[-1].strip()
    
    return response


if __name__ == '__main__':
    main()


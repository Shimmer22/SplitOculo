"""
使用训练好的 CNN+Projector 替换 Qwen2.5-VL 的视觉编码器进行推理

端云协同场景:
- 端侧: Image -> CNN -> Projector -> visual_tokens (2048 dim)
- 云端: visual_tokens + text -> Qwen LLM -> response
"""
import argparse
import torch
import torch.nn as nn
import timm
from PIL import Image
from pathlib import Path


class LLMProjector(nn.Module):
    """与 train_qwen_align.py 中定义相同"""
    def __init__(self, in_channels, llm_hidden_size=2048, 
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
        self.pw_conv2 = nn.Conv2d(hidden_channels, llm_hidden_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(llm_hidden_size)
        
    def forward(self, x):
        x = self.act1(self.bn1(self.pw_conv1(x)))
        x = self.avg_pool(x)
        residual = x
        x = self.act2(self.bn2(self.dw_conv(x)))
        x = x + residual
        x = self.bn3(self.pw_conv2(x))
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x


class EdgeVisualEncoder(nn.Module):
    """
    端侧视觉编码器
    
    CNN (MobileNetV2) + Projector -> visual tokens for Qwen LLM
    """
    def __init__(self, student_model='mobilenetv2_100', student_layer=3,
                 student_channels=96, llm_hidden_size=2048,
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
            llm_hidden_size=llm_hidden_size,
            hidden_channels=projector_hidden,
            downsample_ratio=downsample_ratio
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) input images
        Returns:
            (B, num_tokens, 2048) visual tokens for Qwen LLM
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


class QwenWithEdgeVision:
    """
    使用端侧视觉编码器的 Qwen2.5-VL
    
    替换原始 ViT+Merger，直接使用 CNN+Projector 的输出
    """
    def __init__(self, 
                 qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
                 edge_checkpoint=None,
                 device='cuda'):
        self.device = device
        self.qwen_model_name = qwen_model_name
        
        # 加载端侧视觉编码器
        self.edge_encoder = EdgeVisualEncoder(
            student_model='mobilenetv2_100',
            student_layer=3,
            student_channels=96,
            llm_hidden_size=2048
        )
        
        if edge_checkpoint:
            self.edge_encoder.load_checkpoint(edge_checkpoint, device)
        
        self.edge_encoder = self.edge_encoder.to(device)
        self.edge_encoder.eval()
        
        # Qwen LLM (暂不加载，节省显存)
        self.qwen_model = None
        self.tokenizer = None
    
    def load_qwen_llm(self):
        """加载 Qwen LLM 部分（云端调用）"""
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer
        
        print(f"📥 Loading Qwen LLM from {self.qwen_model_name}...")
        
        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.qwen_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.qwen_model_name,
            trust_remote_code=True
        )
        
        print("✅ Qwen LLM loaded")
        return self
    
    @torch.no_grad()
    def encode_image_edge(self, image_tensor):
        """
        端侧编码图像
        
        Args:
            image_tensor: (B, 3, H, W) normalized image tensor
        Returns:
            visual_tokens: (B, num_tokens, 2048) 可传输到云端
        """
        visual_tokens = self.edge_encoder(image_tensor)
        return visual_tokens
    
    def generate_with_precomputed_vision(self, visual_tokens, prompt, max_length=512):
        """
        使用预计算的视觉 tokens 生成回复
        
        这是云端执行的部分,接收端侧传来的 visual_tokens
        
        Args:
            visual_tokens: (1, num_tokens, 2048) from edge device
            prompt: text prompt
            max_length: max generation length
        Returns:
            generated text
        """
        if self.qwen_model is None:
            raise RuntimeError("请先调用 load_qwen_llm() 加载 Qwen 模型")
        
        # 这里需要修改 Qwen 的 forward 来注入预计算的视觉 tokens
        # 具体实现需要 hook Qwen 的 embedding 层
        # 这是一个简化的演示
        
        # 1. 编码文本
        text_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 2. 获取文本 embeddings
        text_embeds = self.qwen_model.model.embed_tokens(text_inputs.input_ids)
        
        # 3. 拼接视觉和文本 embeddings
        # visual_tokens: (1, num_vis_tokens, 2048)
        # text_embeds: (1, num_text_tokens, 2048)
        combined_embeds = torch.cat([visual_tokens.to(text_embeds.dtype), text_embeds], dim=1)
        
        # 4. 创建 attention mask
        vis_mask = torch.ones(1, visual_tokens.shape[1], device=self.device)
        combined_mask = torch.cat([vis_mask, text_inputs.attention_mask], dim=1)
        
        # 5. 生成 (使用 inputs_embeds 而不是 input_ids)
        outputs = self.qwen_model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        # 6. 解码输出 (跳过视觉 tokens 部分)
        generated_text = self.tokenizer.decode(
            outputs[0][visual_tokens.shape[1]:], 
            skip_special_tokens=True
        )
        
        return generated_text


def get_image_transform(image_size=224):
    """图像预处理"""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def demo_edge_encoding():
    """演示端侧视觉编码"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, 
                        default='checkpoints/qwen_aligned/best_qwen_aligned.pth')
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--dummy', action='store_true')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=str, default=None,
                        help='Save visual tokens to file (for cloud transmission)')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"🔧 Device: {device}")
    
    # 加载端侧编码器
    encoder = EdgeVisualEncoder(
        student_model='mobilenetv2_100',
        student_layer=3,
        student_channels=96,
        llm_hidden_size=2048
    )
    
    if Path(args.checkpoint).exists():
        encoder.load_checkpoint(args.checkpoint, device)
    else:
        print(f"⚠️ Checkpoint not found: {args.checkpoint}")
        print("   Using untrained model for demo...")
    
    encoder = encoder.to(device)
    encoder.eval()
    
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
    
    # 编码
    with torch.no_grad():
        visual_tokens = encoder(image)
    
    print(f"\n✅ 视觉 tokens 形状: {visual_tokens.shape}")
    print(f"   预期 Qwen LLM 输入: (B, num_tokens, 2048)")
    print(f"   Token 数量: {visual_tokens.shape[1]}")
    print(f"   隐藏维度: {visual_tokens.shape[2]}")
    print(f"   值范围: [{visual_tokens.min():.4f}, {visual_tokens.max():.4f}]")
    print(f"   均值: {visual_tokens.mean():.4f}")
    
    # 计算传输大小
    token_bytes = visual_tokens.numel() * 4  # float32
    print(f"\n📡 传输数据量估算:")
    print(f"   Float32: {token_bytes / 1024:.2f} KB")
    print(f"   Float16: {token_bytes / 2 / 1024:.2f} KB")
    print(f"   Int8量化: {token_bytes / 4 / 1024:.2f} KB")
    
    # 保存 tokens
    if args.output:
        torch.save({
            'visual_tokens': visual_tokens.cpu(),
            'shape': visual_tokens.shape,
        }, args.output)
        print(f"\n💾 视觉 tokens 已保存: {args.output}")
        print(f"   可传输到云端用于 Qwen LLM 推理")
    
    return visual_tokens


if __name__ == '__main__':
    demo_edge_encoding()

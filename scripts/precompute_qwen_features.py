"""
Qwen2.5-VL 视觉特征预计算脚本

离线提取所有图片的 Qwen 视觉特征并保存，支持断点续传。
训练时直接加载预计算的特征，无需再运行 Qwen。

Usage:
    # 预计算所有特征
    python precompute_qwen_features.py --data_dir ./data/imagenette2-320 --output_dir ./data/qwen_features
    
    # 限制样本数量（用于测试）
    python precompute_qwen_features.py --data_dir ./data/imagenette2-320 --max_samples 100
    
    # 从断点继续
    python precompute_qwen_features.py --data_dir ./data/imagenette2-320 --resume
"""
import argparse
import torch
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision import transforms


class QwenFeatureExtractor:
    """
    Qwen2.5-VL 视觉特征提取器
    
    支持提取中间层特征（浅层更容易被 CNN 学习）
    
    Qwen ViT 结构:
        patch_embed → [Block 0-31] → merger
                        ↑
                    可在任意层提取
    """
    
    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct", device='cuda', 
                 extract_layer=8):
        """
        Args:
            extract_layer: 提取哪一层的输出 (1-32)
                - 8: 浅层，容易学习 (默认推荐)
                - 16: 中层
                - 32: 深层 (原始行为，等同于 merger 输入)
                - -1: 最终 merger 输出 (2048 dim，非常难)
        """
        self.model_name = model_name
        self.device = device
        self.extract_layer = extract_layer
        self.model = None
        self.processor = None
        self.total_layers = 32  # Qwen 3B 有 32 层
        
    def load(self):
        """加载 Qwen 模型"""
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        print(f"📥 Loading Qwen2.5-VL from {self.model_name}...")
        
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
        
        # 获取实际层数
        self.total_layers = len(self.model.visual.blocks)
        print(f"✅ Model loaded (ViT has {self.total_layers} layers)")
        print(f"📍 Will extract from layer {self.extract_layer}")
        
        return self
    
    @torch.no_grad()
    def extract_features(self, pil_image):
        """
        提取指定层的视觉特征
        
        Args:
            pil_image: PIL.Image 对象
        Returns:
            features: (num_tokens, hidden_size) tensor
                - 中间层: hidden_size = 1280
                - merger 输出: hidden_size = 2048
        """
        # 构造消息
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": pil_image}]
        }]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[pil_image],
            return_tensors="pt",
            padding=True
        )
        
        pixel_values = inputs["pixel_values"].to(self.device)
        grid_thw = inputs["image_grid_thw"].to(self.device)
        
        if self.extract_layer == -1:
            # 提取最终 merger 输出 (原始行为)
            hidden_states = self.model.visual(
                pixel_values.to(self.model.visual.patch_embed.proj.weight.dtype),
                grid_thw=grid_thw
            )
        else:
            # 提取中间层
            hidden_states = self._extract_intermediate_layer(
                pixel_values.to(self.model.visual.patch_embed.proj.weight.dtype),
                grid_thw=grid_thw
            )
        
        return hidden_states.cpu()
    
    def _extract_intermediate_layer(self, pixel_values, grid_thw):
        """
        手动执行 forward 并在指定层停止
        
        完全匹配 Qwen2_5_VisionTransformerPretrainedModel.forward() 实现
        """
        import torch.nn.functional as F
        
        visual = self.model.visual
        
        # 1. Patch embedding
        hidden_states = visual.patch_embed(pixel_values)
        
        # 2. Rotary position embedding
        rotary_pos_emb = visual.rot_pos_emb(grid_thw)
        
        # 3. Window indexing (关键步骤)
        window_index, cu_window_seqlens = visual.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        
        # 4. 重排 hidden_states
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // visual.spatial_merge_unit, visual.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        
        # 5. 重排 rotary_pos_emb 并创建 position_embeddings
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // visual.spatial_merge_unit, visual.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        
        # 6. cu_seqlens
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], 
            grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        
        # 7. 逐层执行 blocks
        for layer_num, blk in enumerate(visual.blocks):
            # 选择正确的 cu_seqlens
            if layer_num in visual.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
            )
            
            # 在指定层停止
            if layer_num == self.extract_layer - 1:
                break
        
        # 需要反转 window indexing 以恢复原始顺序
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states.view(seq_len // visual.spatial_merge_unit, visual.spatial_merge_unit, -1)
        hidden_states = hidden_states[reverse_indices, :, :]
        hidden_states = hidden_states.view(seq_len, -1)
        
        return hidden_states  # (num_tokens, 1280)





def get_image_paths(data_dir, split='train'):
    """获取所有图片路径"""
    data_path = Path(data_dir) / split
    if not data_path.exists():
        raise ValueError(f"路径不存在: {data_path}")
    
    # 使用 ImageFolder 获取所有图片
    dataset = ImageFolder(data_path)
    
    paths = []
    for idx in range(len(dataset)):
        img_path, label = dataset.samples[idx]
        paths.append({
            'path': img_path,
            'label': label,
            'idx': idx
        })
    
    return paths


def load_checkpoint(checkpoint_path):
    """加载检查点"""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {'processed': [], 'last_idx': -1}


def save_checkpoint(checkpoint_path, processed_items, last_idx):
    """保存检查点"""
    with open(checkpoint_path, 'w') as f:
        json.dump({
            'processed': processed_items,
            'last_idx': last_idx
        }, f)


def main():
    parser = argparse.ArgumentParser(description='Precompute Qwen Vision Features')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data/imagenette2-320',
                        help='ImageNet-style data directory')
    parser.add_argument('--output_dir', type=str, default='./data/qwen_features',
                        help='Output directory for features')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val'], help='Which split to process')
    
    # 控制参数
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples to process (for testing)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    
    # 模型参数
    parser.add_argument('--qwen_model', type=str, default='Qwen/Qwen2.5-VL-3B-Instruct',
                        help='Qwen model name')
    parser.add_argument('--layer', type=int, default=8,
                        help='Which ViT layer to extract (1-32, default 8 for shallow, -1 for merger output)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查点路径
    checkpoint_path = output_dir / 'checkpoint.json'
    
    # 获取所有图片路径
    print(f"📂 Scanning {args.data_dir}/{args.split}...")
    all_images = get_image_paths(args.data_dir, args.split)
    total_images = len(all_images)
    print(f"   Found {total_images} images")
    
    # 限制样本数量
    if args.max_samples:
        all_images = all_images[:args.max_samples]
        print(f"   Limited to {len(all_images)} samples")
    
    # 加载检查点
    if args.resume:
        checkpoint = load_checkpoint(checkpoint_path)
        processed_set = set(checkpoint['processed'])
        print(f"📌 Resuming from checkpoint: {len(processed_set)} already processed")
    else:
        processed_set = set()
    
    # 过滤已处理的
    to_process = [img for img in all_images if img['path'] not in processed_set]
    print(f"📋 To process: {len(to_process)} images")
    
    if len(to_process) == 0:
        print("✅ All images already processed!")
        return
    
    # 加载模型
    extractor = QwenFeatureExtractor(
        model_name=args.qwen_model,
        device=args.device,
        extract_layer=args.layer
    ).load()
    
    # 预处理 transform (只做基本的 resize)
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
    ])
    
    # 开始处理
    processed_list = list(processed_set)
    errors = []
    
    print(f"\n🚀 Starting feature extraction...")
    pbar = tqdm(to_process, desc="Extracting")
    
    for item in pbar:
        img_path = item['path']
        idx = item['idx']
        label = item['label']
        
        try:
            # 加载并预处理图像
            pil_img = Image.open(img_path).convert('RGB')
            pil_img = transform(pil_img)
            
            # 提取特征
            features = extractor.extract_features(pil_img)
            
            # 保存特征
            # 使用 idx 作为文件名，方便后续对应
            feature_path = output_dir / f"{idx:06d}.pt"
            torch.save({
                'features': features,
                'label': label,
                'path': img_path,
                'num_tokens': features.shape[0],
                'hidden_size': features.shape[1]
            }, feature_path)
            
            # 更新进度
            processed_list.append(img_path)
            
            # 定期保存检查点
            if len(processed_list) % 50 == 0:
                save_checkpoint(checkpoint_path, processed_list, idx)
                pbar.set_postfix({'saved': len(processed_list)})
                
        except Exception as e:
            errors.append({'path': img_path, 'error': str(e)})
            pbar.set_postfix({'errors': len(errors)})
    
    # 最终保存检查点
    save_checkpoint(checkpoint_path, processed_list, -1)
    
    # 保存元数据
    hidden_size = 2048 if args.layer == -1 else 1280  # merger output vs intermediate
    metadata = {
        'total_processed': len(processed_list),
        'total_errors': len(errors),
        'hidden_size': hidden_size,
        'extract_layer': args.layer,
        'split': args.split,
        'data_dir': str(args.data_dir),
        'qwen_model': args.qwen_model
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    if errors:
        with open(output_dir / 'errors.json', 'w') as f:
            json.dump(errors, f, indent=2)
    
    print(f"\n✅ Done!")
    print(f"   Processed: {len(processed_list)}")
    print(f"   Errors: {len(errors)}")
    print(f"   Output: {output_dir}")


if __name__ == '__main__':
    main()

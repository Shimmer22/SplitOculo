"""
Qwen2.5-VL 视觉特征提取器

可复用模块，支持:
- 中间层特征提取 (layer 1-32)
- 最终 merger 输出 (layer -1)
- 训练时动态提取或离线预计算
"""
import torch
import torch.nn.functional as F
from PIL import Image


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
            model_name: Qwen 模型名称或本地路径
            device: 运行设备
            extract_layer: 提取哪一层的输出 (1-32)
                - 4: 非常浅层 (推荐用于训练)
                - 8: 浅层
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
        self._loaded = False
        
    def load(self):
        """加载 Qwen 模型"""
        if self._loaded:
            return self
            
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        print(f"Loading Qwen2.5-VL from {self.model_name}...")
        
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
        print(f"Model loaded (ViT has {self.total_layers} layers, extract layer {self.extract_layer})")
        
        self._loaded = True
        return self
    
    @property
    def hidden_size(self):
        """返回特征维度"""
        return 2048 if self.extract_layer == -1 else 1280
    
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
        if not self._loaded:
            self.load()
            
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
    
    @torch.no_grad()
    def extract_features_batch(self, pil_images):
        """
        批量提取特征
        
        Args:
            pil_images: PIL.Image 对象列表
        Returns:
            features_list: [(num_tokens, hidden_size), ...] tensor 列表
        """
        # 由于不同图片 token 数量可能不同，逐个处理
        return [self.extract_features(img) for img in pil_images]
    
    def _extract_intermediate_layer(self, pixel_values, grid_thw):
        """
        手动执行 forward 并在指定层停止
        
        完全匹配 Qwen2_5_VisionTransformerPretrainedModel.forward() 实现
        """
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

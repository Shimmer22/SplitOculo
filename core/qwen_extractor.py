"""
Qwen2.5-VL 视觉特征提取器

可复用模块，支持:
- 像素 patch 提取 (layer -1): patch_embed 之前的原始像素块
- 中间层特征提取 (layer 0-32): patch_embed 后 / 各 block 后
- 训练时动态提取或离线预计算

层级语义:
  layer -1 : 原始像素 patches (patch_embed 输入)  dim = 3*patch_h*patch_w
  layer  0 : patch_embed 输出 (无 transformer block)  dim = 1280
  layer  N : 经过 N 个 transformer block 之后的输出   dim = 1280
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
                 extract_layer=4, local_files_only=False, min_pixels=None,
                 max_pixels=None):
        """
        Args:
            model_name: Qwen 模型名称或本地路径
            device: 运行设备
            extract_layer: 提取哪一层的输出
                - -1: 原始像素 patches (patch_embed 输入)，用于 JPEG 级别对齐
                       dim = 3 * patch_height * patch_width (取决于 Qwen 配置)
                -  0: patch_embed 输出，不经过任何 transformer block  dim = 1280
                -  4: 经过 4 个 block 之后 (默认，推荐)  dim = 1280
                -  8: 经过 8 个 block 之后  dim = 1280
        """
        self.model_name = model_name
        self.device = device
        self.extract_layer = extract_layer
        self.local_files_only = local_files_only
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.model = None
        self.processor = None
        self.total_layers = 32  # Qwen 3B 有 32 层
        self._pixel_patch_dim = None  # 延迟初始化，加载模型后确定
        self._loaded = False
        
    def load(self):
        """加载 Qwen 模型"""
        if self._loaded:
            return self
            
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        
        print(f"Loading Qwen2.5-VL from {self.model_name}...")
        
        torch_dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        device_map = "cpu" if self.device == "cpu" else "auto"

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        processor_kwargs = {
            "trust_remote_code": True,
            "local_files_only": self.local_files_only,
        }
        if self.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.min_pixels
        if self.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.max_pixels
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            **processor_kwargs,
        )
        # In current transformers releases the top-level min/max pixel kwargs
        # can configure the image processor without updating the video
        # processor's active ``size`` dictionary. Set both explicitly so a
        # requested 224-scale video does not silently run near source resolution.
        if hasattr(self.processor, "video_processor"):
            video_size = dict(self.processor.video_processor.size)
            if self.min_pixels is not None:
                video_size["shortest_edge"] = int(self.min_pixels)
            if self.max_pixels is not None:
                video_size["longest_edge"] = int(self.max_pixels)
            self.processor.video_processor.size = video_size
        
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        # 获取实际层数及 patch 维度
        self.total_layers = len(self.model.visual.blocks)
        # 计算 pixel patch dim: patch_embed 的 proj 权重 shape = (out_ch, in_ch, kH, kW)
        # in_ch * kH * kW = pixel_patch_dim
        proj_w = self.model.visual.patch_embed.proj.weight
        self._pixel_patch_dim = proj_w.shape[1] * proj_w.shape[2] * proj_w.shape[3]
        print(f"Model loaded (ViT has {self.total_layers} layers, extract layer {self.extract_layer})")
        if self.extract_layer == -1:
            print(f"  Mode: pixel patches (JPEG level), dim={self._pixel_patch_dim}")
        elif self.extract_layer == 0:
            print(f"  Mode: patch_embed output (no blocks), dim=1280")
        else:
            print(f"  Mode: after {self.extract_layer} transformer blocks, dim=1280")
        
        self._loaded = True
        return self
    
    @property
    def hidden_size(self):
        """返回特征维度"""
        if self.extract_layer == -1:
            # 像素 patch 维度 (延迟到模型加载后才能确定)
            return self._pixel_patch_dim if self._pixel_patch_dim else 588  # 3*14*14 fallback
        return 1280
    
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
            # 提取 pixel patches: patch_embed 的输入 (JPEG 级别对齐)
            hidden_states = self._extract_pixel_patches(pixel_values)
        else:
            # 提取中间层 (layer 0 = patch_embed, layer N = after N blocks)
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

    @torch.no_grad()
    def extract_video_features(self, pil_frames):
        """
        Extract visual tokens from a video represented as PIL frames.

        Args:
            pil_frames: list[PIL.Image.Image], sampled in temporal order.

        Returns:
            features: (num_tokens, hidden_size) float32 CPU tensor.
            video_grid_thw: (3,) CPU tensor containing T, H, W grid.
        """
        if not self._loaded:
            self.load()
        if not pil_frames:
            raise ValueError("pil_frames must contain at least one frame")

        messages = [{
            "role": "user",
            "content": [{"type": "video", "video": "sampled_frames"}],
        }]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            videos=[pil_frames],
            return_tensors="pt",
            padding=True,
        )

        if "pixel_values_videos" not in inputs or "video_grid_thw" not in inputs:
            raise RuntimeError(
                "Processor did not return video tensors. Check transformers/Qwen2.5-VL video support."
            )

        pixel_values = inputs["pixel_values_videos"].to(self.device)
        grid_thw = inputs["video_grid_thw"].to(self.device)

        if self.extract_layer == -1:
            hidden_states = self._extract_pixel_patches(pixel_values)
        else:
            hidden_states = self._extract_intermediate_layer(
                pixel_values.to(self.model.visual.patch_embed.proj.weight.dtype),
                grid_thw=grid_thw,
            )

        return hidden_states.cpu(), grid_thw[0].detach().cpu()
    
    def _extract_pixel_patches(self, pixel_values):
        """
        提取 pixel patches：patch_embed 输入前的原始像素块。

        在 Qwen2.5-VL 中，pixel_values 的 shape 为:
            (N_patches, C, patch_h, patch_w)  e.g. (256, 3, 14, 14)
        展平后即为 (N_patches, C*patch_h*patch_w)。

        Returns:
            (N_patches, C*patch_h*patch_w) float32 tensor
        """
        # pixel_values: (N_patches, C, pH, pW)
        # 仅做 flatten，不过任何网络层
        N = pixel_values.shape[0]
        patches = pixel_values.float().view(N, -1)  # (N, C*pH*pW)
        return patches.cpu()

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
        if self.extract_layer > 0:
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

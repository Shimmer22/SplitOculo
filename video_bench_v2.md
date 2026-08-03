# Video Inference Benchmark — Final Report

## Test Config

| Item | Value |
|---|---|
| Video | 1920×1080, 29.97fps, 8m17s, 14,896 frames |
| Task | "During the middle part, the wearer picked items in front of which product area?" |
| Ground truth | **乳制品 (Dairy)** |
| Frame sampling | **Uniform** across full video (identical frames for Qwen & SO) |
| Model | Qwen/Qwen2.5-VL-3B-Instruct |
| Resolution | Qwen max=448×448 (42×24 grid, 1008 patches/frame), SO edge=224×224 |
| Max new tokens | 32 |
| Timing | From frame input to first_token / total_end. **Models pre-loaded. Frame decode excluded (identical cost).** |
| Device | NVIDIA A100-SXM4-80GB |

## Training Config

| Item | Value |
|---|---|
| Dataset | CC3M 50k (45k train / 5k val) |
| Teacher | Qwen2.5-VL layer 4 |
| Backbone | MobileNetV2-100 → StridedTokenProjector → 196t×1280d |
| Bottleneck | Linear 1280→128 |
| Upsampler | TransformerUpsampler 4-layer (pixelshuffle) |
| Payload levels | 49×64, 49×128, 196×64, 196×128 |
| Batch | 512, AMP, Warmup 20ep + GAN 30ep |
| Best val_cos_sim | **0.9251** |
| Edge / Cloud | 2.73M (10.5MB) / 126.71M (486.5MB) |

---

## Results: Answer Accuracy

| Method | 2f | 4f | 8f | 16f | 32f | 64f | 128f |
|--------|----|----|----|----|----|----|----|
| **Qwen Native (Video API)** | 零食区 | 饮料区 | **乳制品** ✅ | 零食区 | **乳制品** ✅ | 酸奶 | 酸奶 |
| **SO 49×64** | 饮料 | 饮料 | Beverages | **Dairy** ✅ | Beverages | Beverages | Bread |
| **SO 49×128** | 饮料 | 饮料 | Beverages | Beverages | Beverages | Beverages | Bread |
| **SO 196×64** | **Dairy** ✅ | **Dairy** ✅ | **Dairy** ✅ | **dairy** ✅ | Beverages | **Dairy** ✅ | **Dairy** ✅ |
| **SO 196×128** | **Dairy** ✅ | **Dairy** ✅ | **Dairy** ✅ | **dairy** ✅ | **Dairy** ✅ | **Dairy** ✅ | **Dairy** ✅ |

**Correct count**: Qwen 2/7, SO 49×64 1/7, SO 49×128 0/7, SO 196×64 5/7, **SO 196×128 6/7**

## Results: Total Latency (s)

| Method | 2f | 4f | 8f | 16f | 32f | 64f | 128f |
|--------|-----|-----|-----|-----|-----|-----|-----|
| **Qwen Native (Video API)** | 0.56 | 0.19 | 0.27 | 0.42 | 0.73 | 1.50 | 2.78 |
| **SO 49×64** | 3.30* | 0.41 | 0.57 | 1.03 | 1.87 | 3.74 | 7.45 |
| **SO 49×128** | 0.24 | 0.54 | 0.57 | 0.96 | 1.88 | 4.49 | 7.08 |
| **SO 196×64** | 0.29 | 0.40 | 0.54 | 1.21 | 2.12 | 4.19 | 7.69 |
| **SO 196×128** | 0.27 | 0.40 | 0.54 | 1.04 | 1.85 | 3.66 | 7.35 |

> *First SO run includes one-time Qwen model loading. Subsequent runs reuse the loaded model.

## Results: First Token Latency (s)

| Method | 2f | 4f | 8f | 16f | 32f | 64f | 128f |
|--------|-----|-----|-----|-----|-----|-----|-----|
| **Qwen Native** | 0.48 | 0.14 | 0.22 | 0.37 | 0.67 | 1.47 | 2.75 |
| **SO 196×128** | 0.12 | 0.14 | 0.12 | 0.13 | 0.17 | 0.24 | 0.41 |

## Results: Edge Encode vs Cloud Infer Breakdown — SO 196×128

| Frames | Edge(s) | Cloud(s) | Total(s) | Payload |
|--------|---------|----------|----------|---------|
| 2 | 0.11 | 0.17 | 0.27 | 50 KB |
| 4 | 0.19 | 0.21 | 0.40 | 100 KB |
| 8 | 0.32 | 0.22 | 0.54 | 201 KB |
| 16 | 0.72 | 0.32 | 1.04 | 401 KB |
| 32 | 1.31 | 0.54 | 1.85 | 803 KB |
| 64 | 2.62 | 1.03 | 3.66 | 1.6 MB |
| 128 | 5.44 | 1.91 | 7.35 | 3.2 MB |

Edge encode dominates at higher frame counts (74% of total at 128f). Cloud (28 ViT layers + LLM) stays under 2s even at 128 frames.

## Qwen vs SO — Head-to-Head (196×128)

| Frames | Qwen Total | SO Total | Qwen faster? | Qwen Acc | SO Acc |
|--------|-----------|----------|-------------|----------|--------|
| 2 | 0.56s | 0.27s | ❌ (SO 2.1×) | ❌ | **✅** |
| 4 | 0.19s | 0.40s | ✅ 2.1× | ❌ | **✅** |
| 8 | 0.27s | 0.54s | ✅ 2.0× | ✅ | **✅** |
| 16 | 0.42s | 1.04s | ✅ 2.5× | ❌ | **✅** |
| 32 | 0.73s | 1.85s | ✅ 2.5× | ✅ | **✅** |
| 64 | 1.50s | 3.66s | ✅ 2.4× | ❌ | **✅** |
| 128 | 2.78s | 7.35s | ✅ 2.6× | ❌ | **✅** |

---

## Key Findings

1. **Qwen Native is ~2.5× faster than SO** (not slower). This is expected because Qwen uses temporal patch merging (2 frames → 1) and processes all frames in one forward pass with efficient window attention, while SO processes each frame individually through 28 ViT layers plus a per-frame edge encoder.

2. **SO is more accurate** — SO 196×128 answers correctly at 6/7 frame counts (2-128f), while Qwen only at 2/7 (8f, 32f). Qwen's answers are inconsistent across frame counts.

3. **Payload matters**: 196×64/128 consistently output "Dairy". 49×64/128 suffer from insufficient spatial resolution (49 tokens) and tend toward generic "Beverages" or hallucinated "Bread".

4. **First-token latency**: SO's FT is remarkably stable (0.12-0.41s) vs Qwen's linear scaling (0.14-2.75s). SO first-token is faster than Qwen at 32f+ because the LLM receives pre-computed visual tokens without waiting for ViT.

5. **Edge encoding dominates SO latency** at higher frame counts (74% at 128f). The per-frame MobileNetV2 + projector + bottleneck encoding is the scalability bottleneck. Batching or parallelization could improve this.

6. **SO trades latency for bandwidth**: at 64 frames, SO transmits 1.6 MB of compressed features vs needing the full frames at the cloud. The value proposition is bandwidth reduction, not inference latency.

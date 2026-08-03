# Video Inference Benchmark Results

## Test Config

| Item | Value |
|---|---|
| Video | `data/supermarket.mp4` (1920x1080, H.264, 29.97fps, 8m17s, 14896 frames) |
| Model | Qwen/Qwen2.5-VL-3B-Instruct |
| Prompt | These are uniformly sampled first-person supermarket video frames. Question: During the middle part of the video, the wearer selected or picked items in front of which product area? Answer in one short Chinese noun phrase. Describe only visible product category or shelf area. |
| Ground truth | 乳制品 (dairy products) |
| Max new tokens | 32 |
| Split layer | 4 |
| Device | NVIDIA A100-SXM4-80GB |

## Training Config

| Item | Value |
|---|---|
| Dataset | CC3M ~50k (45k train / 5k val) |
| Teacher layer | Qwen2.5-VL layer 4 |
| CNN backbone | MobileNetV2-100 |
| Projector | Strided token projector |
| Upsampler | TransformerUpsampler (4 layers, pixelshuffle) |
| Bottleneck | Linear 1280→128 |
| Transmission tokens | 196 |
| Payload levels | 49x64, 49x128, 196x64, 196x128 |
| Batch size | 512 |
| Phase 1 Warmup | 20 epochs, lr=1e-4, MSE only |
| Phase 2 GAN | 30 epochs, lr_G=1e-4, lr_D=4e-5, λ_mse=10, λ_adv=0.1 |
| Best val_cos_sim | 0.9251 |
| Edge model | 2.73M params, 10.5 MB |
| Cloud model | 126.71M params, 486.5 MB |

## Results

### Qwen Native (Baseline)

| Label | Frames | Answer | FT(s) | Total(s) | Gen tokens | TPS |
|---|---|---|---|---|---|---|
| Q2 | 2 | 果冻 | 0.993 | 3.128 | 3 | 2.8 |
| Q4 | 4 | 果冻 | 0.865 | 5.115 | 3 | 3.3 |
| Q8 | 8 | 果冻 | 1.679 | 9.656 | 3 | 1.7 |
| Q16 | 16 | 果冻 | 3.479 | 19.691 | 3 | 0.8 |
| Q32 | 32 | 果冻 | 7.832 | 40.250 | 3 | 0.4 |
| Q64 | 64 | 零食 | 18.508 | 83.347 | 2 | 0.1 |

### SplitOculo (Multi-Level Payload)

#### Payload: 49x64 (3.1 KB / frame)

| Label | Frames | Answer | FT(s) | Total(s) | Payload(bytes) |
|---|---|---|---|---|---|
| S4964-2 | 2 | 饮料区 | 0.064 | 16.524* | 6272 |
| S4964-4 | 4 | Frozen foods | 0.109 | 0.393 | 12544 |
| S4964-8 | 8 | Frozen foods | 0.091 | 0.588 | 25088 |
| S4964-16 | 16 | Frozen food | 0.101 | 1.026 | 50176 |
| S4964-32 | 32 | Frozen food | 0.133 | 1.878 | 100352 |
| S4964-64 | 64 | Frozen food | 0.212 | 3.893 | 200704 |

#### Payload: 49x128 (6.1 KB / frame)

| Label | Frames | Answer | FT(s) | Total(s) | Payload(bytes) |
|---|---|---|---|---|---|
| S49128-2 | 2 | 饮料区 | 0.061 | 0.279 | 12544 |
| S49128-4 | 4 | Frozen foods | 0.090 | 0.375 | 25088 |
| S49128-8 | 8 | Frozen foods | 0.094 | 0.774 | 50176 |
| S49128-16 | 16 | Frozen food | 0.102 | 0.999 | 100352 |
| S49128-32 | 32 | Frozen food | 0.135 | 2.085 | 200704 |
| S49128-64 | 64 | Frozen food | 0.212 | 3.858 | 401408 |

#### Payload: 196x64 (12.3 KB / frame)

| Label | Frames | Answer | FT(s) | Total(s) | Payload(bytes) |
|---|---|---|---|---|---|
| S19664-2 | 2 | Frozen food | 0.088 | 0.292 | 25088 |
| S19664-4 | 4 | Frozen food | 0.090 | 0.421 | 50176 |
| S19664-8 | 8 | Frozen food | 0.092 | 0.581 | 100352 |
| S19664-16 | 16 | Frozen food | 0.101 | 0.995 | 200704 |
| S19664-32 | 32 | Frozen food | 0.134 | 1.841 | 401408 |
| S19664-64 | 64 | Frozen food | 0.212 | 3.631 | 802816 |

#### Payload: 196x128 (24.5 KB / frame)

| Label | Frames | Answer | FT(s) | Total(s) | Payload(bytes) |
|---|---|---|---|---|---|
| S196128-2 | 2 | Frozen foods | 0.088 | 0.267 | 50176 |
| S196128-4 | 4 | Frozen food | 0.089 | 0.390 | 100352 |
| S196128-8 | 8 | Frozen food | 0.091 | 0.584 | 200704 |
| S196128-16 | 16 | Frozen food | 0.102 | 0.992 | 401408 |
| S196128-32 | 32 | Frozen food | 0.135 | 1.850 | 802816 |
| S196128-64 | 64 | Frozen food | 0.211 | 3.813 | 1605632 |

> *S4964-2 total includes one-time Qwen model loading (~16s). All subsequent runs reuse the loaded model.

### Speedup Summary (Total Latency, Q64 vs S*-64)

| Method | Total(s) | Speedup vs Qwen |
|---|---|---|
| Q64 (baseline) | 83.347 | 1.0x |
| S4964-64 | 3.893 | **21.4x** |
| S49128-64 | 3.858 | **21.6x** |
| S19664-64 | 3.631 | **23.0x** |
| S196128-64 | 3.813 | **21.9x** |

## Answer Accuracy

Ground truth: **乳制品**

| Method | Answers | Match? |
|---|---|---|
| Qwen (Q2-Q32, first-N) | 果冻 | ❌ 零食/糖果 |
| Qwen (Q64, first-N) | 零食 | ❌ |
| Qwen (extended, uniform 8f) | **乳制品** | ✅ |
| Qwen (extended, uniform 32f) | **乳制品** | ✅ |
| Qwen (extended, uniform 128f) | 酸奶 | ❌ (related: dairy) |
| Qwen (extended, uniform 256f) | 乳品 | ❌ (related: dairy) |
| SO (S*-2) | 饮料区 | ❌ |
| SO (S*-4~S*-64) | Frozen food(s) | ❌ |

**Critical finding**: Qwen Native CAN answer correctly when frames are uniformly sampled across the full video (not just the beginning). At 8 and 32 uniformly sampled frames, Qwen correctly outputs `乳制品`. SplitOculo (currently limited to first-N frames) never produces the correct answer.

## Qwen Extended — Uniform Frame Sampling (Full Video)

Frames uniformly sampled across entire 14,896-frame video. `max_pixels=448x448`.

| Frames | Answer | FT(s) | Total(s) | Gen(s) | Tokens | TPS |
|---|---|---|---|---|---|---|
| 2 | 零食区 | 0.839 | 2.989 | 1.101 | 4 | 3.6 |
| 4 | 饮料区 | 0.146 | 4.444 | 0.570 | 4 | 7.0 |
| **8** | **乳制品** ✅ | 0.223 | 8.312 | 0.578 | 4 | 6.9 |
| 16 | 零食区 | 0.368 | 16.501 | 0.656 | 3 | 4.6 |
| **32** | **乳制品** ✅ | 0.902 | 32.251 | 1.195 | 3 | 2.5 |
| 64 | 酸奶 | 1.299 | 62.631 | 1.757 | 4 | 2.3 |
| 128 | 酸奶 | 2.679 | 136.781 | 3.426 | 4 | 1.2 |
| 256 | 乳品 | 5.692 | 261.612 | 6.718 | 3 | 0.4 |

> Note: Uniform sampling is key — the "middle part" (dairy aisle) is only visible when frames span the full video duration. The previous Qwen results used first-N frames and missed the dairy section entirely.

### First Token Latency Summary (s)

| Frames | Qwen | S4964 | S49128 | S19664 | S196128 |
|---|---|---|---|---|---|
| 2 | 0.993 | 0.064 | 0.061 | 0.088 | 0.088 |
| 4 | 0.865 | 0.109 | 0.090 | 0.090 | 0.089 |
| 8 | 1.679 | 0.091 | 0.094 | 0.092 | 0.091 |
| 16 | 3.479 | 0.101 | 0.102 | 0.101 | 0.102 |
| 32 | 7.832 | 0.133 | 0.135 | 0.134 | 0.135 |
| 64 | 18.508 | 0.212 | 0.212 | 0.212 | 0.211 |

## Raw Data

Full JSON: `video_bench_results.json`

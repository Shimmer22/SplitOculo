# Qwen Native Video Inference — Extended Frame Test

- Video: `/data/downloads/20260115_0001_dajiangaction5pro_lingshouxiaofei_chaoshi_0001.MP4` (14896 frames, 29.97 fps)
- Model: Qwen/Qwen2.5-VL-3B-Instruct
- Ground truth: **乳制品**
- Prompt: These are uniformly sampled first-person supermarket video frames. Question: During the middle part of the video, the wearer selected or picked items in front of which product area? Answer in one short Chinese noun phrase. Describe only visible product category or shelf area.

| Frames | Answer | FT(s) | Total(s) | Gen(s) | Tokens | TPS |
|---|---|---|---|---|---|---|
| 2 | 零食区 | 0.839 | 2.989 | 0.955 | 3 | 3.1 |
| 4 | 饮料区 | 0.146 | 4.444 | 0.204 | 3 | 14.7 |
| 8 | 乳制品 | 0.223 | 8.312 | 0.281 | 3 | 10.7 |
| 16 | 零食区 | 0.368 | 16.501 | 0.426 | 3 | 7.0 |
| 32 | 乳制品 | 0.902 | 32.251 | 0.961 | 3 | 3.1 |
| 64 | 酸奶 | 1.299 | 62.631 | 1.332 | 2 | 1.5 |
| 128 | 酸奶 | 2.679 | 136.781 | 2.709 | 2 | 0.7 |
| 256 | 乳品 | 5.692 | 261.612 | 5.752 | 3 | 0.5 |

## RAW JSON
See `/workspace/SplitOculo/qwen_native_extended.json`

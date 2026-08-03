# Fair Comparison: Qwen Native Video API vs SplitOculo

- All models pre-loaded. Timing: frame decode (excluded, identical) → image → first_token / end
- Prompt: These are uniformly sampled first-person supermarket video frames. Question: During the middle part of the video, the wearer selected or picked items in front of which product area? Answer in one short Chinese noun phrase. Describe only visible product category or shelf area.
- Ground truth: 乳制品 / Dairy
- Resolution: Qwen max=448×448, SO edge=224×224

## Qwen Native (Video API)
| Frames | Answer | Match | FT(s) | Total(s) | Gen(s) | Tokens |
|---|---|---|---|---|---|---|
| 2 | 零食区 | ❌ | 0.477 | 0.558 | 0.558 | 3 |
| 4 | 饮料区 | ❌ | 0.135 | 0.191 | 0.191 | 3 |
| 8 | 乳制品 | ✅ | 0.216 | 0.272 | 0.272 | 3 |
| 16 | 零食区 | ❌ | 0.365 | 0.421 | 0.421 | 3 |
| 32 | 乳制品 | ✅ | 0.666 | 0.725 | 0.725 | 3 |
| 64 | 酸奶 | ❌ | 1.472 | 1.503 | 1.503 | 2 |
| 128 | 酸奶 | ❌ | 2.754 | 2.783 | 2.783 | 2 |

## SplitOculo — 49x64
| Frames | Answer | Match | Edge(s) | Cloud(s) | Total(s) | FT(s) | Payload |
|---|---|---|---|---|---|---|---|
| 2 | 饮料 | ❌ | 0.111 | 3.190 | 3.301 | 0.063 | 6,272 |
| 4 | 饮料 | ❌ | 0.258 | 0.154 | 0.412 | 0.061 | 12,544 |
| 8 | Beverages | ❌ | 0.323 | 0.245 | 0.568 | 0.144 | 25,088 |
| 16 | Dairy | ✅ | 0.713 | 0.315 | 1.028 | 0.127 | 50,176 |
| 32 | Beverages | ❌ | 1.314 | 0.557 | 1.871 | 0.193 | 100,352 |
| 64 | Beverages | ❌ | 2.623 | 1.119 | 3.741 | 0.304 | 200,704 |
| 128 | Bread | ❌ | 5.367 | 2.082 | 7.449 | 0.409 | 401,408 |

## SplitOculo — 49x128
| Frames | Answer | Match | Edge(s) | Cloud(s) | Total(s) | FT(s) | Payload |
|---|---|---|---|---|---|---|---|
| 2 | 饮料 | ❌ | 0.110 | 0.131 | 0.242 | 0.060 | 12,544 |
| 4 | 饮料 | ❌ | 0.389 | 0.152 | 0.541 | 0.059 | 25,088 |
| 8 | Beverages | ❌ | 0.323 | 0.246 | 0.568 | 0.146 | 50,176 |
| 16 | Beverages | ❌ | 0.621 | 0.343 | 0.964 | 0.156 | 100,352 |
| 32 | Beverages | ❌ | 1.316 | 0.561 | 1.877 | 0.195 | 200,704 |
| 64 | Beverages | ❌ | 3.384 | 1.102 | 4.486 | 0.319 | 401,408 |
| 128 | Bread | ❌ | 5.192 | 1.886 | 7.078 | 0.412 | 802,816 |

## SplitOculo — 196x64
| Frames | Answer | Match | Edge(s) | Cloud(s) | Total(s) | FT(s) | Payload |
|---|---|---|---|---|---|---|---|
| 2 | Dairy | ✅ | 0.127 | 0.167 | 0.294 | 0.118 | 25,088 |
| 4 | Dairy | ✅ | 0.186 | 0.214 | 0.400 | 0.144 | 50,176 |
| 8 | Dairy | ✅ | 0.322 | 0.215 | 0.537 | 0.115 | 100,352 |
| 16 | dairy | ✅ | 0.885 | 0.321 | 1.206 | 0.131 | 200,704 |
| 32 | Beverages | ❌ | 1.562 | 0.555 | 2.117 | 0.191 | 401,408 |
| 64 | Dairy | ✅ | 3.190 | 1.001 | 4.191 | 0.243 | 802,816 |
| 128 | Dairy | ✅ | 5.552 | 2.133 | 7.686 | 0.411 | 1,605,632 |

## SplitOculo — 196x128
| Frames | Answer | Match | Edge(s) | Cloud(s) | Total(s) | FT(s) | Payload |
|---|---|---|---|---|---|---|---|
| 2 | Dairy | ✅ | 0.106 | 0.166 | 0.272 | 0.115 | 50,176 |
| 4 | Dairy | ✅ | 0.188 | 0.212 | 0.400 | 0.142 | 100,352 |
| 8 | Dairy | ✅ | 0.321 | 0.218 | 0.539 | 0.117 | 200,704 |
| 16 | dairy | ✅ | 0.720 | 0.318 | 1.038 | 0.132 | 401,408 |
| 32 | Dairy | ✅ | 1.308 | 0.537 | 1.845 | 0.165 | 802,816 |
| 64 | Dairy | ✅ | 2.624 | 1.032 | 3.656 | 0.240 | 1,605,632 |
| 128 | Dairy | ✅ | 5.436 | 1.911 | 7.346 | 0.410 | 3,211,264 |

## Speedup vs Qwen Native (Total Latency)
| Frames | Qwen(s) | S196128(s) | Speedup | Qwen Acc | SO Acc |
|---|---|---|---|---|---|
| 2 | 0.6 | 0.3 | 2.0× | ❌ | ✅ |
| 4 | 0.2 | 0.4 | 0.5× | ❌ | ✅ |
| 8 | 0.3 | 0.5 | 0.5× | ✅ | ✅ |
| 16 | 0.4 | 1.0 | 0.4× | ❌ | ✅ |
| 32 | 0.7 | 1.8 | 0.4× | ✅ | ✅ |
| 64 | 1.5 | 3.7 | 0.4× | ❌ | ✅ |
| 128 | 2.8 | 7.3 | 0.4× | ❌ | ✅ |

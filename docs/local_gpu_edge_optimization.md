# Local-GPU edge inference optimization

## Scope

These measurements treat the local NVIDIA GPU as the edge device. They cover
only edge encoding: image preprocessing, MobileNetV2, token projector,
bottleneck, decoder-MV propagation, and optional LSFA memory. Video decoding,
model loading, network transfer, and cloud/VLM inference are excluded.

The fixed regression video is selected from the 75-video UCF101 test manifest
with seed `20260729`:

`BasketballDunk/v_BasketballDunk_g14_c06.avi`

Each run uses 16 frames at 224x224. Timing values are CUDA-synchronized.

## Recommended operating points

| Path | 16-frame latency | Payload quality | Intended use |
| --- | ---: | ---: | --- |
| Full CNN, batch 1 | 135.16 ms | Reference | Lowest buffering latency |
| Full CNN, batch 8 | **35.41 ms** | MSE `3.57e-5` vs batch 1 | Buffered/offline GPU edge |
| Warp-only, projection batch 4 | 60.34 ms | cosine `0.8392` vs full | Streaming, quality-risking |
| LSFA exact RGB, projection batch 4 | 99.14 ms | cosine `0.8579` vs full | Streaming, quality-preserving default |
| LSFA fast RGB, projection batch 4 | 90.84 ms | cosine `0.8559` vs full | Explicit speed/quality trade-off |

For buffered local-GPU inference, full CNN with `--edge_batch_size 8` is the
best operating point. It is 3.82x faster than sequential full-CNN encoding and
also avoids temporal approximation. The small batch-dependent payload
difference is normal floating-point convolution variation; cosine is
approximately 1.0.

For strict streaming where frames must be emitted immediately, batching is not
free because it adds queueing delay. The exact-RGB LSFA path remains available,
but on this GPU it is only 1.15x faster than sequential full CNN. Its benefit is
larger on the 75-video quality aggregate than on the fixed sample, but it should
not replace full-CNN batch 8 when buffering is acceptable.

## Implemented optimizations

- `EdgeEncoder.encode_pil_batch()` runs preprocessing, backbone, projector, and
  bottleneck as a microbatch.
- `--edge_batch_size` enables microbatching in non-codec video inference.
- Codec memory caches appearance at the 14x14 feature grid instead of moving a
  224x224 RGB tensor to CUDA for every P-frame.
- `--codec_memory_rgb_mode exact` preserves the training-time resize semantics.
- `--codec_memory_rgb_mode fast` directly builds the feature-grid RGB input.
- `--codec_projection_batch_size` defers only projector/bottleneck work and
  microbatches it after causal temporal feature propagation.

## Rejected optimizations on this GPU

| Attempt | Observation |
| --- | --- |
| FP16 backbone | About 13% slower than FP32 NCHW |
| FP16 LSFA memory | Slightly slower than FP32 |
| Channels-last backbone | About 2% improvement, within run-to-run variance |
| TorchScript-frozen LSFA | Regressed from about 95 ms to 186 ms |
| Key-frame composed warp | Lower cosine than recursive warp on the fixed sample |

These options are not exposed in the inference CLI because they did not improve
the stated local-GPU target.

## Reproduction

Full-CNN microbatch sweep:

```powershell
E:\anaconda\envs\cnn_vit\python.exe scripts\benchmark_edge_backbone.py `
  --video E:\datasets\ucf101_subset\extracted\UCF101_subset\test\BasketballDunk\v_BasketballDunk_g14_c06.avi `
  --edge_checkpoint checkpoints\cc3m10k_multilevel_layer4\split_gan_best\edge_weights.pth `
  --max_frames 16 --batch_size 8 --rounds 40 --device cuda `
  --output outputs\codec_memory_ucf101\final_backbone_batch8.json
```

Quality-preserving LSFA microbatch:

```powershell
E:\anaconda\envs\cnn_vit\python.exe scripts\benchmark_codec_mv_edge.py `
  --video E:\datasets\ucf101_subset\extracted\UCF101_subset\test\BasketballDunk\v_BasketballDunk_g14_c06.avi `
  --edge_checkpoint checkpoints\cc3m10k_multilevel_layer4\split_gan_best\edge_weights.pth `
  --memory_checkpoint outputs\codec_memory_ucf101\lsfa_ucf101_300_e35_payloadcos.pth `
  --memory_arch lsfa --memory_rgb_mode exact `
  --projection_batch_size 4 --max_frames 16 --rounds 40 --device cuda `
  --output outputs\codec_memory_ucf101\final_lsfa_exact_projection_batch4.json
```

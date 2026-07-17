# Decoder-MV codec acceleration: optimized prototype

## Scope

This is an edge-only A/B benchmark. Both paths use the same checkpoint and
produce the same number of payloads. Model loading and video decoding are
excluded; CUDA is synchronized around each complete measured path.

The default optimized implementation builds a compact NumPy flow lookup and
analytically composes the legacy resize/crop/bilinear sampling coordinates. It
gathers only the 16 source samples needed by each feature cell, transfers a
small `[2, Hf, Wf]` flow tensor, and uses PyTorch `grid_sample` on CPU or CUDA.
This preserves the old numerical semantics without performing two large flow
resizes or transferring a model-input-sized flow tensor to the GPU.

## Accuracy/speed trade-off

| Operating point | Device | Legacy dense | Equivalent feature-grid | Approximate center-grid |
| --- | --- | ---: | ---: | ---: |
| Codec/VLM 2 fps, 4 outputs, I+3P | CPU | 62.27 ms | 41.33 ms | 40.02 ms |
| Codec/VLM 2 fps, 4 outputs, I+3P | CUDA | 35.33 ms | 15.74 ms | 13.85 ms |
| Codec/VLM 15 fps, 25 outputs, I+24P | CPU | 600.74 ms | 195.86 ms | 120.76 ms |
| Codec/VLM 15 fps, 25 outputs, I+24P | CUDA | 331.73 ms | 133.29 ms | 37.66 ms |

The initial center-grid optimization was not lossless: against legacy dense,
its per-frame payload MSE accumulated from 0.0195 on the first P-frame to 0.1735
on the third P-frame, while cosine fell to 0.9825. It remains available only as
the explicit `feature_grid_center` speed/accuracy trade-off.

The default `feature_grid` path restores the old boundary interpolation. On the
same four frames, MSE versus dense is at most `5.4e-11`, cosine is approximately
1.0, and maximum absolute payload error is `2.5e-4`. This is numerically
equivalent for inference purposes; it does not add a meaningful error beyond
the codec warp already present in the dense implementation.

## Optimized end-to-end edge results

| Input operating point | Device | Full selected-frame CNN | Decoder MV | Speedup | Latency change |
| --- | --- | ---: | ---: | ---: | ---: |
| Codec 2 fps, VLM 2 fps, 4 outputs, I+3P | CPU | 103.50 ms | 41.33 ms | 2.50x | -60.06% |
| Codec 2 fps, VLM 2 fps, 4 outputs, I+3P | CUDA | 29.37 ms | 15.74 ms | 1.87x | -46.41% |
| Codec 15 fps, 25 outputs, I+24P | CPU | 711.89 ms | 195.86 ms | 3.63x | -72.49% |
| Codec 15 fps, 25 outputs, I+24P | CUDA | 246.43 ms | 133.29 ms | 1.85x | -45.91% |

Sparse and real-B-frame numbers must be rerun for the equivalent path before
being used as headline results. The earlier center-grid measurements showed
that selected B-frame full-CNN fallback remains a separate limitation.

## Quality limitation

For the aligned BabyCrawling clip and prompt `What is the main action?`:

- Full frame encoding: `baby crawling`
- Decoder-MV warp only: `sleeping`

Payload similarity also decays under long recursive warp: cosine similarity to
full CNN is about 0.55 for four aligned frames and 0.38 over 25 I/P frames.
Decoder MVs describe prediction geometry, but they do not contain decoded
residual information. A confidence refresh policy or learned residual
correction remains necessary before treating this as a quality-preserving
acceleration method.

## Decision

The equivalent compute implementation is suitable for an experimental feature
branch: it removes much of the CPU/GPU overhead without custom CUDA or Triton
and without adding the center-grid approximation error. Codec warp versus full
CNN remains inherently lossy, so answer quality and B-frame handling still
prevent enabling acceleration by default.

## Reproduction

```powershell
python scripts\benchmark_codec_mv_edge.py `
  --video video_2fps_ip.mp4 `
  --edge_checkpoint checkpoints\...\edge_weights.pth `
  --sample_fps 2 `
  --max_frames 4 `
  --device cuda `
  --flow_impl feature_grid
```

Use `--flow_impl dense` for legacy rasterization or
`--flow_impl feature_grid_center` for the maximum-speed approximate path.

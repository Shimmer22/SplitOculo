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

## Quality evaluation

The earlier headline numbers came from one unusually difficult, locally
transcoded BabyCrawling clip. That clip remains useful as a stress test, but it
is not the default regression sample.

The replacement single-video sample is selected reproducibly from the sorted
75-video UCF101 test manifest with Python `random.Random(20260729).choice(...)`:

`BasketballDunk/v_BasketballDunk_g14_c06.avi`

The standard single-video protocol uses the first 16 selected frames,
`sample_fps=30`, recursive references, and no periodic CNN refresh. Cosine is
measured against the full-CNN payload on exactly the same decoded frames.

| Method | First P cosine | Mean P cosine | Last P cosine |
| --- | ---: | ---: | ---: |
| Decoder-MV warp only | 0.9772 | 0.8256 | 0.7434 |
| LSFA, 200 videos / 20 epochs | 0.9773 | 0.8472 | 0.7823 |
| LSFA, 300 videos / 35 epochs | 0.9771 | 0.8475 | 0.7825 |

This fixed random sample is the smoke/regression reporting point. It must not
replace the full held-out statistic in quality claims. Across all 75 held-out
videos (1,200 selected frames), the strict no-refresh P-frame results are:

| Method | Per-video first P mean | Global P mean | Per-video last P mean |
| --- | ---: | ---: | ---: |
| Decoder-MV warp only | 0.9469 | 0.8335 | 0.8036 |
| LSFA, 200 videos / 20 epochs | 0.9512 | 0.8645 | 0.8442 |
| LSFA, 300 videos / 35 epochs | 0.9516 | 0.8664 | 0.8466 |

The expanded training run improves every held-out action class, but the global
gain over the 20-epoch model is only 0.0019 P-frame cosine. Training quantity
is therefore no longer the main limitation. Decoder MVs describe prediction
geometry but do not contain native codec residuals, while the portable PyAV
implementation uses a decoded-RGB residual proxy. Long sampled-frame chains
and non-rigid motion remain difficult. `--codec_mv_min_coverage` and
`--codec_max_p_chain` provide conservative full-CNN refresh guards.

## Decision

The equivalent compute implementation is suitable for an experimental feature
branch: it removes much of the CPU/GPU overhead without custom CUDA or Triton
and without adding the center-grid approximation error. Codec warp versus full
CNN remains inherently lossy, so answer quality and B-frame handling still
prevent enabling acceleration by default.

## Reproduction

```powershell
E:\anaconda\envs\cnn_vit\python.exe scripts\evaluate_codec_manifest.py `
  --video_manifest outputs\codec_memory_ucf101\ucf101_test_75.manifest.txt `
  --edge_checkpoint checkpoints\cc3m10k_multilevel_layer4\split_gan_best\edge_weights.pth `
  --memory_checkpoint `
    outputs\codec_memory_ucf101\lsfa_ucf101_200_e20_payloadcos.pth `
    outputs\codec_memory_ucf101\lsfa_ucf101_300_e35_payloadcos.pth `
  --memory_arch lsfa `
  --reference_mode recursive `
  --sample_fps 30 `
  --max_frames 16 `
  --max_p_chain 0 `
  --device cuda `
  --output outputs\codec_memory_ucf101\evaluation_test75_e20_vs_e35_norefresh.json
```

Add `--random_sample_seed 20260729` to run only the fixed BasketballDunk
smoke/regression sample and write it to a separate output JSON. Omit the option,
as above, for the authoritative 75-video aggregate.

Set `--max_p_chain 4` for the guarded deployment candidate. The strict
no-refresh protocol above is retained because it isolates memory quality from
periodic full-CNN recomputation.

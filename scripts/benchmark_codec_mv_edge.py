"""Benchmark full CNN, decoder-MV warp, and optional MMNet memory paths."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from core.codec_video_reader import read_video_records_with_mvs
from models.codec_accelerator import DecoderMotionVectorAccelerator
from scripts.edge_client import EdgeEncoder


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--edge_checkpoint", required=True)
    parser.add_argument("--sample_fps", type=float, default=None)
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--warmup_rounds", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--flow_impl", choices=("feature_grid", "feature_grid_center", "dense"),
        default="feature_grid",
        help="Decoder-MV flow construction path; dense is the legacy comparison path.",
    )
    parser.add_argument(
        "--memory_checkpoint",
        default=None,
        help="Optional MMNet-style memory checkpoint for the decoder-MV path.",
    )
    parser.add_argument(
        "--memory_arch",
        choices=("mmnet", "lsfa"),
        default="mmnet",
        help="Memory architecture when the checkpoint has no embedded metadata.",
    )
    parser.add_argument(
        "--reference_mode",
        choices=("recursive", "keyframe"),
        default="recursive",
        help="Use recursive predicted features or key-frame features with composed MVs.",
    )
    parser.add_argument(
        "--min_coverage",
        type=float,
        default=0.0,
        help="Full-CNN fallback threshold for past-reference MV coverage.",
    )
    parser.add_argument(
        "--max_p_chain",
        type=int,
        default=0,
        help="Periodic full-CNN refresh after this many causal P frames; 0 disables.",
    )
    parser.add_argument(
        "--memory_rgb_mode",
        choices=("exact", "fast"),
        default="exact",
        help="Exact training-time RGB resize or faster direct feature-grid resize.",
    )
    parser.add_argument(
        "--projection_batch_size",
        type=int,
        default=1,
        help="Microbatch projector/bottleneck after temporal feature propagation.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def run_full(edge, selected_images, device):
    synchronize(device)
    started = time.perf_counter()
    payloads = [edge.encode_pil(image)[0] for image in selected_images]
    synchronize(device)
    return time.perf_counter() - started, payloads


@torch.no_grad()
def run_decoder_mv(
    edge,
    records,
    device,
    flow_impl,
    memory_checkpoint=None,
    memory_arch="mmnet",
    reference_mode="recursive",
    min_coverage=0.0,
    max_p_chain=0,
    memory_rgb_mode="exact",
    projection_batch_size=1,
):
    accelerator = DecoderMotionVectorAccelerator(
        edge,
        flow_impl=flow_impl,
        memory_checkpoint=memory_checkpoint,
        memory_arch=memory_arch,
        reference_mode=reference_mode,
        min_coverage=min_coverage,
        max_p_chain=max_p_chain,
        memory_rgb_mode=memory_rgb_mode,
    )
    synchronize(device)
    started = time.perf_counter()
    payloads = []
    deferred_features = []
    frame_info = []
    for record in records:
        payload, _, info = accelerator.encode_record(
            record,
            defer_payload=projection_batch_size > 1,
        )
        frame_info.append(info)
        if record["selected"]:
            if projection_batch_size > 1:
                deferred_features.append(payload)
            else:
                payloads.append(payload)
    if projection_batch_size > 1:
        for start in range(0, len(deferred_features), projection_batch_size):
            feature_batch = torch.cat(
                deferred_features[start : start + projection_batch_size], dim=0
            )
            payload_batch, _ = accelerator.project_features(feature_batch)
            payloads.extend(
                payload_batch[index : index + 1]
                for index in range(payload_batch.shape[0])
            )
    synchronize(device)
    return time.perf_counter() - started, payloads, frame_info


def stats(values):
    return {
        "mean_ms": 1000 * statistics.mean(values),
        "median_ms": 1000 * statistics.median(values),
        "stdev_ms": 1000 * statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_ms": 1000 * min(values),
        "max_ms": 1000 * max(values),
    }


def payload_similarity(full_payloads, mv_payloads):
    if len(full_payloads) != len(mv_payloads) or not full_payloads:
        return None
    full = torch.cat([payload.detach().float().cpu() for payload in full_payloads], dim=0)
    mv = torch.cat([payload.detach().float().cpu() for payload in mv_payloads], dim=0)
    if full.shape != mv.shape:
        return {"shape_match": False, "full_shape": list(full.shape), "mv_shape": list(mv.shape)}
    return {
        "shape_match": True,
        "mse": float(F.mse_loss(mv, full)),
        "cosine_similarity": float(F.cosine_similarity(mv.flatten(), full.flatten(), dim=0)),
    }


def main():
    args = parse_args()
    if args.projection_batch_size <= 0:
        raise ValueError("--projection_batch_size must be positive")
    decode_started = time.perf_counter()
    records, native_fps, reader = read_video_records_with_mvs(
        args.video, max_frames=args.max_frames, sample_fps=args.sample_fps
    )
    decode_seconds = time.perf_counter() - decode_started
    selected_images = [record["image"] for record in records if record["selected"]]
    edge = EdgeEncoder(args.edge_checkpoint, device=args.device)

    for _ in range(args.warmup_rounds):
        run_full(edge, selected_images, args.device)
        run_decoder_mv(
            edge,
            records,
            args.device,
            args.flow_impl,
            args.memory_checkpoint,
            args.memory_arch,
            args.reference_mode,
            args.min_coverage,
            args.max_p_chain,
            args.memory_rgb_mode,
            args.projection_batch_size,
        )

    full_times = []
    mv_times = []
    final_info = None
    full_payloads = None
    mv_payloads = None
    for _ in range(args.rounds):
        elapsed, full_payloads = run_full(edge, selected_images, args.device)
        full_times.append(elapsed)
        elapsed, mv_payloads, final_info = run_decoder_mv(
            edge,
            records,
            args.device,
            args.flow_impl,
            args.memory_checkpoint,
            args.memory_arch,
            args.reference_mode,
            args.min_coverage,
            args.max_p_chain,
            args.memory_rgb_mode,
            args.projection_batch_size,
        )
        mv_times.append(elapsed)

    full_mean = statistics.mean(full_times)
    mv_mean = statistics.mean(mv_times)
    selected_info = [record for record in final_info if record["selected"]]
    result = {
        "video": str(Path(args.video).resolve()),
        "reader": reader,
        "native_fps": native_fps,
        "sample_fps": args.sample_fps,
        "selected_frames": len(selected_images),
        "source_frames_processed": len(records),
        "warmup_rounds": args.warmup_rounds,
        "rounds": args.rounds,
        "device": args.device,
        "flow_impl": args.flow_impl,
        "memory_checkpoint": args.memory_checkpoint,
        "memory_arch": args.memory_arch,
        "reference_mode": args.reference_mode,
        "min_coverage": args.min_coverage,
        "max_p_chain": args.max_p_chain,
        "memory_rgb_mode": args.memory_rgb_mode,
        "projection_batch_size": args.projection_batch_size,
        "decode_with_mv_seconds": decode_seconds,
        "full": stats(full_times),
        "decoder_mv": stats(mv_times),
        "speedup": full_mean / mv_mean,
        "latency_change_pct": 100 * (mv_mean / full_mean - 1),
        "processed_cnn_frames": sum(record["cnn_executed"] for record in final_info),
        "processed_warp_frames": sum(record["warp_executed"] for record in final_info),
        "sampled_cnn_frames": sum(record["cnn_executed"] for record in selected_info),
        "sampled_warp_frames": sum(record["warp_executed"] for record in selected_info),
        "selected_modes": [record["mode"] for record in selected_info],
        "payload_similarity_to_full": payload_similarity(full_payloads, mv_payloads),
        "notes": [
            "Video decode and model load are excluded from both timed paths.",
            "CUDA is synchronized before and after each timed path.",
            "The decoder-MV path advances intervening source reference frames.",
            "feature_grid analytically preserves the legacy two-stage bilinear sampling semantics.",
            "feature_grid_center is the faster approximate one-MV-per-cell implementation.",
            "dense is the legacy full-resolution Python rasterization comparison path.",
            "memory uses decoded-RGB minus MV-warped RGB as a portable residual proxy.",
        ],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

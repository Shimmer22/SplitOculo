"""Benchmark local-GPU edge backbone microbatch sizes."""

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
from scripts.edge_client import EdgeEncoder


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--edge_checkpoint", required=True)
    parser.add_argument("--sample_fps", type=float, default=30.0)
    parser.add_argument("--max_frames", type=int, default=16)
    parser.add_argument("--warmup_rounds", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@torch.inference_mode()
def _run(encoder, images, batch_size):
    if batch_size == 1:
        return [encoder.encode_pil(image)[0] for image in images]
    payloads = []
    for start in range(0, len(images), batch_size):
        payload, _ = encoder.encode_pil_batch(images[start : start + batch_size])
        payloads.extend(payload[index : index + 1] for index in range(payload.shape[0]))
    return payloads


def _measure(encoder, images, device, warmup_rounds, rounds, batch_size):
    for _ in range(warmup_rounds):
        _run(encoder, images, batch_size)
    values = []
    payloads = None
    for _ in range(rounds):
        _sync(device)
        started = time.perf_counter()
        payloads = _run(encoder, images, batch_size)
        _sync(device)
        values.append(time.perf_counter() - started)
    return payloads, {
        "mean_ms": 1000 * statistics.mean(values),
        "median_ms": 1000 * statistics.median(values),
        "stdev_ms": 1000 * statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_ms": 1000 * min(values),
        "max_ms": 1000 * max(values),
    }


def _similarity(reference, candidate):
    reference = torch.cat([item.detach().double().cpu() for item in reference])
    candidate = torch.cat([item.detach().double().cpu() for item in candidate])
    return {
        "cosine_similarity": float(
            F.cosine_similarity(candidate.flatten(), reference.flatten(), dim=0)
        ),
        "mse": float(F.mse_loss(candidate, reference)),
        "max_abs_error": float((candidate - reference).abs().max()),
    }


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    records, native_fps, reader = read_video_records_with_mvs(
        args.video,
        max_frames=args.max_frames,
        sample_fps=args.sample_fps,
    )
    images = [record["image"] for record in records if record["selected"]]

    reference = EdgeEncoder(args.edge_checkpoint, device=args.device)
    reference_payloads, reference_stats = _measure(
        reference, images, args.device, args.warmup_rounds, args.rounds, 1
    )
    candidate = EdgeEncoder(args.edge_checkpoint, device=args.device)
    candidate_payloads, candidate_stats = _measure(
        candidate,
        images,
        args.device,
        args.warmup_rounds,
        args.rounds,
        args.batch_size,
    )
    result = {
        "video": str(Path(args.video).resolve()),
        "reader": reader,
        "native_fps": native_fps,
        "frames": len(images),
        "reference": {
            "batch_size": 1,
            **reference_stats,
        },
        "candidate": {
            "batch_size": args.batch_size,
            **candidate_stats,
        },
        "speedup": reference_stats["mean_ms"] / candidate_stats["mean_ms"],
        "payload_similarity": _similarity(reference_payloads, candidate_payloads),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

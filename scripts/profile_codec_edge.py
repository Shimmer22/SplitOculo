"""Profile the local CUDA codec edge path on a short compressed video."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from core.codec_video_reader import read_video_records_with_mvs
from models.codec_accelerator import DecoderMotionVectorAccelerator
from scripts.edge_client import EdgeEncoder


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--edge_checkpoint", required=True)
    parser.add_argument("--memory_checkpoint", default=None)
    parser.add_argument("--memory_arch", choices=("mmnet", "lsfa"), default="lsfa")
    parser.add_argument("--max_frames", type=int, default=16)
    parser.add_argument("--sample_fps", type=float, default=30.0)
    parser.add_argument("--warmup_rounds", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def run(accelerator, records):
    accelerator.reset()
    for record in records:
        accelerator.encode_record(record)


def main():
    args = parse_args()
    records, _, _ = read_video_records_with_mvs(
        args.video, max_frames=args.max_frames, sample_fps=args.sample_fps
    )
    edge = EdgeEncoder(args.edge_checkpoint, device=args.device)
    accelerator = DecoderMotionVectorAccelerator(
        edge,
        flow_impl="feature_grid",
        memory_checkpoint=args.memory_checkpoint,
        memory_arch=args.memory_arch,
    )
    for _ in range(args.warmup_rounds):
        run(accelerator, records)
    torch.cuda.synchronize()

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        run(accelerator, records)
        torch.cuda.synchronize()

    averages = profiler.key_averages(group_by_input_shape=True)
    print("\nCUDA hotspots")
    print(averages.table(sort_by="self_cuda_time_total", row_limit=30))
    print("\nCPU hotspots")
    print(averages.table(sort_by="self_cpu_time_total", row_limit=30))


if __name__ == "__main__":
    main()

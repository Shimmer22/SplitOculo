"""Evaluate warp-only and one or more codec memories on a video manifest."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from core.codec_video_reader import read_video_records_with_mvs
from models.codec_accelerator import DecoderMotionVectorAccelerator
from scripts.benchmark_codec_chain import _frame_rows, _full_payloads, _summarize
from scripts.edge_client import EdgeEncoder


def _read_manifest(path: Path) -> list[str]:
    videos = []
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        video = Path(line)
        if not video.is_absolute():
            video = (path.parent / video).resolve()
        videos.append(str(video))
    if not videos:
        raise ValueError(f"No videos found in {path}")
    return videos


@torch.no_grad()
def _run_accelerator(accelerator, records, device):
    accelerator.reset()
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    payloads = []
    infos = []
    for record in records:
        payload, _, info = accelerator.encode_record(record)
        infos.append(info)
        if record["selected"]:
            payloads.append(payload)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    return payloads, infos, time.perf_counter() - started


def _class_name(video: str) -> str:
    return Path(video).parent.name


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_manifest", type=Path, required=True)
    parser.add_argument("--edge_checkpoint", required=True)
    parser.add_argument("--memory_checkpoint", nargs="+", required=True)
    parser.add_argument("--memory_arch", choices=("mmnet", "lsfa"), default="lsfa")
    parser.add_argument(
        "--reference_mode", choices=("recursive", "keyframe"), default="recursive"
    )
    parser.add_argument("--sample_fps", type=float, default=None)
    parser.add_argument("--max_frames", type=int, default=16)
    parser.add_argument(
        "--random_sample_seed",
        type=int,
        default=None,
        help="Reproducibly evaluate one video selected from the sorted manifest.",
    )
    parser.add_argument("--min_coverage", type=float, default=0.0)
    parser.add_argument("--max_p_chain", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    videos = _read_manifest(args.video_manifest)
    if args.random_sample_seed is not None:
        videos = [
            random.Random(args.random_sample_seed).choice(sorted(videos))
        ]
        print(
            f"Random sample seed {args.random_sample_seed}: {videos[0]}"
        )
    edge = EdgeEncoder(args.edge_checkpoint, device=args.device)
    warp = DecoderMotionVectorAccelerator(
        edge, flow_impl="feature_grid", reference_mode=args.reference_mode
    )
    memories = {
        Path(checkpoint).stem: DecoderMotionVectorAccelerator(
            edge,
            flow_impl="feature_grid",
            memory_checkpoint=checkpoint,
            memory_arch=args.memory_arch,
            reference_mode=args.reference_mode,
            min_coverage=args.min_coverage,
            max_p_chain=args.max_p_chain,
        )
        for checkpoint in args.memory_checkpoint
    }

    all_rows = defaultdict(list)
    class_rows = defaultdict(lambda: defaultdict(list))
    per_video = []
    timings = defaultdict(float)
    for index, video in enumerate(videos, 1):
        records, native_fps, reader = read_video_records_with_mvs(
            video, max_frames=args.max_frames, sample_fps=args.sample_fps
        )
        selected = [record for record in records if record["selected"]]
        full_payloads, elapsed = _full_payloads(edge, selected, args.device)
        timings["full"] += elapsed

        methods = {"warp": warp, **memories}
        video_result = {
            "video": str(Path(video).resolve()),
            "class": _class_name(video),
            "reader": reader,
            "native_fps": native_fps,
            "selected_frames": len(selected),
        }
        for name, accelerator in methods.items():
            payloads, infos, elapsed = _run_accelerator(
                accelerator, records, args.device
            )
            rows = _frame_rows(full_payloads, payloads, infos)
            timings[name] += elapsed
            all_rows[name].extend(rows)
            class_rows[_class_name(video)][name].extend(rows)
            video_result[name] = _summarize(rows)
        per_video.append(video_result)
        print(f"[{index}/{len(videos)}] {Path(video).name}")

    result = {
        "manifest": str(args.video_manifest.resolve()),
        "videos": len(videos),
        "random_sample_seed": args.random_sample_seed,
        "reference_mode": args.reference_mode,
        "max_p_chain": args.max_p_chain,
        "timings_seconds": dict(timings),
        "summary": {name: _summarize(rows) for name, rows in all_rows.items()},
        "classes": {
            class_name: {
                name: _summarize(rows) for name, rows in methods.items()
            }
            for class_name, methods in class_rows.items()
        },
        "per_video": per_video,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

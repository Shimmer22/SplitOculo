"""Measure recursive codec feature quality frame by frame.

The benchmark compares full CNN payloads with decoder-MV warp-only and an
optional MMNet memory checkpoint. It is intentionally separate from the
latency benchmark because the useful chain signal is the per-frame drift.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from core.codec_video_reader import read_video_records_with_mvs
from models.codec_accelerator import DecoderMotionVectorAccelerator
from scripts.edge_client import EdgeEncoder


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def _full_payloads(edge, records, device):
    _sync(device)
    started = time.perf_counter()
    payloads = [edge.encode_pil(record["image"])[0] for record in records if record["selected"]]
    _sync(device)
    return payloads, time.perf_counter() - started


@torch.no_grad()
def _decoder_payloads(
    edge,
    records,
    device,
    memory_checkpoint,
    memory_arch,
    reference_mode,
    min_coverage,
    max_p_chain,
):
    accelerator = DecoderMotionVectorAccelerator(
        edge,
        flow_impl="feature_grid",
        memory_checkpoint=memory_checkpoint,
        memory_arch=memory_arch,
        reference_mode=reference_mode,
        min_coverage=min_coverage,
        max_p_chain=max_p_chain,
    )
    _sync(device)
    started = time.perf_counter()
    payloads = []
    infos = []
    for record in records:
        payload, _, info = accelerator.encode_record(record)
        infos.append(info)
        if record["selected"]:
            payloads.append(payload)
    _sync(device)
    return payloads, infos, time.perf_counter() - started


def _frame_rows(full_payloads, decoder_payloads, infos):
    rows = []
    for full, decoder, info in zip(full_payloads, decoder_payloads, [item for item in infos if item["selected"]]):
        full = full.detach().float()
        decoder = decoder.detach().float()
        rows.append(
            {
                "source_index": info["source_index"],
                "codec_frame_type": info["codec_frame_type"],
                "mode": info["mode"],
                "cnn_executed": info["cnn_executed"],
                "memory_executed": info.get("memory_executed", False),
                "past_mv_coverage": info.get("past_mv_coverage"),
                "p_chain_length": info.get("p_chain_length"),
                "cosine_similarity": float(
                    F.cosine_similarity(decoder.flatten(), full.flatten(), dim=0)
                ),
                "mse": float(F.mse_loss(decoder, full)),
            }
        )
    return rows


def _summarize(rows):
    if not rows:
        return {"frames": 0}
    cosine = [row["cosine_similarity"] for row in rows]
    mse = [row["mse"] for row in rows]
    summary = {
        "frames": len(rows),
        "first_cosine": cosine[0],
        "last_cosine": cosine[-1],
        "min_cosine": min(cosine),
        "mean_cosine": sum(cosine) / len(cosine),
        "first_mse": mse[0],
        "last_mse": mse[-1],
        "max_mse": max(mse),
        "cnn_frames": sum(row["cnn_executed"] for row in rows),
        "memory_frames": sum(row["memory_executed"] for row in rows),
    }
    groups = {
        "p_frames": [row for row in rows if row["codec_frame_type"] == "P"],
        "approximated_frames": [
            row for row in rows if row["mode"] in {"P_MV", "P_MMNET", "P_LSFA"}
        ],
        "full_cnn_frames": [row for row in rows if row["cnn_executed"]],
    }
    for name, group in groups.items():
        values = [row["cosine_similarity"] for row in group]
        summary[name] = {
            "frames": len(group),
            "mean_cosine": sum(values) / len(values) if values else None,
            "first_cosine": values[0] if values else None,
            "last_cosine": values[-1] if values else None,
        }
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--edge_checkpoint", required=True)
    parser.add_argument("--memory_checkpoint", default=None)
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
        help="Propagate predicted features recursively or warp the key feature with composed MVs.",
    )
    parser.add_argument("--sample_fps", type=float, default=None)
    parser.add_argument("--max_frames", type=int, default=16)
    parser.add_argument("--min_coverage", type=float, default=0.0)
    parser.add_argument("--max_p_chain", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    records, native_fps, reader = read_video_records_with_mvs(
        args.video,
        max_frames=args.max_frames,
        sample_fps=args.sample_fps,
    )
    selected_records = [record for record in records if record["selected"]]
    edge = EdgeEncoder(args.edge_checkpoint, device=args.device)
    full_payloads, full_seconds = _full_payloads(edge, selected_records, args.device)

    warp_payloads, warp_infos, warp_seconds = _decoder_payloads(
        edge,
        records,
        args.device,
        None,
        args.memory_arch,
        args.reference_mode,
        0.0,
        0,
    )
    warp_rows = _frame_rows(full_payloads, warp_payloads, warp_infos)

    result = {
        "video": str(Path(args.video).resolve()),
        "reader": reader,
        "native_fps": native_fps,
        "selected_frames": len(selected_records),
        "source_frames_processed": len(records),
        "reference_mode": args.reference_mode,
        "full_seconds": full_seconds,
        "warp_seconds": warp_seconds,
        "warp_summary": _summarize(warp_rows),
        "warp_frames": warp_rows,
    }

    if args.memory_checkpoint:
        memory_payloads, memory_infos, memory_seconds = _decoder_payloads(
            edge,
            records,
            args.device,
            args.memory_checkpoint,
            args.memory_arch,
            args.reference_mode,
            args.min_coverage,
            args.max_p_chain,
        )
        memory_rows = _frame_rows(full_payloads, memory_payloads, memory_infos)
        result.update(
            {
                "memory_checkpoint": args.memory_checkpoint,
                "memory_arch": args.memory_arch,
                "memory_seconds": memory_seconds,
                "memory_summary": _summarize(memory_rows),
                "memory_frames": memory_rows,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

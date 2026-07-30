"""Paired action-classification evaluation for original and temporal SplitOculo."""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from models.temporal_pair import load_temporal_pair_fusion
from scripts.cloud_server import CloudInferenceEngine
from scripts.edge_client import EdgeEncoder
from scripts.infer_qwen_video import read_video_frames
from scripts.train_temporal_pair import square_center_crop


def normalize_label(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def parse_prediction(response, labels):
    normalized = normalize_label(response)
    for label in sorted(labels, key=len, reverse=True):
        if normalize_label(label) in normalized:
            return label
    return None


@torch.inference_mode()
def encode_original(edge, frames):
    payload, _ = edge.encode_pil_batch(frames)
    return payload


@torch.inference_mode()
def encode_temporal(edge, fusion, frames):
    paired = []
    for index in range(0, len(frames), 2):
        paired.extend(
            (
                frames[index],
                frames[index + 1] if index + 1 < len(frames) else frames[index],
            )
        )
    image_tensor = torch.stack(
        [edge.transform(frame.convert("RGB")) for frame in paired], dim=0
    )
    backbone = edge.student(edge.input_to_device(image_tensor))[-1]
    fused = fusion(backbone[0::2], backbone[1::2])
    tokens = edge.projector(fused)
    return edge.bottleneck.encode(tokens) if edge.bottleneck is not None else tokens


@torch.inference_mode()
def encode_mean_pair(edge, frames):
    paired = []
    for index in range(0, len(frames), 2):
        paired.extend(
            (
                frames[index],
                frames[index + 1] if index + 1 < len(frames) else frames[index],
            )
        )
    image_tensor = torch.stack(
        [edge.transform(frame.convert("RGB")) for frame in paired], dim=0
    )
    backbone = edge.student(edge.input_to_device(image_tensor))[-1]
    fused = (backbone[0::2] + backbone[1::2]) * 0.5
    tokens = edge.projector(fused)
    return edge.bottleneck.encode(tokens) if edge.bottleneck is not None else tokens


def summarize(records, modes):
    summary = {"videos": len(records), "modes": {}}
    for mode in modes:
        valid = [item for item in records if mode in item["predictions"]]
        correct = sum(
            item["predictions"][mode]["prediction"] == item["label"]
            for item in valid
        )
        summary["modes"][mode] = {
            "count": len(valid),
            "correct": correct,
            "accuracy": correct / len(valid) if valid else None,
        }
    if "original" in modes and "temporal" in modes:
        regressions = []
        improvements = []
        for item in records:
            original_ok = (
                item["predictions"].get("original", {}).get("prediction")
                == item["label"]
            )
            temporal_ok = (
                item["predictions"].get("temporal", {}).get("prediction")
                == item["label"]
            )
            if original_ok and not temporal_ok:
                regressions.append(item["video"])
            elif temporal_ok and not original_ok:
                improvements.append(item["video"])
        summary["paired"] = {
            "baseline_correct_temporal_wrong": len(regressions),
            "baseline_wrong_temporal_correct": len(improvements),
            "regression_videos": regressions,
            "improvement_videos": improvements,
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--edge_checkpoint", required=True)
    parser.add_argument("--cloud_checkpoint", required=True)
    parser.add_argument("--temporal_checkpoint", required=True)
    parser.add_argument("--qwen_path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--modes", default="original,temporal")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--max_videos", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    labels = sorted(path.name for path in test_dir.iterdir() if path.is_dir())
    videos = sorted(
        path
        for path in test_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv", ".webm"}
    )
    if args.max_videos:
        videos = videos[: args.max_videos]
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    unknown = set(modes) - {"original", "mean", "temporal", "native"}
    if unknown:
        parser.error(f"Unknown modes: {sorted(unknown)}")

    edge = EdgeEncoder(args.edge_checkpoint, device=args.device)
    fusion, temporal_metadata = load_temporal_pair_fusion(
        args.temporal_checkpoint, device=args.device
    )
    fusion.eval()
    cloud = CloudInferenceEngine(
        args.cloud_checkpoint,
        device=args.device,
        split_layer=int(temporal_metadata.get("split_layer", 4)),
    )
    cloud.qwen_path = args.qwen_path
    cloud.offline_mode = args.offline

    prompt = (
        "Classify the primary action in this video. Choose exactly one label "
        f"from: {', '.join(labels)}. Answer with the label only."
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for video_index, video_path in enumerate(videos, 1):
        frames, native_fps, reader = read_video_frames(
            video_path, max_frames=args.max_frames, sample_fps=None
        )
        frames = [square_center_crop(frame, edge.image_size) for frame in frames]
        record = {
            "video": str(video_path),
            "label": video_path.parent.name,
            "frames": len(frames),
            "native_fps": native_fps,
            "reader": reader,
            "predictions": {},
        }
        if "original" in modes:
            payload = encode_original(edge, frames)
            response, _ = cloud.infer_video_from_frame_features_with_timing(
                payload, prompt=prompt, max_new_tokens=args.max_new_tokens
            )
            record["predictions"]["original"] = {
                "response": response,
                "prediction": parse_prediction(response, labels),
                "temporal_grids": len(frames),
            }
        if "temporal" in modes:
            payload = encode_temporal(edge, fusion, frames)
            response, _ = cloud.infer_video_from_frame_features_with_timing(
                payload, prompt=prompt, max_new_tokens=args.max_new_tokens
            )
            record["predictions"]["temporal"] = {
                "response": response,
                "prediction": parse_prediction(response, labels),
                "temporal_grids": (len(frames) + 1) // 2,
            }
        if "mean" in modes:
            payload = encode_mean_pair(edge, frames)
            response, _ = cloud.infer_video_from_frame_features_with_timing(
                payload, prompt=prompt, max_new_tokens=args.max_new_tokens
            )
            record["predictions"]["mean"] = {
                "response": response,
                "prediction": parse_prediction(response, labels),
                "temporal_grids": (len(frames) + 1) // 2,
            }
        if "native" in modes:
            response, _ = cloud.infer_qwen_frames_with_timing(
                frames,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                video_pixel_budget=edge.image_size * edge.image_size,
                video_fps=2.0,
            )
            record["predictions"]["native"] = {
                "response": response,
                "prediction": parse_prediction(response, labels),
                "temporal_grids": (len(frames) + 1) // 2,
            }
        records.append(record)
        payload = {
            "config": vars(args),
            "labels": labels,
            "summary": summarize(records, modes),
            "records": records,
        }
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        current = payload["summary"]["modes"]
        print(f"[{video_index}/{len(videos)}] {video_path.name}: {current}")

    print(json.dumps(summarize(records, modes), indent=2, ensure_ascii=False))
    print(f"Saved paired semantic evaluation to {output_path}")


if __name__ == "__main__":
    main()

"""First-pass SplitOculo video inference.

The video path intentionally reuses the trained single-image SplitOculo model:
each sampled frame is encoded independently on the edge, reconstructed on the
cloud, then the reconstructed frame tokens are concatenated in temporal order
and passed to Qwen as video tokens.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from models.multilevel import parse_payload_levels
from scripts.edge_client import EdgeEncoder
from scripts.cloud_server import CloudInferenceEngine
from scripts.infer_qwen_video import read_video_frames


def main():
    parser = argparse.ArgumentParser(description="SplitOculo per-frame video inference")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--edge_checkpoint", type=str, required=True)
    parser.add_argument("--cloud_checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Describe this video briefly.")
    parser.add_argument("--max_frames", type=int, default=4)
    parser.add_argument("--sample_fps", type=float, default=None)
    parser.add_argument("--level", type=str, default=None,
                        help="Optional multi-level payload, e.g. 49x64, 49x128, 196x64, 196x128")
    parser.add_argument("--qwen_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--split_layer", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--save_payload", type=str, default=None)
    parser.add_argument("--metadata", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    payload_level = parse_payload_levels(args.level)[0] if args.level else None

    total_start = time.perf_counter()
    decode_start = time.perf_counter()
    frames, native_fps, reader = read_video_frames(
        args.video,
        max_frames=args.max_frames,
        sample_fps=args.sample_fps,
    )
    decode_seconds = time.perf_counter() - decode_start
    print(f"Decoded {len(frames)} frames with {reader} (native_fps={native_fps:.3f})")

    edge_load_start = time.perf_counter()
    edge = EdgeEncoder(args.edge_checkpoint, device=args.device)
    edge_load_seconds = time.perf_counter() - edge_load_start
    frame_features = []
    edge_encode_seconds = 0.0
    for idx, frame in enumerate(frames):
        encode_start = time.perf_counter()
        if payload_level:
            features, is_compressed = edge.encode_pil_level(frame, payload_level)
        else:
            features, is_compressed = edge.encode_pil(frame)
        edge_encode_seconds += time.perf_counter() - encode_start
        frame_features.append(features.squeeze(0).detach())
        print(f"Encoded frame {idx + 1}/{len(frames)}: {tuple(features.shape)}, compressed={is_compressed}")

    compressed_frame_features = torch.stack(frame_features, dim=0)
    payload_tensor_bytes = (
        compressed_frame_features.numel() * compressed_frame_features.element_size()
    )
    payload_int8_bytes = compressed_frame_features.numel()

    if args.save_payload:
        payload_path = Path(args.save_payload)
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "compressed_frame_features": compressed_frame_features.cpu(),
                "video": str(Path(args.video)),
                "frames_sampled": len(frames),
                "native_fps": native_fps,
                "sample_fps": args.sample_fps,
                "edge_checkpoint": args.edge_checkpoint,
                "cloud_checkpoint": args.cloud_checkpoint,
            },
            payload_path,
        )
        print(f"Saved edge payload: {payload_path}")

    cloud_load_start = time.perf_counter()
    cloud = CloudInferenceEngine(
        args.cloud_checkpoint,
        device=args.device,
        split_layer=args.split_layer,
    )
    cloud_load_seconds = time.perf_counter() - cloud_load_start
    cloud.qwen_path = args.qwen_path
    cloud.offline_mode = args.offline

    cloud_infer_start = time.perf_counter()
    response, cloud_metrics = cloud.infer_video_from_frame_features_with_timing(
        compressed_frame_features.to(args.device),
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        multilevel_payload=payload_level is not None,
    )
    cloud_infer_seconds = time.perf_counter() - cloud_infer_start
    total_seconds = time.perf_counter() - total_start
    print("\nResponse:")
    print(response)

    target_tokens = cloud.target_tokens
    target_side = int(target_tokens ** 0.5)
    metadata = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "video": str(Path(args.video)),
        "prompt": args.prompt,
        "response": response,
        "frames_sampled": len(frames),
        "native_fps": native_fps,
        "reader": reader,
        "edge_feature_shape": list(compressed_frame_features.shape),
        "video_grid_thw": [len(frames), target_side, target_side],
        "edge_checkpoint": args.edge_checkpoint,
        "cloud_checkpoint": args.cloud_checkpoint,
        "qwen_path": args.qwen_path,
        "split_layer": args.split_layer,
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
        "payload_level": args.level,
        "payload_tensor_bytes": int(payload_tensor_bytes),
        "payload_int8_bytes": int(payload_int8_bytes),
        "metrics": {
            "decode_seconds": decode_seconds,
            "edge_load_seconds": edge_load_seconds,
            "edge_encode_seconds": edge_encode_seconds,
            "cloud_load_seconds": cloud_load_seconds,
            "cloud_infer_seconds": cloud_infer_seconds,
            "total_seconds": total_seconds,
            **cloud_metrics,
        },
    }

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "run.json"
        jsonl_path = output_dir / "runs.jsonl"
        review_path = output_dir / "human_review.md"
        json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        review_lines = [
            "# SplitOculo Video Human Review",
            "",
            f"- Video: `{args.video}`",
            f"- Frames: `{len(frames)}`",
            f"- Device: `{args.device}`",
            f"- Payload tensor bytes: `{payload_tensor_bytes}`",
            f"- Payload int8 bytes estimate: `{payload_int8_bytes}`",
            f"- First token latency: `{cloud_metrics.get('first_token_seconds')}`",
            f"- Average TPS: `{cloud_metrics.get('average_tps')}`",
            "",
            "## Response",
            "",
            "```text",
            response,
            "```",
            "",
            "## Human Verification",
            "",
            "| Check Item | Verdict | Notes |",
            "|---|---|---|",
            "| Scene type correct |  |  |",
            "| Indoor/outdoor correct |  |  |",
            "| Key objects present |  |  |",
            "| Key objects missing |  |  |",
            "| Hallucinated objects/events |  |  |",
            "| Wearer activity correct |  |  |",
            "| Safety/navigation details correct |  |  |",
            "| Overall usable for this scene |  |  |",
            "",
        ]
        review_path.write_text("\n".join(review_lines), encoding="utf-8")
        print(f"Saved JSON: {json_path}")
        print(f"Saved JSONL: {jsonl_path}")
        print(f"Saved review: {review_path}")

    if args.metadata:
        metadata_path = Path(args.metadata)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()

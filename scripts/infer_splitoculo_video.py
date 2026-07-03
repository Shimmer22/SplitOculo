"""First-pass SplitOculo video inference.

The video path intentionally reuses the trained single-image SplitOculo model:
each sampled frame is encoded independently on the edge, reconstructed on the
cloud, then the reconstructed frame tokens are concatenated in temporal order
and passed to Qwen as video tokens.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

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
    parser.add_argument("--qwen_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--split_layer", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--save_payload", type=str, default=None)
    parser.add_argument("--metadata", type=str, default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    frames, native_fps, reader = read_video_frames(
        args.video,
        max_frames=args.max_frames,
        sample_fps=args.sample_fps,
    )
    print(f"Decoded {len(frames)} frames with {reader} (native_fps={native_fps:.3f})")

    edge = EdgeEncoder(args.edge_checkpoint, device=args.device)
    frame_features = []
    for idx, frame in enumerate(frames):
        features, is_compressed = edge.encode_pil(frame)
        frame_features.append(features.squeeze(0).detach())
        print(f"Encoded frame {idx + 1}/{len(frames)}: {tuple(features.shape)}, compressed={is_compressed}")

    compressed_frame_features = torch.stack(frame_features, dim=0)

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

    cloud = CloudInferenceEngine(
        args.cloud_checkpoint,
        device=args.device,
        split_layer=args.split_layer,
    )
    cloud.qwen_path = args.qwen_path
    cloud.offline_mode = args.offline

    response = cloud.infer_video_from_frame_features(
        compressed_frame_features.to(args.device),
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )
    print("\nResponse:")
    print(response)

    if args.metadata:
        target_tokens = cloud.target_tokens
        target_side = int(target_tokens ** 0.5)
        metadata = {
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
        }
        metadata_path = Path(args.metadata)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()

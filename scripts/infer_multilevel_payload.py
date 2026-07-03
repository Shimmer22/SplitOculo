"""Run SplitOculo inference with a selected multi-level payload.

This script is intended for checkpoints trained with:

    --multilevel_payload
    --transmission_tokens 196
    --bottleneck_dim 128
    --payload_levels 49x64,49x128,196x64,196x128

It also works with split edge/cloud checkpoints when the args are preserved.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from models.multilevel import parse_payload_levels
from scripts.cloud_server import CloudInferenceEngine
from scripts.edge_client import EdgeEncoder


def main():
    parser = argparse.ArgumentParser(description="SplitOculo multi-level payload inference")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--edge_checkpoint", type=str, required=True)
    parser.add_argument("--cloud_checkpoint", type=str, required=True)
    parser.add_argument("--level", type=str, default="49x64",
                        help="payload level, one of 49x64, 49x128, 196x64, 196x128")
    parser.add_argument("--prompt", type=str, default="Describe this image.")
    parser.add_argument("--qwen_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--split_layer", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--save_payload", type=str, default=None)
    parser.add_argument("--metadata", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    level = parse_payload_levels(args.level)[0]

    edge = EdgeEncoder(args.edge_checkpoint, device=args.device)
    payload, stats = edge.encode_to_payload_level(args.image, level)
    payload["prompt"] = args.prompt

    print(f"Payload level: {stats['payload_level']}")
    print(f"Feature shape: {stats['feature_shape']}")
    print(f"Payload bytes (base64 chars): {stats['payload_bytes']}")
    print(f"Edge encode: {stats['encode_time_ms']:.2f} ms")

    if args.save_payload:
        payload_path = Path(args.save_payload)
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        with open(payload_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved payload: {payload_path}")

    cloud = CloudInferenceEngine(
        args.cloud_checkpoint,
        device=args.device,
        split_layer=args.split_layer,
    )
    cloud.qwen_path = args.qwen_path
    cloud.offline_mode = args.offline

    features = cloud.decode_features(
        payload["features"],
        payload["scale"],
        payload["zero_point"],
        payload_tokens=payload["payload_tokens"],
        payload_dim=payload["payload_dim"],
    )
    response = cloud.infer_payload(
        features,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )

    print("\nResponse:")
    print(response)

    if args.metadata:
        metadata = {
            "image": str(Path(args.image)),
            "level": stats["payload_level"],
            "feature_shape": stats["feature_shape"],
            "payload_base64_chars": stats["payload_bytes"],
            "prompt": args.prompt,
            "response": response,
            "edge_checkpoint": args.edge_checkpoint,
            "cloud_checkpoint": args.cloud_checkpoint,
            "qwen_path": args.qwen_path,
            "split_layer": args.split_layer,
        }
        metadata_path = Path(args.metadata)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()

"""SplitOculo video inference with optional codec-guided edge acceleration.

By default every sampled frame is encoded independently. With ``--codec_acc``,
each GOP I-frame runs the edge CNN and P-frames recursively warp the preceding
predicted CNN feature before the normal projector/bottleneck/cloud path.  A
trained ``--codec_memory_checkpoint`` enables an MMNet-style residual update
after the motion warp.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from models.multilevel import parse_payload_levels
from models.codec_accelerator import CodecWarpAccelerator, DecoderMotionVectorAccelerator
from models.temporal_pair import load_temporal_pair_fusion
from core.codec_video_reader import read_video_records_with_mvs
from scripts.edge_client import EdgeEncoder
from scripts.cloud_server import CloudInferenceEngine
from scripts.infer_qwen_video import read_video_frames


def main():
    parser = argparse.ArgumentParser(description="SplitOculo per-frame video inference")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--edge_checkpoint", type=str, required=True)
    parser.add_argument("--cloud_checkpoint", type=str, required=True)
    parser.add_argument(
        "--temporal_pair_checkpoint",
        type=str,
        default=None,
        help="Fuse adjacent frames into Qwen-native temporal grid units",
    )
    parser.add_argument("--prompt", type=str, default="Describe this video briefly.")
    parser.add_argument("--max_frames", type=int, default=4)
    parser.add_argument("--sample_fps", type=float, default=None)
    parser.add_argument("--level", type=str, default=None,
                        help="Optional multi-level payload, e.g. 49x64, 49x128, 196x64, 196x128")
    parser.add_argument("--codec_acc", action="store_true",
                        help="Use codec-guided I/P CNN feature reuse on the edge")
    parser.add_argument("--codec_mv_backend", choices=["decoder", "farneback"], default="decoder",
                        help="Motion source for --codec_acc (default: real decoder MVs)")
    parser.add_argument(
        "--codec_flow_impl",
        choices=["feature_grid", "feature_grid_center", "dense"],
        default="feature_grid",
        help="Decoder-MV flow path: equivalent optimized, approximate fast, or legacy dense",
    )
    parser.add_argument("--codec_gop_frames", type=int, default=4,
                        help="Synthetic sampled-frame GOP for the Farneback backend only")
    parser.add_argument(
        "--codec_memory_checkpoint",
        type=str,
        default=None,
        help="MMNet-style feature memory checkpoint trained on real codec sequences",
    )
    parser.add_argument(
        "--codec_memory_arch",
        choices=["mmnet", "lsfa"],
        default="mmnet",
        help="Memory architecture when the checkpoint has no embedded architecture metadata",
    )
    parser.add_argument(
        "--codec_reference_mode",
        choices=["recursive", "keyframe"],
        default="recursive",
        help="Use recursive predicted features or key-frame features with composed MVs",
    )
    parser.add_argument(
        "--codec_mv_min_coverage",
        type=float,
        default=0.0,
        help="Fallback to full CNN when past-reference MV coverage is below this value",
    )
    parser.add_argument(
        "--codec_max_p_chain",
        type=int,
        default=0,
        help="Optional periodic full-CNN refresh after this many causal P-frames (0 disables)",
    )
    parser.add_argument(
        "--codec_memory_rgb_mode",
        choices=["exact", "fast"],
        default="exact",
        help="Exact training-time RGB resize or faster direct feature-grid resize",
    )
    parser.add_argument("--qwen_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--split_layer", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--save_payload", type=str, default=None)
    parser.add_argument("--metadata", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--edge_batch_size",
        type=int,
        default=1,
        help="Microbatch size for non-codec full-CNN video encoding",
    )
    parser.add_argument(
        "--codec_projection_batch_size",
        type=int,
        default=1,
        help="Microbatch projector/bottleneck after decoder-MV temporal propagation",
    )
    args = parser.parse_args()

    if args.codec_memory_checkpoint and not (
        args.codec_acc and args.codec_mv_backend == "decoder"
    ):
        parser.error("--codec_memory_checkpoint requires --codec_acc with --codec_mv_backend decoder")
    if args.edge_batch_size <= 0:
        parser.error("--edge_batch_size must be positive")
    if args.codec_projection_batch_size <= 0:
        parser.error("--codec_projection_batch_size must be positive")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    payload_level = parse_payload_levels(args.level)[0] if args.level else None
    if args.temporal_pair_checkpoint and args.codec_acc:
        parser.error("--temporal_pair_checkpoint does not yet support --codec_acc")
    if args.temporal_pair_checkpoint and payload_level is not None:
        parser.error("--temporal_pair_checkpoint currently requires the checkpoint's full payload")
    if args.edge_batch_size > 1 and args.codec_acc:
        parser.error("--edge_batch_size > 1 is only supported without --codec_acc")
    if args.edge_batch_size > 1 and payload_level is not None:
        parser.error("--edge_batch_size > 1 does not yet support --level")
    if args.codec_projection_batch_size > 1 and not (
        args.codec_acc and args.codec_mv_backend == "decoder"
    ):
        parser.error(
            "--codec_projection_batch_size > 1 requires decoder --codec_acc"
        )

    total_start = time.perf_counter()
    decode_start = time.perf_counter()
    decoder_records = None
    if args.codec_acc and args.codec_mv_backend == "decoder":
        decoder_records, native_fps, reader = read_video_records_with_mvs(
            args.video,
            max_frames=args.max_frames,
            sample_fps=args.sample_fps,
        )
        frames = [record["image"] for record in decoder_records if record["selected"]]
    else:
        frames, native_fps, reader = read_video_frames(
            args.video,
            max_frames=args.max_frames,
            sample_fps=args.sample_fps,
        )
    decode_seconds = time.perf_counter() - decode_start
    print(f"Decoded {len(frames)} frames with {reader} (native_fps={native_fps:.3f})")

    edge_load_start = time.perf_counter()
    edge = EdgeEncoder(args.edge_checkpoint, device=args.device)
    temporal_fusion = None
    temporal_metadata = None
    if args.temporal_pair_checkpoint:
        temporal_fusion, temporal_metadata = load_temporal_pair_fusion(
            args.temporal_pair_checkpoint, device=args.device
        )
        checkpoint_split_layer = int(
            temporal_metadata.get("split_layer", args.split_layer)
        )
        if checkpoint_split_layer != args.split_layer:
            parser.error(
                "--split_layer does not match temporal checkpoint: "
                f"{args.split_layer} != {checkpoint_split_layer}"
            )
        if int(temporal_metadata.get("temporal_patch_size", 2)) != 2:
            parser.error("Only temporal_patch_size=2 checkpoints are supported")
        temporal_fusion.eval()
    edge_load_seconds = time.perf_counter() - edge_load_start
    frame_features = []
    edge_encode_seconds = 0.0
    farneback_accelerator = (
        CodecWarpAccelerator(edge, gop_frames=args.codec_gop_frames)
        if args.codec_acc and args.codec_mv_backend == "farneback"
        else None
    )
    decoder_accelerator = (
        DecoderMotionVectorAccelerator(
            edge,
            flow_impl=args.codec_flow_impl,
            memory_checkpoint=args.codec_memory_checkpoint,
            memory_arch=args.codec_memory_arch,
            reference_mode=args.codec_reference_mode,
            min_coverage=args.codec_mv_min_coverage,
            max_p_chain=args.codec_max_p_chain,
            memory_rgb_mode=args.codec_memory_rgb_mode,
        )
        if args.codec_acc and args.codec_mv_backend == "decoder"
        else None
    )
    codec_frame_records = []
    if temporal_fusion is not None:
        encode_start = time.perf_counter()
        paired_frames = []
        for index in range(0, len(frames), 2):
            frame0 = frames[index]
            frame1 = frames[index + 1] if index + 1 < len(frames) else frame0
            paired_frames.extend((frame0, frame1))
        image_tensor = torch.stack(
            [edge.transform(frame.convert("RGB")) for frame in paired_frames],
            dim=0,
        )
        image_tensor = edge.input_to_device(image_tensor)
        with torch.inference_mode():
            backbone = edge.student(image_tensor)[-1]
            fused = temporal_fusion(backbone[0::2], backbone[1::2])
            tokens = edge.projector(fused)
            if edge.bottleneck is not None:
                tokens = edge.bottleneck.encode(tokens)
            frame_features.extend(tokens[index].detach() for index in range(tokens.shape[0]))
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
        edge_encode_seconds = time.perf_counter() - encode_start
        print(
            f"Encoded {len(frames)} frames as {len(frame_features)} native temporal pairs: "
            f"{tuple(tokens.shape)}"
        )
    elif decoder_accelerator is not None:
        deferred_features = []
        for record in decoder_records:
            features, is_compressed, codec_info = decoder_accelerator.encode_record(
                record,
                payload_level=payload_level,
                defer_payload=args.codec_projection_batch_size > 1,
            )
            codec_frame_records.append(codec_info)
            edge_encode_seconds += codec_info["encode_seconds"]
            if not record["selected"]:
                continue
            if args.codec_projection_batch_size > 1:
                deferred_features.append(features)
            else:
                frame_features.append(features.squeeze(0).detach())
            print(
                f"Encoded sampled frame {len(frame_features)}/{len(frames)} "
                f"(source={record['source_index']}): {tuple(features.shape)}, "
                f"compressed={is_compressed}, mode={codec_info['mode']}"
            )
        if args.codec_projection_batch_size > 1:
            projection_start = time.perf_counter()
            for start in range(
                0, len(deferred_features), args.codec_projection_batch_size
            ):
                feature_batch = torch.cat(
                    deferred_features[
                        start : start + args.codec_projection_batch_size
                    ],
                    dim=0,
                )
                payload_batch, _ = decoder_accelerator.project_features(
                    feature_batch, payload_level=payload_level
                )
                frame_features.extend(
                    payload_batch[index].detach()
                    for index in range(payload_batch.shape[0])
                )
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize()
            edge_encode_seconds += time.perf_counter() - projection_start
    elif args.edge_batch_size > 1:
        encode_start = time.perf_counter()
        for start in range(0, len(frames), args.edge_batch_size):
            batch = frames[start : start + args.edge_batch_size]
            features, is_compressed = edge.encode_pil_batch(batch)
            frame_features.extend(
                features[index].detach() for index in range(features.shape[0])
            )
            print(
                f"Encoded frames {start + 1}-{start + len(batch)}/{len(frames)}: "
                f"{tuple(features.shape)}, compressed={is_compressed}, mode=full_batch"
            )
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
        edge_encode_seconds = time.perf_counter() - encode_start
    else:
        for idx, frame in enumerate(frames):
            encode_start = time.perf_counter()
            if farneback_accelerator is not None:
                features, is_compressed, codec_info = farneback_accelerator.encode_pil(
                    frame, payload_level=payload_level
                )
                codec_frame_records.append(codec_info)
            elif payload_level:
                features, is_compressed = edge.encode_pil_level(frame, payload_level)
            else:
                features, is_compressed = edge.encode_pil(frame)
            edge_encode_seconds += time.perf_counter() - encode_start
            frame_features.append(features.squeeze(0).detach())
            mode = codec_frame_records[-1]["frame_type"] if codec_frame_records else "full"
            print(
                f"Encoded frame {idx + 1}/{len(frames)}: {tuple(features.shape)}, "
                f"compressed={is_compressed}, mode={mode}"
            )

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
                "codec_acc": args.codec_acc,
                "temporal_pair_checkpoint": args.temporal_pair_checkpoint,
                "temporal_patch_size": 2 if temporal_fusion is not None else 1,
                "codec_gop_frames": args.codec_gop_frames
                if args.codec_acc and args.codec_mv_backend == "farneback" else None,
                "codec_mv_backend": args.codec_mv_backend if args.codec_acc else None,
                "codec_flow_impl": args.codec_flow_impl
                if args.codec_acc and args.codec_mv_backend == "decoder" else None,
                "codec_memory_checkpoint": args.codec_memory_checkpoint
                if args.codec_acc and args.codec_mv_backend == "decoder" else None,
                "codec_memory_arch": args.codec_memory_arch
                if args.codec_acc and args.codec_mv_backend == "decoder" else None,
                "codec_reference_mode": args.codec_reference_mode
                if args.codec_acc and args.codec_mv_backend == "decoder" else None,
                "codec_mv_min_coverage": args.codec_mv_min_coverage
                if args.codec_acc and args.codec_mv_backend == "decoder" else None,
                "codec_max_p_chain": args.codec_max_p_chain
                if args.codec_acc and args.codec_mv_backend == "decoder" else None,
                "codec_memory_rgb_mode": args.codec_memory_rgb_mode
                if args.codec_acc and args.codec_mv_backend == "decoder" else None,
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
        "video_grid_thw": [
            int(compressed_frame_features.shape[0]),
            target_side,
            target_side,
        ],
        "edge_checkpoint": args.edge_checkpoint,
        "cloud_checkpoint": args.cloud_checkpoint,
        "qwen_path": args.qwen_path,
        "split_layer": args.split_layer,
        "device": args.device,
        "edge_batch_size": args.edge_batch_size,
        "codec_projection_batch_size": args.codec_projection_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "payload_level": args.level,
        "temporal_pair_checkpoint": args.temporal_pair_checkpoint,
        "temporal_pair_metadata": {
            "split_layer": temporal_metadata.get("split_layer"),
            "temporal_patch_size": temporal_metadata.get("temporal_patch_size", 2),
            "sample_fps": args.sample_fps,
            "seconds_per_grid": (
                2.0 / args.sample_fps
                if args.sample_fps and args.sample_fps > 0
                else None
            ),
        } if temporal_metadata is not None else None,
        "temporal_grid_count": int(compressed_frame_features.shape[0]),
        "codec_acc": args.codec_acc,
        "codec_gop_frames": args.codec_gop_frames
        if args.codec_acc and args.codec_mv_backend == "farneback" else None,
        "codec_mv_backend": args.codec_mv_backend if args.codec_acc else None,
        "codec_flow_impl": args.codec_flow_impl
        if args.codec_acc and args.codec_mv_backend == "decoder" else None,
        "codec_memory_checkpoint": args.codec_memory_checkpoint
        if args.codec_acc and args.codec_mv_backend == "decoder" else None,
        "codec_memory_arch": args.codec_memory_arch
        if args.codec_acc and args.codec_mv_backend == "decoder" else None,
        "codec_reference_mode": args.codec_reference_mode
        if args.codec_acc and args.codec_mv_backend == "decoder" else None,
        "codec_mv_min_coverage": args.codec_mv_min_coverage
        if args.codec_acc and args.codec_mv_backend == "decoder" else None,
        "codec_max_p_chain": args.codec_max_p_chain
        if args.codec_acc and args.codec_mv_backend == "decoder" else None,
        "codec_memory_rgb_mode": args.codec_memory_rgb_mode
        if args.codec_acc and args.codec_mv_backend == "decoder" else None,
        "codec_mode": (
            f"{args.codec_mv_backend}_{args.codec_memory_arch}_residual"
            if args.codec_acc
            and args.codec_mv_backend == "decoder"
            and args.codec_memory_checkpoint
            else f"{args.codec_mv_backend}_warp_only"
            if args.codec_acc else None
        ),
        "codec_frame_records": codec_frame_records,
        "codec_source_frames_processed": len(codec_frame_records) if args.codec_acc else len(frames),
        "decoder_mv_reference_policy": (
            "last decoded I/P reference; selected B frames use full-CNN fallback"
            if args.codec_acc and args.codec_mv_backend == "decoder" else None
        ),
        "edge_cnn_frames": sum(record["cnn_executed"] for record in codec_frame_records)
        if args.codec_acc else len(frames),
        "edge_warp_frames": sum(
            record.get("warp_executed", not record["cnn_executed"])
            for record in codec_frame_records
        ),
        "sampled_edge_cnn_frames": sum(
            record["selected"] and record["cnn_executed"] for record in codec_frame_records
        ) if args.codec_acc and args.codec_mv_backend == "decoder" else (
            sum(record["cnn_executed"] for record in codec_frame_records)
            if args.codec_acc else len(frames)
        ),
        "sampled_edge_warp_frames": sum(
            record["selected"] and record["warp_executed"] for record in codec_frame_records
        ) if args.codec_acc and args.codec_mv_backend == "decoder" else (
            sum(not record["cnn_executed"] for record in codec_frame_records)
            if args.codec_acc else 0
        ),
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
            f"- Codec acceleration: `{args.codec_acc}`",
            f"- Codec MV backend: `{args.codec_mv_backend if args.codec_acc else None}`",
            f"- Codec memory: `{args.codec_memory_checkpoint}`",
            f"- Codec memory architecture: `{args.codec_memory_arch}`",
            f"- Codec reference mode: `{args.codec_reference_mode}`",
            f"- MV coverage threshold: `{args.codec_mv_min_coverage}`",
            f"- Max causal P chain: `{args.codec_max_p_chain}`",
            f"- Codec memory RGB mode: `{args.codec_memory_rgb_mode}`",
            f"- Codec GOP frames: `{metadata['codec_gop_frames']}`",
            f"- Edge CNN / warp frames: `{metadata['edge_cnn_frames']} / {metadata['edge_warp_frames']}`",
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

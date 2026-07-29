"""Train the MMNet-style codec feature memory on real compressed videos.

The existing SplitOculo checkpoint is trained on independent still images. This
script freezes that edge CNN and learns only a small temporal correction module
against the full CNN feature of each decoded P-frame:

    full CNN(I) -> teacher feature
    MV warp(previous predicted feature) + decoded residual proxy -> prediction

The proxy residual is ``decoded_current - warp(decoded_reference, MV)``. It is
available with the current PyAV reader even though portable PyAV builds do not
expose inverse-transformed codec residual planes directly.

Example:

    python scripts/train_codec_memory.py `
      --edge_checkpoint checkpoints/.../edge_weights.pth `
      --video outputs/codec_mv_ip_only/babycrawling_ip.mp4 `
      --output checkpoints/codec_memory/mmnet_best.pth `
      --epochs 10

Use several real videos with ``--video`` to avoid fitting a single camera
trajectory. Training data should use the same codec family and resolution
range as deployment.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from tqdm import tqdm

from core.codec_video_reader import read_video_records_with_mvs
from models.codec_accelerator import DecoderMotionVectorAccelerator
from models.codec_memory import LSFAFeatureMemory, MMNetFeatureMemory
from scripts.edge_client import EdgeEncoder


def _set_requires_grad(module, enabled: bool):
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _full_feature(edge, accelerator, image):
    normalized, current_rgb = accelerator._prepare(image)
    with torch.no_grad():
        target = edge.student(normalized[None].to(edge.device))[-1]
    return target, current_rgb


def _new_stats(frames):
    return {
        "frames": frames,
        "p_frames": 0,
        "train_steps": 0,
        "loss": 0.0,
        "feature_loss": 0.0,
        "feature_cosine_loss": 0.0,
        "payload_loss": 0.0,
        "payload_cosine_loss": 0.0,
        "coverage": 0.0,
    }


def _record_step(
    stats, loss, feature_loss, cosine_loss, payload_loss, payload_cosine_loss
):
    stats["train_steps"] += 1
    stats["loss"] += float(loss.detach())
    stats["feature_loss"] += float(feature_loss)
    stats["feature_cosine_loss"] += float(cosine_loss)
    stats["payload_loss"] += float(payload_loss)
    stats["payload_cosine_loss"] += float(payload_cosine_loss)


def _finish_stats(stats, native_fps, reader):
    steps = max(stats["train_steps"], 1)
    stats["native_fps"] = native_fps
    stats["reader"] = reader
    for key in (
        "loss",
        "feature_loss",
        "feature_cosine_loss",
        "payload_loss",
        "payload_cosine_loss",
    ):
        stats[key] /= steps
    if stats["p_frames"]:
        stats["coverage"] /= stats["p_frames"]
    return stats


def _payload(edge, feature):
    tokens = edge.projector(feature)
    if edge.bottleneck is not None:
        tokens = edge.bottleneck.encode(tokens)
    return tokens


def _step_loss(
    edge,
    predicted,
    target,
    lambda_cosine,
    lambda_payload,
    lambda_payload_cosine,
):
    feature_loss = F.smooth_l1_loss(predicted, target)
    feature_cosine = 1.0 - F.cosine_similarity(
        predicted.flatten(1), target.flatten(1), dim=1
    ).mean()
    loss = feature_loss + lambda_cosine * feature_cosine

    payload_loss = predicted.new_zeros(())
    payload_cosine_loss = predicted.new_zeros(())
    if lambda_payload > 0 or lambda_payload_cosine > 0:
        predicted_payload = _payload(edge, predicted)
        with torch.no_grad():
            target_payload = _payload(edge, target)
        if lambda_payload > 0:
            payload_loss = F.smooth_l1_loss(predicted_payload, target_payload)
            loss = loss + lambda_payload * payload_loss
        if lambda_payload_cosine > 0:
            payload_cosine_loss = 1.0 - F.cosine_similarity(
                predicted_payload.flatten(1),
                target_payload.flatten(1),
                dim=1,
            ).mean()
            loss = loss + lambda_payload_cosine * payload_cosine_loss
    return (
        loss,
        feature_loss.detach(),
        feature_cosine.detach(),
        payload_loss.detach(),
        payload_cosine_loss.detach(),
    )


def _memory_prediction(
    memory,
    memory_arch,
    warped,
    residual,
    current_rgb,
    feature_flow,
    feature_covered,
):
    """Run one temporal correction with either the legacy or LSFA branch."""
    if memory_arch == "lsfa":
        return memory(
            warped,
            residual[None],
            current_rgb[None],
            feature_flow[None],
            feature_covered[None, None].float(),
        )
    return memory(
        warped,
        residual[None],
        feature_flow[None],
        feature_covered[None, None].float(),
    )


def train_video(
    edge,
    memory,
    optimizer,
    video_path,
    args,
):
    records, native_fps, reader = read_video_records_with_mvs(
        video_path,
        max_frames=args.max_frames,
        sample_fps=args.sample_fps,
    )
    accelerator = DecoderMotionVectorAccelerator(
        edge,
        flow_impl=args.flow_impl,
        min_coverage=0.0,
    )
    accelerator.memory = None

    reference_feature = None
    reference_rgb = None
    stats = _new_stats(len(records))

    for record in records:
        frame_type = record["pict_type"]
        if frame_type == "I":
            target, current_rgb = _full_feature(edge, accelerator, record["image"])
            reference_feature = target.detach()
            reference_rgb = current_rgb
            continue

        if (
            frame_type != "P"
            or reference_feature is None
            or record["motion_vectors"] is None
            or len(record["motion_vectors"]) == 0
        ):
            # B-frames are deliberately not used as causal references. A P
            # frame without MVs becomes a teacher refresh point.
            if frame_type == "P":
                target, current_rgb = _full_feature(edge, accelerator, record["image"])
                reference_feature = target.detach()
                reference_rgb = current_rgb
            continue

        stats["p_frames"] += 1
        feature_height, feature_width = reference_feature.shape[-2:]
        flow, covered, feature_flow, feature_covered = accelerator.build_feature_flow(
            record["motion_vectors"],
            record["image"].size[0],
            record["image"].size[1],
            feature_height,
            feature_width,
        )
        stats["coverage"] += float(covered.float().mean())

        if not covered.any():
            target, current_rgb = _full_feature(edge, accelerator, record["image"])
            reference_feature = target.detach()
            reference_rgb = current_rgb
            continue

        if args.flow_impl in {"feature_grid", "feature_grid_center"}:
            warped = accelerator._warp_feature_grid(reference_feature, flow)
        else:
            warped = accelerator._warp_feature(reference_feature, flow)

        accelerator.reference_rgb = reference_rgb
        target, current_rgb = _full_feature(edge, accelerator, record["image"])
        residual = accelerator._residual_proxy(current_rgb, feature_flow, feature_covered)

        predicted = _memory_prediction(
            memory,
            args.memory_arch,
            warped,
            residual,
            current_rgb,
            feature_flow,
            feature_covered,
        )
        loss, feature_loss, cosine_loss, payload_loss, payload_cosine_loss = _step_loss(
            edge,
            predicted,
            target,
            args.lambda_cosine,
            args.lambda_payload,
            args.lambda_payload_cosine,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(memory.parameters(), args.grad_clip)
        optimizer.step()

        _record_step(
            stats,
            loss,
            feature_loss,
            cosine_loss,
            payload_loss,
            payload_cosine_loss,
        )

        # Scheduled sampling: the next P-frame sees this prediction, but the
        # graph is cut so a long GOP does not create an unbounded BPTT graph.
        reference_feature = predicted.detach()
        reference_rgb = current_rgb

    return _finish_stats(stats, native_fps, reader)


def _cache_matches(cache_path, video_path, args):
    if not cache_path.is_file():
        return False
    try:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    config = cache.get("config", {})
    return (
        cache.get("video") == str(Path(video_path).resolve())
        and config.get("max_frames") == args.max_frames
        and config.get("sample_fps") == args.sample_fps
        and config.get("flow_impl") == args.flow_impl
    )


@torch.no_grad()
def _build_video_cache(edge, video_path, args, cache_path):
    records, native_fps, reader = read_video_records_with_mvs(
        video_path,
        max_frames=args.max_frames,
        sample_fps=args.sample_fps,
    )
    accelerator = DecoderMotionVectorAccelerator(edge, flow_impl=args.flow_impl)
    teacher_indices = []
    normalized_images = []
    rgb_by_index = {}
    for index, record in enumerate(records):
        if record["pict_type"] not in {"I", "P"}:
            continue
        normalized, current_rgb = accelerator._prepare(record["image"])
        teacher_indices.append(index)
        normalized_images.append(normalized)
        rgb_by_index[index] = current_rgb

    targets = {}
    for start in range(0, len(normalized_images), args.target_batch_size):
        batch = torch.stack(normalized_images[start : start + args.target_batch_size])
        features = edge.student(batch.to(edge.device))[-1].cpu()
        for offset, feature in enumerate(features):
            targets[teacher_indices[start + offset]] = feature.contiguous()

    if not targets:
        raise RuntimeError(f"No I/P teacher frames found in {video_path}")
    feature_height, feature_width = next(iter(targets.values())).shape[-2:]
    frames = []
    for index, record in enumerate(records):
        frame = {
            "source_index": record["source_index"],
            "pict_type": record["pict_type"],
            "target": targets.get(index),
            "rgb": None,
            "warp_flow": None,
            "warp_covered": None,
            "feature_flow": None,
            "feature_covered": None,
        }
        if index in rgb_by_index:
            frame["rgb"] = F.interpolate(
                rgb_by_index[index][None],
                size=(feature_height, feature_width),
                mode="bilinear",
                align_corners=False,
            )[0].contiguous()
        if (
            record["pict_type"] == "P"
            and record["motion_vectors"] is not None
            and len(record["motion_vectors"]) > 0
        ):
            flow, covered, feature_flow, feature_covered = accelerator.build_feature_flow(
                record["motion_vectors"],
                record["image"].size[0],
                record["image"].size[1],
                feature_height,
                feature_width,
            )
            frame["warp_flow"] = flow.contiguous()
            frame["warp_covered"] = covered.contiguous()
            frame["feature_flow"] = feature_flow.contiguous()
            frame["feature_covered"] = feature_covered.contiguous()
        frames.append(frame)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "video": str(Path(video_path).resolve()),
            "native_fps": native_fps,
            "reader": reader,
            "feature_shape": [int(feature_height), int(feature_width)],
            "config": {
                "max_frames": args.max_frames,
                "sample_fps": args.sample_fps,
                "flow_impl": args.flow_impl,
            },
            "frames": frames,
        },
        cache_path,
    )


def train_cached_video(edge, memory, optimizer, cache_path, args):
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    frames = cache["frames"]
    accelerator = DecoderMotionVectorAccelerator(edge, flow_impl=args.flow_impl)
    reference_feature = None
    reference_rgb = None
    stats = _new_stats(len(frames))

    for frame in frames:
        frame_type = frame["pict_type"]
        target_cpu = frame["target"]
        if frame_type == "I":
            reference_feature = target_cpu[None].to(edge.device)
            reference_rgb = frame["rgb"]
            continue

        if frame_type != "P" or target_cpu is None or reference_feature is None:
            if frame_type == "P" and target_cpu is not None:
                reference_feature = target_cpu[None].to(edge.device)
                reference_rgb = frame["rgb"]
            continue

        stats["p_frames"] += 1
        covered = frame["warp_covered"]
        if covered is None:
            reference_feature = target_cpu[None].to(edge.device)
            reference_rgb = frame["rgb"]
            continue
        stats["coverage"] += float(covered.float().mean())
        if not covered.any():
            reference_feature = target_cpu[None].to(edge.device)
            reference_rgb = frame["rgb"]
            continue

        warp_flow = frame["warp_flow"]
        if args.flow_impl in {"feature_grid", "feature_grid_center"}:
            warped = accelerator._warp_feature_grid(
                reference_feature, warp_flow.to(edge.device)
            )
        else:
            warped = accelerator._warp_feature(
                reference_feature, warp_flow.to(edge.device)
            )

        accelerator.reference_rgb = reference_rgb
        feature_flow = frame["feature_flow"]
        feature_covered = frame["feature_covered"]
        residual = accelerator._residual_proxy(
            frame["rgb"], feature_flow, feature_covered
        )
        predicted = _memory_prediction(
            memory,
            args.memory_arch,
            warped,
            residual.to(edge.device),
            frame["rgb"].to(edge.device),
            feature_flow.to(edge.device),
            feature_covered.to(edge.device),
        )
        target = target_cpu[None].to(edge.device)
        loss, feature_loss, cosine_loss, payload_loss, payload_cosine_loss = _step_loss(
            edge,
            predicted,
            target,
            args.lambda_cosine,
            args.lambda_payload,
            args.lambda_payload_cosine,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(memory.parameters(), args.grad_clip)
        optimizer.step()
        _record_step(
            stats,
            loss,
            feature_loss,
            cosine_loss,
            payload_loss,
            payload_cosine_loss,
        )
        reference_feature = predicted.detach()
        reference_rgb = frame["rgb"]

    return _finish_stats(stats, cache["native_fps"], "cached_codec_records")


def parse_args():
    parser = argparse.ArgumentParser(description="Train an MMNet-style codec feature memory")
    parser.add_argument("--edge_checkpoint", required=True)
    parser.add_argument("--video", nargs="+", default=None, help="Compressed video paths")
    parser.add_argument(
        "--video_manifest",
        default=None,
        help="Text file with one compressed video path per line; comments start with #",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=200,
        help="Maximum number of videos used per run; excess paths are deterministically sampled",
    )
    parser.add_argument("--output", required=True, help="Output memory checkpoint")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--resume_checkpoint",
        default=None,
        help="Continue from a memory checkpoint; --epochs is the desired total epoch count.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_cosine", type=float, default=0.25)
    parser.add_argument("--lambda_payload", type=float, default=0.5)
    parser.add_argument(
        "--lambda_payload_cosine",
        type=float,
        default=0.0,
        help="Directly optimize cosine similarity of transmitted payloads.",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--sample_fps", type=float, default=None)
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Optional disk cache for teacher features and MV inputs; avoids re-decoding every epoch",
    )
    parser.add_argument(
        "--cache_only",
        action="store_true",
        help="Build --cache_dir and exit without training",
    )
    parser.add_argument("--target_batch_size", type=int, default=32)
    parser.add_argument(
        "--flow_impl",
        choices=["feature_grid", "feature_grid_center", "dense"],
        default="feature_grid",
    )
    parser.add_argument(
        "--memory_arch",
        choices=["mmnet", "lsfa"],
        default="lsfa",
        help="Temporal correction architecture; lsfa adds residual and current-RGB branches",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.max_videos <= 0:
        raise ValueError(f"max_videos must be positive, got {args.max_videos}")
    if args.target_batch_size <= 0:
        raise ValueError(f"target_batch_size must be positive, got {args.target_batch_size}")

    video_paths = list(args.video or [])
    if args.video_manifest:
        manifest_path = Path(args.video_manifest)
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip().lstrip("\ufeff")
            if not line or line.startswith("#"):
                continue
            path = Path(line)
            if not path.is_absolute():
                path = (manifest_path.parent / path).resolve()
            video_paths.append(str(path))
    video_paths = list(dict.fromkeys(video_paths))
    if not video_paths:
        raise ValueError("Provide at least one --video path or --video_manifest")
    if len(video_paths) > args.max_videos:
        video_paths = random.Random(args.seed).sample(video_paths, args.max_videos)
        video_paths.sort()
    print(f"Using {len(video_paths)} training videos (max_videos={args.max_videos})")

    edge = EdgeEncoder(args.edge_checkpoint, device=args.device)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, edge.image_size, edge.image_size, device=edge.device)
        feature_channels = int(edge.student(dummy)[-1].shape[1])

    _set_requires_grad(edge.student, False)
    _set_requires_grad(edge.projector, False)
    if edge.bottleneck is not None:
        _set_requires_grad(edge.bottleneck, False)
    edge.student.eval()
    edge.projector.eval()
    if edge.bottleneck is not None:
        edge.bottleneck.eval()

    cache_paths = []
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        for index, video_path in enumerate(tqdm(video_paths, desc="Cache")):
            cache_path = cache_dir / f"{index:04d}.pt"
            if not _cache_matches(cache_path, video_path, args):
                _build_video_cache(edge, video_path, args, cache_path)
            cache_paths.append(cache_path)
        print(f"Prepared {len(cache_paths)} cached videos in {cache_dir}")
        if args.cache_only:
            return

    memory_type = LSFAFeatureMemory if args.memory_arch == "lsfa" else MMNetFeatureMemory
    memory = memory_type(feature_channels=feature_channels).to(args.device)
    history = []
    best_loss = float("inf")
    if args.resume_checkpoint:
        resume = torch.load(
            args.resume_checkpoint, map_location=args.device, weights_only=False
        )
        resume_arch = resume.get("memory_arch", resume.get("args", {}).get("memory_arch"))
        if resume_arch is not None and str(resume_arch) != args.memory_arch:
            raise ValueError(
                f"Resume checkpoint architecture is {resume_arch}, "
                f"but --memory_arch is {args.memory_arch}"
            )
        resume_channels = resume.get("feature_channels")
        if resume_channels is not None and int(resume_channels) != feature_channels:
            raise ValueError(
                f"Resume checkpoint has {resume_channels} channels, expected {feature_channels}"
            )
        state_dict = resume.get(
            "memory_state_dict", resume.get("model_state_dict", resume)
        )
        memory.load_state_dict(state_dict)
        history = list(resume.get("history", []))
        if history:
            best_loss = min(float(item["loss"]) for item in history)
        print(
            f"Resumed memory from {args.resume_checkpoint} "
            f"after {len(history)} recorded epochs"
        )
    optimizer = torch.optim.AdamW(
        memory.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    training_sources = cache_paths or video_paths
    start_epoch = len(history) + 1
    if start_epoch > args.epochs:
        raise ValueError(
            f"Checkpoint already has {len(history)} epochs, "
            f"but --epochs is only {args.epochs}"
        )
    for epoch in range(start_epoch, args.epochs + 1):
        memory.train()
        epoch_stats = []
        for source in tqdm(training_sources, desc=f"Epoch {epoch}"):
            if cache_paths:
                epoch_stats.append(train_cached_video(edge, memory, optimizer, source, args))
            else:
                epoch_stats.append(train_video(edge, memory, optimizer, source, args))

        count = max(len(epoch_stats), 1)
        summary = {
            "epoch": epoch,
            "loss": sum(item["loss"] for item in epoch_stats) / count,
            "feature_cosine_loss": sum(item["feature_cosine_loss"] for item in epoch_stats)
            / count,
            "payload_loss": sum(item["payload_loss"] for item in epoch_stats) / count,
            "payload_cosine_loss": sum(
                item["payload_cosine_loss"] for item in epoch_stats
            )
            / count,
            "train_steps": sum(item["train_steps"] for item in epoch_stats),
            "coverage": sum(item["coverage"] for item in epoch_stats) / count,
        }
        history.append(summary)
        print(json.dumps(summary, ensure_ascii=False))

        output_path = Path(args.output)
        if summary["loss"] < best_loss or not output_path.is_file():
            best_loss = summary["loss"]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "memory_state_dict": memory.state_dict(),
                    "feature_channels": feature_channels,
                    "memory_arch": args.memory_arch,
                    "residual_mode": "decoded_rgb_minus_mv_warp",
                    "video_paths": video_paths,
                    "cache_paths": [str(path) for path in cache_paths],
                    "args": vars(args),
                    "history": history,
                },
                output_path,
            )
            print(f"Saved best memory checkpoint: {output_path}")


if __name__ == "__main__":
    main()

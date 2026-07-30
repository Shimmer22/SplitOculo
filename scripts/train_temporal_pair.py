"""Cache Qwen native-pair targets and train SplitOculo temporal fusion.

The cache stage loads Qwen only long enough to create layer-N teacher targets.
The train stage then loads only the existing SplitOculo edge/cloud checkpoints,
which keeps training practical on 6 GB GPUs.
"""

import argparse
import gc
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from core.qwen_extractor import QwenFeatureExtractor
from models.temporal_pair import TemporalPairFusion
from scripts.cloud_server import CloudInferenceEngine
from scripts.edge_client import EdgeEncoder


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def find_videos(video_dir=None, videos=None):
    paths = []
    if video_dir:
        root = Path(video_dir)
        paths.extend(
            path for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        )
    paths.extend(Path(path) for path in (videos or []))
    unique = []
    seen = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    if not unique:
        raise ValueError("No videos found. Pass --video or --video_dir.")
    return unique


def pil_to_uint8_tensor(image):
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1)


def uint8_tensor_to_pil(tensor):
    array = tensor.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)


def square_center_crop(image, image_size):
    """Match the existing edge Resize(short side)+CenterCrop contract."""
    return ImageOps.fit(
        image.convert("RGB"),
        (int(image_size), int(image_size)),
        method=Image.Resampling.BICUBIC,
        centering=(0.5, 0.5),
    )


def read_video_pairs(video_path, pair_count, sample_fps):
    """Read evenly spaced pairs without decoding an entire video."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    gap = max(1, int(round(native_fps / sample_fps))) if native_fps > 0 else 1
    latest_start = max(0, total_frames - gap - 1)
    if pair_count == 1:
        starts = [latest_start // 2]
    else:
        starts = np.linspace(0, latest_start, pair_count + 2)[1:-1]
        starts = np.rint(starts).astype(int).tolist()

    pairs = []
    for start in starts:
        decoded = []
        for frame_index in (int(start), min(int(start) + gap, max(0, total_frames - 1))):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                decoded = []
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            decoded.append(Image.fromarray(frame))
        if len(decoded) == 2:
            pairs.append((decoded[0], decoded[1], int(start), int(start) + gap))
    cap.release()
    return pairs, native_fps, "cv2_seek"


def create_teacher_cache(args):
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for stale in cache_dir.glob("pair_*.pt"):
        stale.unlink()

    videos = find_videos(args.video_dir, args.video)
    samples = []
    for video_path in videos:
        pairs, native_fps, reader = read_video_pairs(
            video_path, args.max_pairs_per_video, args.sample_fps
        )
        for pair_index, (raw0, raw1, source_frame0, source_frame1) in enumerate(pairs):
            frame0 = square_center_crop(raw0, args.image_size)
            frame1 = square_center_crop(raw1, args.image_size)
            samples.append(
                {
                    "frame0": frame0,
                    "frame1": frame1,
                    "source": str(video_path),
                    "source_pair": pair_index,
                    "source_frames": [source_frame0, source_frame1],
                    "native_fps": native_fps,
                    "reader": reader,
                    "static": False,
                    "label": video_path.parent.name,
                    "split": video_path.parent.parent.name,
                }
            )

    if not samples:
        raise RuntimeError("Videos decoded zero frame pairs")

    static_count = int(round(len(samples) * args.static_ratio / max(1e-8, 1 - args.static_ratio)))
    samples_by_label = {}
    for sample in samples:
        samples_by_label.setdefault(sample["label"], []).append(sample)
    label_names = sorted(samples_by_label)
    for index in range(static_count):
        label = label_names[index % len(label_names)]
        label_index = index // len(label_names)
        candidates = samples_by_label[label]
        source = candidates[label_index % len(candidates)]
        samples.append(
            {
                **source,
                "frame1": source["frame0"],
                "source_pair": source["source_pair"],
                "source_frames": [source["source_frames"][0]] * 2,
                "static": True,
            }
        )
    random.Random(args.seed).shuffle(samples)

    extractor = QwenFeatureExtractor(
        model_name=args.qwen_path,
        device=args.device,
        extract_layer=args.split_layer,
        local_files_only=args.offline,
        min_pixels=args.image_size * args.image_size,
        max_pixels=args.image_size * args.image_size,
    ).load()

    manifest = {
        "qwen_path": args.qwen_path,
        "split_layer": args.split_layer,
        "sample_fps": args.sample_fps,
        "image_size": args.image_size,
        "temporal_patch_size": 2,
        "samples": [],
    }
    for sample_index, sample in enumerate(tqdm(samples, desc="Caching native Qwen pairs")):
        teacher, grid_thw = extractor.extract_video_features(
            [sample["frame0"], sample["frame1"]]
        )
        if int(grid_thw[0]) != 1:
            raise RuntimeError(f"Two frames should produce grid_t=1, got {grid_thw.tolist()}")
        expected_side = args.image_size // 14
        if tuple(int(value) for value in grid_thw.tolist()) != (
            1,
            expected_side,
            expected_side,
        ):
            raise RuntimeError(
                f"Expected square teacher grid [1,{expected_side},{expected_side}], "
                f"got {grid_thw.tolist()}"
            )
        output_path = cache_dir / f"pair_{sample_index:05d}.pt"
        torch.save(
            {
                "frames_uint8": torch.stack(
                    (
                        pil_to_uint8_tensor(sample["frame0"]),
                        pil_to_uint8_tensor(sample["frame1"]),
                    )
                ),
                "teacher_features": teacher.float(),
                "video_grid_thw": grid_thw,
                "static": sample["static"],
                "source": sample["source"],
                "source_pair": sample["source_pair"],
                "source_frames": sample["source_frames"],
                "label": sample["label"],
                "split": sample["split"],
            },
            output_path,
        )
        manifest["samples"].append(
            {
                "file": output_path.name,
                "static": sample["static"],
                "source": sample["source"],
                "source_pair": sample["source_pair"],
                "source_frames": sample["source_frames"],
                "label": sample["label"],
                "split": sample["split"],
                "teacher_shape": list(teacher.shape),
                "video_grid_thw": grid_thw.tolist(),
            }
        )

    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Cached {len(samples)} temporal pairs in {cache_dir}")

    del extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class CachedTemporalPairDataset(Dataset):
    def __init__(self, cache_dir, transform):
        self.files = sorted(Path(cache_dir).glob("pair_*.pt"))
        if not self.files:
            raise ValueError(f"No pair_*.pt files found in {cache_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        item = torch.load(self.files[index], map_location="cpu", weights_only=False)
        images = [
            self.transform(uint8_tensor_to_pil(frame))
            for frame in item["frames_uint8"]
        ]
        return (
            torch.stack(images),
            item["teacher_features"].float(),
            torch.tensor(bool(item["static"]), dtype=torch.bool),
            item.get("label", "unknown"),
        )


def freeze(module):
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


def build_frozen_pipeline(args):
    device = torch.device(args.device)
    edge = EdgeEncoder(args.edge_checkpoint, device=args.device)
    cloud = CloudInferenceEngine(
        args.cloud_checkpoint, device=args.device, split_layer=args.split_layer
    )
    freeze(edge.student)
    freeze(edge.projector)
    if edge.bottleneck is not None:
        freeze(edge.bottleneck)
    freeze(cloud.upsampler)
    cloud.upsampler.float()
    if cloud.bottleneck is not None:
        freeze(cloud.bottleneck)
        cloud.bottleneck.float()

    with torch.no_grad():
        dummy = torch.zeros(1, 3, edge.image_size, edge.image_size, device=device)
        in_channels = int(edge.student(dummy)[-1].shape[1])
    return device, edge, cloud, in_channels


def forward_pair_pipeline(edge, cloud, fusion, frames):
    batch, pair, channels, height, width = frames.shape
    with torch.no_grad():
        backbone = edge.student(
            frames.reshape(batch * pair, channels, height, width)
        )[-1]
        feature0, feature1 = backbone.reshape(
            batch, pair, *backbone.shape[1:]
        ).unbind(dim=1)

    fused = fusion(feature0, feature1)
    tokens = edge.projector(fused)
    if edge.bottleneck is not None:
        payload = edge.bottleneck.encode(tokens)
        reconstructed = cloud.bottleneck.decode(payload)
    else:
        reconstructed = tokens
    student = cloud.upsampler(reconstructed.float())
    return student, tokens, reconstructed, feature0, feature1


def old_single_output(edge, cloud, feature):
    tokens = edge.projector(feature)
    if edge.bottleneck is not None:
        payload = edge.bottleneck.encode(tokens)
        reconstructed = cloud.bottleneck.decode(payload)
    else:
        reconstructed = tokens
    return cloud.upsampler(reconstructed.float())


def feature_metrics(student, teacher):
    per_sample_mse = (student.float() - teacher.float()).square().mean(dim=(1, 2))
    per_token_cosine = F.cosine_similarity(
        student.float(), teacher.float(), dim=-1
    ).mean(dim=1)
    return per_sample_mse, per_token_cosine


@torch.no_grad()
def evaluate_fusion(edge, cloud, fusion, loader, device):
    fusion.eval()
    groups = {
        "all": {"count": 0, "mse": 0.0, "cosine": 0.0, "baseline_mse": 0.0, "baseline_cosine": 0.0},
        "dynamic": {"count": 0, "mse": 0.0, "cosine": 0.0, "baseline_mse": 0.0, "baseline_cosine": 0.0},
        "static": {"count": 0, "mse": 0.0, "cosine": 0.0, "baseline_mse": 0.0, "baseline_cosine": 0.0},
    }
    by_label = {}
    static_keep = {"count": 0, "mse": 0.0, "cosine": 0.0}
    for frames, teacher, is_static, labels in loader:
        frames = frames.to(device, non_blocking=True)
        teacher = teacher.to(device, non_blocking=True)
        is_static = is_static.to(device)
        student, _, _, feature0, feature1 = forward_pair_pipeline(
            edge, cloud, fusion, frames
        )

        class MeanFusion(torch.nn.Module):
            def forward(self, first, second):
                return (first + second) * 0.5

        baseline_student, _, _, _, _ = forward_pair_pipeline(
            edge, cloud, MeanFusion(), frames
        )
        mse, cosine = feature_metrics(student, teacher)
        baseline_mse, baseline_cosine = feature_metrics(baseline_student, teacher)
        if bool(is_static.any()):
            old_student = old_single_output(edge, cloud, feature0[is_static])
            keep_mse, keep_cosine = feature_metrics(
                student[is_static], old_student
            )
            static_keep["count"] += int(is_static.sum())
            static_keep["mse"] += float(keep_mse.sum())
            static_keep["cosine"] += float(keep_cosine.sum())
        for name, mask in (
            ("all", torch.ones_like(is_static, dtype=torch.bool)),
            ("dynamic", ~is_static),
            ("static", is_static),
        ):
            count = int(mask.sum())
            if not count:
                continue
            groups[name]["count"] += count
            groups[name]["mse"] += float(mse[mask].sum())
            groups[name]["cosine"] += float(cosine[mask].sum())
            groups[name]["baseline_mse"] += float(baseline_mse[mask].sum())
            groups[name]["baseline_cosine"] += float(baseline_cosine[mask].sum())
        for sample_index, label in enumerate(labels):
            if bool(is_static[sample_index]):
                continue
            values = by_label.setdefault(
                label,
                {
                    "count": 0,
                    "mse": 0.0,
                    "cosine": 0.0,
                    "baseline_mse": 0.0,
                    "baseline_cosine": 0.0,
                },
            )
            values["count"] += 1
            values["mse"] += float(mse[sample_index])
            values["cosine"] += float(cosine[sample_index])
            values["baseline_mse"] += float(baseline_mse[sample_index])
            values["baseline_cosine"] += float(baseline_cosine[sample_index])

    result = {}
    for name, values in groups.items():
        count = values.pop("count")
        result[name] = {"count": count}
        for key, value in values.items():
            result[name][key] = value / count if count else None
        if count:
            result[name]["mse_improvement"] = (
                result[name]["baseline_mse"] - result[name]["mse"]
            )
            result[name]["cosine_improvement"] = (
                result[name]["cosine"] - result[name]["baseline_cosine"]
            )
    result["by_label_dynamic"] = {}
    for label, values in sorted(by_label.items()):
        count = values.pop("count")
        result["by_label_dynamic"][label] = {
            "count": count,
            **{key: value / count for key, value in values.items()},
        }
    if static_keep["count"]:
        result["static"]["old_single_mse"] = (
            static_keep["mse"] / static_keep["count"]
        )
        result["static"]["old_single_cosine"] = (
            static_keep["cosine"] / static_keep["count"]
        )
    return result


def make_loader(cache_dir, transform, batch_size, shuffle):
    return DataLoader(
        CachedTemporalPairDataset(cache_dir, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )


def train_temporal_fusion(args):
    device, edge, cloud, in_channels = build_frozen_pipeline(args)
    fusion = TemporalPairFusion(
        in_channels=in_channels, hidden_channels=args.temporal_hidden
    ).to(device)

    loader = make_loader(
        args.cache_dir, edge.transform, args.batch_size, shuffle=True
    )
    val_loader = (
        make_loader(args.val_cache_dir, edge.transform, args.batch_size, shuffle=False)
        if args.val_cache_dir else None
    )
    optimizer = torch.optim.AdamW(
        fusion.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    global_step = 0
    best_val_mse = float("inf")
    for epoch in range(1, args.epochs + 1):
        fusion.train()
        totals = {
            "loss": 0.0,
            "mse": 0.0,
            "cosine": 0.0,
            "recon": 0.0,
            "keep": 0.0,
            "steps": 0,
        }
        progress = tqdm(loader, desc=f"Temporal epoch {epoch}")
        for frames, teacher, is_static, _labels in progress:
            frames = frames.to(device, non_blocking=True)
            teacher = teacher.to(device, non_blocking=True)
            is_static = is_static.to(device)
            student, tokens, reconstructed, feature0, _ = forward_pair_pipeline(
                edge, cloud, fusion, frames
            )

            if student.shape != teacher.shape:
                raise RuntimeError(
                    f"Student/teacher shapes differ: {student.shape} vs {teacher.shape}"
                )
            sample_mse, sample_cosine = feature_metrics(student, teacher)
            mse = sample_mse.mean()
            cosine = sample_cosine.mean()
            recon = F.mse_loss(reconstructed.float(), tokens.detach().float())
            keep = torch.zeros((), device=device)
            if bool(is_static.any()):
                with torch.no_grad():
                    old_student = old_single_output(edge, cloud, feature0[is_static])
                keep_mse, keep_cosine = feature_metrics(
                    student[is_static], old_student
                )
                keep = keep_mse.mean() + (1.0 - keep_cosine.mean())
            loss = (
                args.lambda_mse * mse
                + args.lambda_cosine * (1.0 - cosine)
                + args.lambda_recon * recon
                + args.lambda_keep * keep
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), args.grad_clip)
            optimizer.step()

            totals["loss"] += float(loss.detach())
            totals["mse"] += float(mse.detach())
            totals["cosine"] += float(cosine.detach())
            totals["recon"] += float(recon.detach())
            totals["keep"] += float(keep.detach())
            totals["steps"] += 1
            global_step += 1
            progress.set_postfix(
                loss=f"{float(loss.detach()):.4f}",
                cos=f"{float(cosine.detach()):.4f}",
                static=int(is_static.sum()),
            )
            if args.max_steps and global_step >= args.max_steps:
                break

        metrics = {
            "epoch": epoch,
            **{
                key: value / max(1, totals["steps"])
                for key, value in totals.items()
                if key != "steps"
            },
            "steps": totals["steps"],
        }
        if val_loader is not None:
            val_metrics = evaluate_fusion(edge, cloud, fusion, val_loader, device)
            metrics["val"] = val_metrics
        history.append(metrics)
        checkpoint = {
            "temporal_fusion_state_dict": fusion.state_dict(),
            "in_channels": in_channels,
            "hidden_channels": fusion.hidden_channels,
            "edge_checkpoint": str(Path(args.edge_checkpoint)),
            "cloud_checkpoint": str(Path(args.cloud_checkpoint)),
            "qwen_path": args.qwen_path,
            "split_layer": args.split_layer,
            "temporal_patch_size": 2,
            "sample_fps": args.sample_fps,
            "loss_weights": {
                "mse": args.lambda_mse,
                "cosine": args.lambda_cosine,
                "recon": args.lambda_recon,
                "keep": args.lambda_keep,
            },
            "history": history,
        }
        torch.save(checkpoint, output_dir / "temporal_pair_latest.pth")
        if val_loader is not None:
            val_mse = metrics["val"]["dynamic"]["mse"]
            if val_mse is not None and val_mse < best_val_mse:
                best_val_mse = val_mse
                torch.save(checkpoint, output_dir / "temporal_pair_best.pth")
        if args.max_steps and global_step >= args.max_steps:
            break

    (output_dir / "train_metrics.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"Saved temporal checkpoint to {output_dir / 'temporal_pair_latest.pth'}")


def evaluate_checkpoint(args):
    device, edge, cloud, _ = build_frozen_pipeline(args)
    checkpoint = torch.load(
        args.temporal_checkpoint, map_location=device, weights_only=False
    )
    fusion = TemporalPairFusion(
        checkpoint["in_channels"], checkpoint.get("hidden_channels", 256)
    ).to(device)
    fusion.load_state_dict(checkpoint["temporal_fusion_state_dict"])
    loader = make_loader(
        args.cache_dir, edge.transform, args.batch_size, shuffle=False
    )
    metrics = evaluate_fusion(edge, cloud, fusion, loader, device)
    output_path = Path(args.eval_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluation to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("cache", "train", "eval", "all"), default="all"
    )
    parser.add_argument("--video_dir")
    parser.add_argument("--video", action="append", default=[])
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--val_cache_dir")
    parser.add_argument("--edge_checkpoint")
    parser.add_argument("--cloud_checkpoint")
    parser.add_argument("--qwen_path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--split_layer", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--sample_fps", type=float, default=2.0)
    parser.add_argument("--max_pairs_per_video", type=int, default=8)
    parser.add_argument("--static_ratio", type=float, default=0.25)
    parser.add_argument("--temporal_hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_mse", type=float, default=1.0)
    parser.add_argument("--lambda_cosine", type=float, default=1.0)
    parser.add_argument("--lambda_recon", type=float, default=0.1)
    parser.add_argument("--lambda_keep", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--output_dir", default="checkpoints/temporal_pair")
    parser.add_argument("--temporal_checkpoint")
    parser.add_argument(
        "--eval_output", default="outputs/temporal_pair/evaluation.json"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if not 0 <= args.static_ratio < 1:
        parser.error("--static_ratio must be in [0, 1)")
    if args.stage in {"train", "eval", "all"} and (
        not args.edge_checkpoint or not args.cloud_checkpoint
    ):
        parser.error("--edge_checkpoint and --cloud_checkpoint are required")
    if args.stage == "eval" and not args.temporal_checkpoint:
        parser.error("--temporal_checkpoint is required for --stage eval")
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.stage in {"cache", "all"}:
        create_teacher_cache(args)
    if args.stage in {"train", "all"}:
        train_temporal_fusion(args)
    if args.stage == "eval":
        evaluate_checkpoint(args)


if __name__ == "__main__":
    main()

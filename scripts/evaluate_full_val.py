#!/usr/bin/env python3
"""Evaluate one SplitOculo checkpoint on the complete validation split.

The rolling trainer intentionally validates on a fixed 2,048-image subset.
This utility computes Qwen targets for every image in ``val/`` in a separate
temporary feature directory, then reuses ``GANTrainer.validate`` so the full
validation metrics use exactly the same generator and metric definitions as
training.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from core.qwen_extractor import QwenFeatureExtractor
from core.utils import set_seed
from scripts.train_gan import GANTrainer, PrecomputedFeatureDataset, collate_fn
from scripts.train_gan_rolling_cache import (
    discover_images,
    precompute_assignments,
    write_feature_metadata,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a SplitOculo checkpoint on all validation images"
    )
    parser.add_argument("--data_dir", default="data/llava_pretrain_558k")
    parser.add_argument(
        "--features_dir", default="data/llava_full_val_features_32b_49x64"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="checkpoint containing the trained generator weights",
    )
    parser.add_argument(
        "--output_dir", default="checkpoints/llava558k_32b_49x64_rolling/full_val"
    )
    parser.add_argument(
        "--qwen_model", default="Qwen/Qwen2.5-VL-32B-Instruct"
    )
    parser.add_argument("--qwen_layer", type=int, default=4)
    parser.add_argument("--qwen_local_files_only", action="store_true")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--feature_batch_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--reuse_features",
        action="store_true",
        help="reuse a complete existing feature directory instead of recomputing",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(42)

    data_dir = Path(args.data_dir).resolve()
    features_dir = Path(args.features_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    val_images = discover_images(data_dir, "val")
    val_dir = features_dir / "val"
    existing_features = sorted(val_dir.glob("*.pt")) if val_dir.exists() else []
    if existing_features and len(existing_features) != len(val_images):
        raise RuntimeError(
            f"Incomplete full-val feature directory: found {len(existing_features):,} "
            f"files, expected {len(val_images):,}; remove it or finish it before retrying"
        )

    started = time.perf_counter()
    if len(existing_features) == len(val_images) and args.reuse_features:
        print(f"Reusing {len(existing_features):,} existing full-val features", flush=True)
    else:
        if existing_features:
            raise RuntimeError(
                "Full-val features already exist. Pass --reuse_features to use them."
            )
        write_feature_metadata(
            features_dir, "val", args.qwen_model, args.qwen_layer
        )
        qwen_transform = transforms.Compose(
            [
                transforms.Resize(
                    args.image_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(args.image_size),
            ]
        )
        print(
            f"Precomputing Qwen targets for all {len(val_images):,} validation images",
            flush=True,
        )
        extractor = QwenFeatureExtractor(
            model_name=args.qwen_model,
            device=args.device,
            extract_layer=args.qwen_layer,
            local_files_only=args.qwen_local_files_only,
            min_pixels=args.image_size * args.image_size,
            max_pixels=args.image_size * args.image_size,
            visual_only=True,
        ).load()
        assignments = list(enumerate(val_images))
        precompute_assignments(
            extractor,
            assignments,
            features_dir,
            "val",
            qwen_transform,
            args.feature_batch_size,
            "Full-val Qwen features",
        )
        del extractor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    saved_args = checkpoint.get("args")
    if not isinstance(saved_args, dict):
        raise RuntimeError("Checkpoint does not contain the training arguments")

    trainer_args = SimpleNamespace(**saved_args)
    trainer_args.device = args.device
    trainer_args.output_dir = str(output_dir)
    trainer_args.data_dir = str(data_dir)
    trainer_args.features_dir = str(features_dir)
    trainer_args.dynamic = False
    trainer_args.multilevel_payload = False
    trainer_args.transmission_tokens = 49
    trainer_args.bottleneck_dim = 64
    trainer_args.payload_levels = "49x64"
    trainer = GANTrainer(trainer_args)
    checkpoint_epoch = trainer.load_checkpoint(checkpoint_path)

    dataset = PrecomputedFeatureDataset(
        features_dir=features_dir,
        images_dir=data_dir,
        split="val",
        image_size=args.image_size,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "collate_fn": collate_fn,
    }
    if args.num_workers > 0:
        loader_kwargs["multiprocessing_context"] = "spawn"
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)

    print(
        f"Validating checkpoint epoch {checkpoint_epoch} on {len(dataset):,} images "
        f"({len(loader):,} batches)",
        flush=True,
    )
    metrics = trainer.validate(loader)
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "num_images": len(dataset),
        "num_batches": len(loader),
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    result_path = output_dir / "full_val_metrics.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

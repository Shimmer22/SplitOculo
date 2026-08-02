#!/usr/bin/env python3
"""Train with a rolling cache of Qwen teacher features.

The full LLaVA-558K image set is too large for a persistent 32B feature
cache.  This runner keeps a fixed number of teacher targets on disk, trains
one epoch over that cache, then replaces half of the cache with images that
have not been seen yet.  Qwen stays loaded in visual-only mode so it does not
need to be reloaded between cache rotations.

The SplitOculo payload is intentionally fixed at 49 tokens x 64 dimensions;
this script never enables the multi-level payload path.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from core.qwen_extractor import QwenFeatureExtractor
from core.utils import set_seed
from scripts.train_gan import GANTrainer, PrecomputedFeatureDataset, collate_fn


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_images(data_dir: Path, split: str) -> list[Path]:
    root = data_dir / split
    if not root.is_dir():
        raise FileNotFoundError(f"Image split does not exist: {root}")
    paths = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No images found in {root}")
    return paths


def feature_path(cache_dir: Path, split: str, slot: int) -> Path:
    return cache_dir / split / f"slot_{slot:05d}.pt"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def write_feature_metadata(cache_dir: Path, split: str, qwen_model: str, layer: int) -> None:
    write_json(
        cache_dir / split / "metadata.json",
        {
            "dataset": "LLaVA-Pretrain",
            "qwen_model": qwen_model,
            "extract_layer": layer,
            "image_size": 224,
            "hidden_size": 1280,
            "target_tokens": 256,
        },
    )


def load_state(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def make_selection(
    shuffled_train: list[Path],
    current_slots: list[str],
    round_index: int,
    cache_size: int,
    seed: int,
) -> tuple[list[str], list[tuple[int, Path]]]:
    """Return the new slot map and only the slots needing Qwen extraction."""
    replace_count = cache_size // 2
    if cache_size % 2:
        raise ValueError("--cache_size must be even so exactly half can rotate")

    if round_index == 0 or not current_slots:
        selected = [str(path.resolve()) for path in shuffled_train[:cache_size]]
        if len(selected) < cache_size:
            raise ValueError(
                f"Need {cache_size:,} training images, found {len(shuffled_train):,}"
            )
        assignments = [
            (slot, Path(path)) for slot, path in enumerate(selected)
        ]
        return selected, assignments

    if len(current_slots) != cache_size:
        raise RuntimeError(
            f"Rolling state has {len(current_slots)} slots, expected {cache_size}"
        )

    rng = random.Random(seed + 1_000_003 * round_index)
    retain_count = cache_size - replace_count
    retained_slots = set(rng.sample(range(cache_size), retain_count))
    replace_slots = [slot for slot in range(cache_size) if slot not in retained_slots]

    start = cache_size + (round_index - 1) * replace_count
    new_paths = shuffled_train[start : start + replace_count]
    if len(new_paths) < replace_count:
        # This only matters after the deterministic pass has covered the full
        # dataset.  Refill from images outside the current cache so later
        # rounds still change the teacher targets instead of becoming no-ops.
        current = set(current_slots)
        fallback = [
            path for path in shuffled_train
            if str(path.resolve()) not in current and path not in new_paths
        ]
        if len(fallback) < replace_count - len(new_paths):
            fallback = [path for path in shuffled_train if path not in new_paths]
        new_paths = new_paths + fallback[: replace_count - len(new_paths)]

    if len(new_paths) != replace_count:
        raise RuntimeError("Could not select enough new images for cache rotation")

    selected = list(current_slots)
    assignments = []
    for slot, path in zip(replace_slots, new_paths):
        selected[slot] = str(path.resolve())
        assignments.append((slot, path))
    return selected, assignments


def precompute_assignments(
    extractor: QwenFeatureExtractor,
    assignments: list[tuple[int, Path]],
    cache_dir: Path,
    split: str,
    image_transform,
    batch_size: int,
    desc: str,
) -> None:
    if not assignments:
        return
    output_dir = cache_dir / split
    output_dir.mkdir(parents=True, exist_ok=True)

    for start in tqdm(range(0, len(assignments), batch_size), desc=desc, unit="batch"):
        batch = assignments[start : start + batch_size]
        images = []
        for _, image_path in batch:
            with Image.open(image_path) as source:
                images.append(image_transform(source.convert("RGB")))

        features = extractor.extract_features_batch(images)
        if len(features) != len(batch):
            raise RuntimeError(
                f"Qwen returned {len(features)} features for {len(batch)} images"
            )
        for (slot, image_path), teacher_features in zip(batch, features):
            destination = feature_path(cache_dir, split, slot)
            temporary = destination.with_suffix(destination.suffix + ".part")
            payload = {
                # ``extract_features_batch`` returns split views into one
                # large batch tensor.  A view would make torch.save serialize
                # the entire batch storage for every slot (about 42 MB per
                # image at batch=64).  Clone first so each cache item owns
                # only its 256x1280 teacher target (~0.65 MB in bfloat16).
                "features": teacher_features.detach().cpu().clone().contiguous(),
                "path": str(image_path.resolve()),
                "num_tokens": int(teacher_features.shape[0]),
                "hidden_size": int(teacher_features.shape[1]),
            }
            torch.save(payload, temporary)
            temporary.replace(destination)
        del images, features


def build_loader(
    cache_dir: Path,
    data_dir: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
):
    dataset = PrecomputedFeatureDataset(
        features_dir=cache_dir,
        images_dir=data_dir,
        split=split,
        image_size=224,
    )
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": True,
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        kwargs["multiprocessing_context"] = "spawn"
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SplitOculo rolling Qwen feature cache training")
    parser.add_argument("--data_dir", default="data/llava_pretrain_558k")
    parser.add_argument(
        "--cache_dir", default="data/llava_feature_cache_32b_49x64"
    )
    parser.add_argument(
        "--output_dir", default="checkpoints/llava558k_32b_49x64_rolling"
    )
    parser.add_argument("--qwen_model", default="Qwen/Qwen2.5-VL-32B-Instruct")
    parser.add_argument("--qwen_layer", type=int, default=4)
    parser.add_argument("--qwen_local_files_only", action="store_true")
    parser.add_argument("--image_size", type=int, default=224)

    parser.add_argument("--cache_size", type=int, default=24576)
    parser.add_argument("--val_size", type=int, default=2048)
    parser.add_argument("--feature_batch_size", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=45)

    # Static training uses a much larger batch because Qwen is not in the
    # training forward path.  49x64 stays fixed regardless of CLI defaults.
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--student_model", default="mobilenetv2_100")
    parser.add_argument("--student_layer", type=int, default=3)
    parser.add_argument("--target_hidden_size", type=int, default=1280)
    parser.add_argument("--projector_hidden", type=int, default=512)
    parser.add_argument("--projector_type", choices=["pooling", "strided"], default="strided")
    parser.add_argument("--initial_upsample", choices=["bilinear", "pixelshuffle"], default="pixelshuffle")
    parser.add_argument("--upsampler_type", choices=["transformer", "mlp", "deconv"], default="transformer")
    parser.add_argument("--transformer_layers", type=int, default=4)
    parser.add_argument("--target_tokens", type=int, default=256)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_d", type=float, default=4e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lambda_mse", type=float, default=10.0)
    parser.add_argument("--lambda_adv", type=float, default=0.1)
    parser.add_argument("--lambda_recon", type=float, default=0.1)
    parser.add_argument("--bottleneck_method", choices=["linear", "mlp", "autoencoder"], default="linear")
    parser.add_argument("--phase", choices=["warmup", "gan"], default="warmup")
    parser.add_argument("--warmup_checkpoint", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cache_size <= 0 or args.cache_size % 2:
        raise ValueError("--cache_size must be a positive even number")
    if args.val_size <= 0 or args.feature_batch_size <= 0 or args.batch_size <= 0:
        raise ValueError("cache, feature, and training batch sizes must be positive")
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")
    if args.target_tokens != 256:
        raise ValueError("This Qwen 224px pipeline requires --target_tokens 256")

    # These are the requested fixed payload settings.  Do not expose a
    # multi-level switch in this runner.
    args.transmission_tokens = 49
    args.bottleneck_dim = 64
    args.multilevel_payload = False
    args.payload_levels = "49x64"
    args.dynamic = False
    args.features_dir = str(Path(args.cache_dir).resolve())
    args.epochs = args.rounds
    args.save_freq = 1

    set_seed(args.seed)
    data_dir = Path(args.data_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_images = discover_images(data_dir, "train")
    val_images = discover_images(data_dir, "val")
    shuffle_rng = random.Random(args.seed)
    shuffled_train = list(train_images)
    shuffle_rng.shuffle(shuffled_train)
    val_rng = random.Random(args.seed + 17)
    val_rng.shuffle(val_images)
    val_paths = [str(path.resolve()) for path in val_images[: args.val_size]]

    state_path = cache_dir / "rolling_state.json"
    state = None if args.no_resume else load_state(state_path)
    if state is not None:
        expected = {
            "cache_size": args.cache_size,
            "val_size": len(val_paths),
            "seed": args.seed,
            "qwen_model": args.qwen_model,
            "qwen_layer": args.qwen_layer,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise RuntimeError(
                    f"Rolling cache state mismatch for {key}: "
                    f"state={state.get(key)!r}, requested={value!r}; use --no_resume"
                )
        if state.get("val_paths") != val_paths:
            raise RuntimeError("Validation image selection changed; use --no_resume")
    else:
        state = {
            "version": 1,
            "cache_size": args.cache_size,
            "val_size": len(val_paths),
            "seed": args.seed,
            "qwen_model": args.qwen_model,
            "qwen_layer": args.qwen_layer,
            "val_paths": val_paths,
            "train_slots": [],
            "completed_rounds": -1,
            "best_cos_sim": 0.0,
        }

    completed_rounds = int(state.get("completed_rounds", -1))
    if completed_rounds >= args.rounds - 1:
        print(
            f"Rolling cache already completed {completed_rounds + 1} rounds; "
            "nothing to do. Use --no_resume for a fresh run."
        )
        return

    print(
        f"Rolling cache: {len(train_images):,} train images, "
        f"{len(val_paths):,} fixed val images, cache={args.cache_size:,}, "
        f"replace={args.cache_size // 2:,} per round"
    )
    print(
        "Fixed payload: transmission_tokens=49, bottleneck_dim=64, "
        "multilevel_payload=False"
    )

    qwen_transform = transforms.Compose(
        [
            transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.image_size),
        ]
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

    write_feature_metadata(cache_dir, "train", args.qwen_model, args.qwen_layer)
    write_feature_metadata(cache_dir, "val", args.qwen_model, args.qwen_layer)

    if not state.get("val_ready"):
        val_assignments = [(slot, Path(path)) for slot, path in enumerate(val_paths)]
        precompute_assignments(
            extractor,
            val_assignments,
            cache_dir,
            "val",
            qwen_transform,
            args.feature_batch_size,
            "Precomputing fixed validation features",
        )
        state["val_ready"] = True
        write_json(state_path, state)

    trainer = GANTrainer(args)
    prefix = "warmup_" if args.phase == "warmup" else "gan_"
    checkpoint_path = output_dir / f"{prefix}latest.pth"
    if completed_rounds >= 0 and checkpoint_path.is_file():
        trainer.load_checkpoint(checkpoint_path)
        print(f"Resumed trainer weights from {checkpoint_path}")
    elif args.phase == "gan":
        if not args.warmup_checkpoint:
            raise ValueError("GAN rolling training requires --warmup_checkpoint")
        trainer.load_checkpoint(args.warmup_checkpoint)

    best_cos_sim = float(state.get("best_cos_sim", 0.0))
    for round_index in range(completed_rounds + 1, args.rounds):
        round_started = time.perf_counter()
        current_slots = state.get("train_slots", [])
        selected_slots, assignments = make_selection(
            shuffled_train,
            current_slots,
            round_index,
            args.cache_size,
            args.seed,
        )
        print(
            f"\n=== Rolling round {round_index + 1}/{args.rounds}: "
            f"extracting {len(assignments):,} new features ===",
            flush=True,
        )
        precompute_assignments(
            extractor,
            assignments,
            cache_dir,
            "train",
            qwen_transform,
            args.feature_batch_size,
            f"Qwen features round {round_index + 1}",
        )
        state["train_slots"] = selected_slots

        train_loader = build_loader(
            cache_dir,
            data_dir,
            "train",
            args.batch_size,
            args.num_workers,
            True,
        )
        val_loader = build_loader(
            cache_dir,
            data_dir,
            "val",
            args.batch_size,
            args.num_workers,
            False,
        )

        if args.phase == "warmup":
            train_metrics = trainer.train_epoch_warmup(train_loader, round_index + 1)
        else:
            train_metrics = trainer.train_epoch_gan(train_loader, round_index + 1)
        val_metrics = trainer.validate(val_loader)
        trainer.scheduler_G.step()
        if args.phase == "gan":
            trainer.scheduler_D.step()

        metrics = {
            **train_metrics,
            **val_metrics,
            "round": round_index + 1,
            "cache_size": args.cache_size,
            "new_features": len(assignments),
            "round_seconds": time.perf_counter() - round_started,
        }
        is_best = metrics["val_cos_sim"] > best_cos_sim
        if is_best:
            best_cos_sim = metrics["val_cos_sim"]
        trainer.save_checkpoint(round_index + 1, metrics, is_best, prefix=prefix)

        state["completed_rounds"] = round_index
        state["best_cos_sim"] = best_cos_sim
        state["last_metrics"] = metrics
        write_json(state_path, state)
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)

        del train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del extractor, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Rolling cache training completed", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Randomly split flat image directory into train/val using symlinks."""
import argparse
import os
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--num_total', type=int, default=50000)
    parser.add_argument('--val_size', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    src = Path(args.src_dir)
    out = Path(args.out_dir)
    out_train = out / 'train'
    out_val = out / 'val'
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    all_images = sorted([f for f in src.iterdir() if f.is_file() and f.suffix.lower() in image_extensions])
    print(f"Found {len(all_images)} images in {src}")

    selected = random.sample(all_images, min(args.num_total, len(all_images)))
    print(f"Selected {len(selected)} images")

    train_imgs = selected[args.val_size:]
    val_imgs = selected[:args.val_size]

    for img in train_imgs:
        os.symlink(str(img.resolve()), str(out_train / img.name))
    print(f"Train: {len(train_imgs)} symlinks -> {out_train}")

    for img in val_imgs:
        os.symlink(str(img.resolve()), str(out_val / img.name))
    print(f"Val:   {len(val_imgs)} symlinks -> {out_val}")


if __name__ == '__main__':
    main()

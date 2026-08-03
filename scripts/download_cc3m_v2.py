#!/usr/bin/env python3
"""Download CC3M tars from hf-mirror, extract jpg images only."""

import subprocess
import shutil
import os
from pathlib import Path

HF_MIRROR = "https://hf-mirror.com"
DATASET = "pixparse/cc3m-wds"
BASE_URL = f"{HF_MIRROR}/datasets/{DATASET}/resolve/main"
IMAGES_DIR = Path("/workspace/SplitOculo/data/images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 50000
IMAGES_PER_TAR = 5046


def count_existing():
    return len(list(IMAGES_DIR.glob("*.jpg")))


def tar_already_done(idx):
    cnt = len(list(IMAGES_DIR.glob(f"cc3m-train-{idx:04d}_*.jpg")))
    return cnt >= IMAGES_PER_TAR


def process_tar(idx):
    fname = f"cc3m-train-{idx:04d}.tar"
    tmp = Path(f"/tmp/{fname}")
    prefix = f"cc3m-train-{idx:04d}_"

    if tar_already_done(idx):
        print(f"[skip] {fname} already done")
        return 0

    # Download
    url = f"{BASE_URL}/{fname}"
    print(f"[dl] {fname} ", end="", flush=True)
    subprocess.run([
        "curl", "-sL", "--connect-timeout", "60", "--max-time", "900",
        "--retry", "5", "--retry-delay", "15",
        "-o", str(tmp), url
    ], check=True)
    mb = tmp.stat().st_size / 1024**2
    print(f"({mb:.0f}MB)", end=" ", flush=True)

    # Extract only jpg to temp dir
    exdir = Path(f"/tmp/cc3m_extract_{idx}")
    exdir.mkdir(exist_ok=True)

    subprocess.run([
        "tar", "xf", str(tmp),
        "-C", str(exdir),
        "--wildcards", "*.jpg",
    ], check=True, capture_output=True)

    # Rename with prefix and move
    count = 0
    for f in exdir.iterdir():
        if f.is_file():
            dest = IMAGES_DIR / f"{prefix}{f.name}"
            if not dest.exists():
                shutil.move(str(f), str(dest))
            else:
                f.unlink()
            count += 1

    # Cleanup
    try:
        shutil.rmtree(exdir)
    except Exception:
        pass
    try:
        tmp.unlink()
    except Exception:
        pass

    print(f"-> {count} imgs")
    return count


def main():
    total = count_existing()
    print(f"[start] {total} images exist, need ~{TARGET}")

    idx = 0
    while total < TARGET and idx < 576:
        if tar_already_done(idx):
            idx += 1
            continue
        try:
            n = process_tar(idx)
            total += n
        except subprocess.CalledProcessError as e:
            print(f"[err] tar {idx}: {e.stderr.decode() if e.stderr else str(e)}")
        except Exception as e:
            print(f"[err] tar {idx}: {e}")
        idx += 1

    final = count_existing()
    gb = sum(f.stat().st_size for f in IMAGES_DIR.glob("*.jpg")) / 1024**3
    print(f"\n=== DONE: {final} images, {gb:.1f}GB ===")


if __name__ == "__main__":
    main()

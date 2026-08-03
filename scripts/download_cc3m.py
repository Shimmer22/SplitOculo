#!/usr/bin/env python3
"""Download a reproducible CC3M image subset from WebDataset tar files.

The default mirror contains roughly 5,046 images per tar.  Archives are
downloaded into ``.part`` files and renamed only after a complete tar has been
validated, so an interrupted run can be resumed safely.  Only image members
are copied to the output directory; the original archive is removed after a
successful extraction unless ``--keep-tars`` is supplied.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_BASE_URL = (
    "https://hf-mirror.com/datasets/pixparse/cc3m-wds/resolve/main"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "images"
DEFAULT_TMP_DIR = Path(__file__).resolve().parents[1] / "data" / "tmp"
IMAGES_PER_TAR = 5_046
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _tar_is_complete(path: Path) -> bool:
    """Read the tar index to distinguish a complete archive from a partial."""
    try:
        with tarfile.open(path, "r:") as archive:
            for _ in archive:
                pass
        return True
    except (OSError, tarfile.TarError):
        return False


def download_tar(index: int, base_url: str, tmp_dir: Path) -> Path:
    filename = f"cc3m-train-{index:04d}.tar"
    url = f"{base_url.rstrip('/')}/{filename}"
    archive_path = tmp_dir / filename
    partial_path = tmp_dir / f"{filename}.part"

    if archive_path.exists() and _tar_is_complete(archive_path):
        print(f"[skip] {filename} is already complete")
        return archive_path

    # Recover an archive left by an older version of this script.  Renaming is
    # recoverable and lets curl continue it instead of silently trusting it.
    if archive_path.exists() and not partial_path.exists():
        archive_path.replace(partial_path)

    print(f"[download] {filename} ({url})")
    subprocess.run(
        [
            "curl",
            "-fL",
            "--continue-at",
            "-",
            "--connect-timeout",
            "30",
            "--max-time",
            "600",
            "--retry",
            "3",
            "--retry-delay",
            "10",
            "-#",
            "-o",
            str(partial_path),
            url,
        ],
        check=True,
    )

    if not _tar_is_complete(partial_path):
        raise RuntimeError(f"downloaded archive is incomplete: {partial_path}")
    partial_path.replace(archive_path)
    size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"[done] {filename} ({size_mb:.0f} MB)")
    return archive_path


def extract_images_only(tar_path: Path, output_dir: Path) -> int:
    """Extract image members without trusting paths embedded in the tar."""
    tar_name = tar_path.stem
    count = 0
    print(f"[extract] {tar_name}")
    with tarfile.open(tar_path, "r:") as archive:
        for member in archive:
            if not member.isfile() or Path(member.name).suffix.lower() not in IMAGE_SUFFIXES:
                continue

            destination = output_dir / f"{tar_name}_{Path(member.name).name}"
            if destination.exists():
                continue

            source = archive.extractfile(member)
            if source is None:
                continue
            partial = destination.with_suffix(destination.suffix + ".part")
            with source, partial.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            partial.replace(destination)
            count += 1

    print(f"[extract] {tar_name} -> {count} images")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory for extracted images (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--tmp-dir",
        "--tmp_dir",
        type=Path,
        default=DEFAULT_TMP_DIR,
        help=f"directory for resumable tar downloads (default: {DEFAULT_TMP_DIR})",
    )
    parser.add_argument("--target-count", "--target_count", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--base-url", "--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--keep-tars",
        action="store_true",
        help="keep downloaded tar files after successful extraction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_count <= 0:
        raise SystemExit("--target-count must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    num_tars = max(1, (args.target_count + IMAGES_PER_TAR - 1) // IMAGES_PER_TAR)
    print(f"Target: ~{args.target_count} images ({num_tars} tar files)")
    print(f"Output: {args.output_dir}")

    downloaded: list[Path] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_tar, index, args.base_url, args.tmp_dir): index
            for index in range(num_tars)
        }
        for future in as_completed(futures):
            try:
                downloaded.append(future.result())
            except Exception as exc:
                print(f"[error] download failed: {exc}")

    print(f"\nDownloaded or resumed {len(downloaded)} tar files; extracting...\n")
    for tar_path in sorted(downloaded):
        try:
            extract_images_only(tar_path, args.output_dir)
            if not args.keep_tars:
                tar_path.unlink()
        except Exception as exc:
            print(f"[error] extraction failed for {tar_path}: {exc}")

    images = [
        path
        for path in args.output_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    size_gib = sum(path.stat().st_size for path in images) / (1024**3)
    print(f"\n=== Done: {len(images)} images ({size_gib:.1f} GiB) ===")


if __name__ == "__main__":
    main()

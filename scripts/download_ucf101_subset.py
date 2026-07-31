"""Download the public 10-class UCF101 subset and build balanced manifests.

The default endpoint uses hf-mirror.com for mainland-China connectivity. The
archive download is resumable and extraction rejects unsafe archive paths.
"""

import argparse
import hashlib
import random
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm


DEFAULT_URL = (
    "https://hf-mirror.com/datasets/sayakpaul/ucf101-subset/resolve/main/"
    "UCF101_subset.tar.gz"
)
VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv", ".webm"}


def download(url, destination, expected_size=0):
    destination = Path(destination)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        append = offset > 0 and response.status_code == 206
        if offset and not append:
            offset = 0
        content_length = int(response.headers.get("content-length", 0))
        total = offset + content_length if content_length else expected_size or None
        with partial.open("ab" if append else "wb") as output, tqdm(
            total=total,
            initial=offset,
            unit="B",
            unit_scale=True,
            desc="UCF101 subset",
        ) as progress:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    output.write(chunk)
                    progress.update(len(chunk))
    actual_size = partial.stat().st_size
    if expected_size and actual_size != expected_size:
        raise RuntimeError(
            f"Downloaded {actual_size} bytes, expected {expected_size}; rerun to resume"
        )
    partial.replace(destination)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    # The Hub file is named .tar.gz but currently contains an uncompressed tar.
    # r:* auto-detects both that form and a genuinely gzip-compressed mirror copy.
    with tarfile.open(archive, "r:*") as bundle:
        bundle.extractall(destination, filter="data")


def balanced_sample(paths, limit, seed):
    by_class = {}
    for path in paths:
        by_class.setdefault(path.parent.name, []).append(path)
    rng = random.Random(seed)
    for values in by_class.values():
        values.sort()
        rng.shuffle(values)
    selected = []
    while by_class and (not limit or len(selected) < limit):
        progressed = False
        for label in sorted(by_class):
            values = by_class[label]
            if values and (not limit or len(selected) < limit):
                selected.append(values.pop())
                progressed = True
        if not progressed:
            break
    return selected


def write_manifests(extracted, output_dir, limits, seed):
    videos = sorted(
        path.resolve()
        for path in Path(extracted).rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    counts = {}
    for split in ("train", "val", "test"):
        candidates = [path for path in videos if split in path.parts]
        selected = balanced_sample(candidates, limits[split], seed)
        manifest = Path(output_dir) / f"ucf101_{split}_{len(selected)}.manifest.txt"
        manifest.write_text(
            "".join(f"{path}\n" for path in selected), encoding="utf-8"
        )
        counts[split] = {
            "available": len(candidates),
            "selected": len(selected),
            "manifest": str(manifest),
        }
    return counts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/ucf101_subset")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--expected_size", type=int, default=171386880)
    parser.add_argument("--sha256")
    parser.add_argument("--train_limit", type=int, default=100)
    parser.add_argument("--val_limit", type=int, default=30)
    parser.add_argument("--test_limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "UCF101_subset.tar.gz"
    if not archive.is_file() or (
        args.expected_size and archive.stat().st_size != args.expected_size
    ):
        download(args.url, archive, args.expected_size)
    archive_sha256 = sha256(archive)
    if args.sha256 and archive_sha256.lower() != args.sha256.lower():
        raise RuntimeError("Archive SHA-256 does not match --sha256")
    print(
        f"Archive: {archive} ({archive.stat().st_size} bytes, "
        f"sha256={archive_sha256})"
    )
    if args.download_only:
        return
    extracted = output_dir / "extracted"
    safe_extract(archive, extracted)
    counts = write_manifests(
        extracted,
        output_dir,
        {
            "train": args.train_limit,
            "val": args.val_limit,
            "test": args.test_limit,
        },
        args.seed,
    )
    for split, values in counts.items():
        print(
            f"{split}: selected {values['selected']}/{values['available']} "
            f"-> {values['manifest']}"
        )


if __name__ == "__main__":
    main()

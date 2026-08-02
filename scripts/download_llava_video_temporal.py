"""Build a disk-bounded temporal teacher cache from LLaVA-Video-178K.

The dataset is split into multi-gigabyte tar shards.  This script processes
one shard at a time:

1. download one shard from the mainland-friendly HF mirror;
2. extract only the selected videos into a staging directory;
3. sample one temporal pair per video and cache Qwen-32B layer-4 targets;
4. append the pair cache to deterministic train/val stores;
5. remove the staging videos and tar before moving on.

The cache contains the two uint8 RGB frames needed by the edge encoder and a
bfloat16 Qwen teacher target.  The training Dataset converts the teacher back
to float32 when it is loaded.  State and manifests are written atomically so a
network or GPU interruption can be resumed with the same command.
"""

import argparse
import concurrent.futures
import gc
import hashlib
import json
import os
import random
import shutil
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image
from tqdm import tqdm

from core.qwen_extractor import QwenFeatureExtractor
from scripts.train_temporal_pair import (
    pil_to_uint8_tensor,
    read_video_pairs,
    square_center_crop,
    uint8_tensor_to_pil,
)


DATASET_ID = "lmms-lab/LLaVA-Video-178K"
DEFAULT_MIRROR = "https://hf-mirror.com"
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def fetch_json(url):
    request = Request(url, headers={"User-Agent": "SplitOculo-LLaVA-Video/1.0"})
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def file_url(mirror, dataset_id, relative_path):
    return (
        f"{mirror.rstrip('/')}/datasets/{dataset_id}/resolve/main/"
        f"{quote(relative_path, safe='/')}?download=true"
    )


def download_with_resume(url, output_path, expected_size=None):
    """Download to ``.part`` and atomically rename after a size check."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if expected_size and output_path.is_file() and output_path.stat().st_size == expected_size:
        print(f"Reusing complete shard: {output_path} ({expected_size / 2**30:.2f} GiB)")
        return

    part_path = output_path.with_name(output_path.name + ".part")
    existing = part_path.stat().st_size if part_path.is_file() else 0
    headers = {"User-Agent": "SplitOculo-LLaVA-Video/1.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    print(
        f"Downloading {url}\n"
        f"  destination={part_path} resume={existing / 2**30:.2f} GiB"
    )
    request = Request(url, headers=headers)
    with urlopen(request, timeout=120) as response:
        status = getattr(response, "status", response.getcode())
        append = bool(existing and status == 206)
        if not append:
            existing = 0
        mode = "ab" if append else "wb"
        written = existing
        last_report = time.monotonic()
        with part_path.open(mode) as handle:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                now = time.monotonic()
                if now - last_report >= 5:
                    print(f"  downloaded={written / 2**30:.2f} GiB", flush=True)
                    last_report = now
    if expected_size and written != expected_size:
        raise RuntimeError(
            f"Incomplete shard {part_path}: got {written} bytes, "
            f"expected {expected_size}"
        )
    os.replace(part_path, output_path)
    print(f"Finished shard: {output_path} ({written / 2**30:.2f} GiB)")


def download_with_parallel_ranges(url, output_path, expected_size, workers):
    """Resume/download a large shard with independent HTTP byte ranges.

    HF mirrors support ``206 Partial Content``.  Keeping one file per range
    makes interruption recoverable and avoids holding extracted videos while
    the download is in progress.  The final concatenation is atomic.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file() and output_path.stat().st_size == expected_size:
        print(f"Reusing complete shard: {output_path} ({expected_size / 2**30:.2f} GiB)")
        return
    if expected_size <= 0:
        raise ValueError(f"Invalid expected shard size: {expected_size}")

    segment_dir = output_path.with_name(output_path.name + ".segments")
    segment_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(int(workers), 16))
    chunk_size = (expected_size + workers - 1) // workers
    ranges = []
    for index in range(workers):
        start = index * chunk_size
        if start >= expected_size:
            break
        end = min(expected_size - 1, start + chunk_size - 1)
        ranges.append((index, start, end))

    def fetch_range(spec):
        index, start, end = spec
        segment = segment_dir / f"range_{index:03d}.part"
        length = end - start + 1
        if segment.is_file() and segment.stat().st_size == length:
            return index, segment, length, True
        temporary = segment.with_name(segment.name + ".tmp")
        headers = {
            "User-Agent": "SplitOculo-LLaVA-Video/1.0",
            "Range": f"bytes={start}-{end}",
        }
        request = Request(url, headers=headers)
        with urlopen(request, timeout=120) as response:
            status = getattr(response, "status", response.getcode())
            if status != 206:
                raise RuntimeError(
                    f"Mirror ignored Range {start}-{end}: HTTP status {status}"
                )
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {start}-{end}/"):
                raise RuntimeError(
                    f"Unexpected Content-Range for {segment}: {content_range!r}"
                )
            written = 0
            with temporary.open("wb") as handle:
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    written += len(chunk)
        if written != length:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"Incomplete byte range {start}-{end}: got {written}, expected {length}"
            )
        os.replace(temporary, segment)
        return index, segment, length, False

    # A previous sequential .part cannot be safely interpreted as a range.
    # It is an exact temporary file in the controlled download directory.
    sequential_part = output_path.with_name(output_path.name + ".part")
    if sequential_part.exists():
        sequential_part.unlink()
    print(
        f"Parallel download: {len(ranges)} HTTP ranges, workers={workers}, "
        f"total={expected_size / 2**30:.2f} GiB"
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ranges)) as pool:
        futures = [pool.submit(fetch_range, spec) for spec in ranges]
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            index, _segment, length, reused = future.result()
            print(
                f"  range {index + 1}/{len(ranges)} ready "
                f"({length / 2**30:.2f} GiB, {'resume' if reused else 'downloaded'})",
                flush=True,
            )

    assembly = output_path.with_name(output_path.name + ".part")
    with assembly.open("wb") as destination:
        for index, _start, _end in ranges:
            segment = segment_dir / f"range_{index:03d}.part"
            with segment.open("rb") as source:
                while True:
                    chunk = source.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    destination.write(chunk)
    if assembly.stat().st_size != expected_size:
        raise RuntimeError(
            f"Assembled shard has {assembly.stat().st_size} bytes, expected {expected_size}"
        )
    os.replace(assembly, output_path)
    for segment in segment_dir.glob("range_*.part"):
        segment.unlink()
    segment_dir.rmdir()
    print(f"Finished parallel shard: {output_path} ({expected_size / 2**30:.2f} GiB)")


def safe_remove_tree(path, expected_parent):
    """Remove only a known per-shard staging directory."""
    path = Path(path).resolve()
    expected_parent = Path(expected_parent).resolve()
    if not path.is_dir() or path.parent != expected_parent:
        raise RuntimeError(f"Refusing to remove unexpected staging path: {path}")
    shutil.rmtree(path)


def safe_remove_file(path, expected_parent):
    # Do not resolve the final component: a pilot may use a symlink to an
    # already downloaded exact shard.  The parent is still checked after
    # resolution, and unlinking removes only that exact directory entry.
    path = Path(path)
    expected_parent = Path(expected_parent).resolve()
    if not (path.is_file() or path.is_symlink()) or path.parent.resolve() != expected_parent:
        raise RuntimeError(f"Refusing to remove unexpected shard path: {path}")
    path.unlink()


def load_shard_catalog(mirror):
    api_url = (
        f"{mirror.rstrip('/')}/api/datasets/{DATASET_ID}/tree/main"
        "?recursive=true&expand=false"
    )
    entries = fetch_json(api_url)
    shards = [
        {
            "path": item["path"],
            "size": int(item["size"]),
            "lfs_oid": item.get("lfs", {}).get("oid"),
        }
        for item in entries
        if item.get("type") == "file" and item.get("path", "").endswith(".tar.gz")
    ]
    if not shards:
        raise RuntimeError(f"No .tar.gz video shards returned by {api_url}")
    return shards


def select_shards(catalog, seed, explicit_paths, max_shards):
    by_path = {item["path"]: item for item in catalog}
    if explicit_paths:
        missing = [path for path in explicit_paths if path not in by_path]
        if missing:
            raise ValueError(f"Unknown shard path(s): {missing}")
        selected = [by_path[path] for path in explicit_paths]
    else:
        groups = {}
        for item in catalog:
            group = item["path"].split("/", 1)[0]
            groups.setdefault(group, []).append(item)
        rng = random.Random(seed)
        for values in groups.values():
            rng.shuffle(values)
        selected = []
        # Round-robin across duration/source groups.  This keeps the first
        # several dozen shards broad instead of exhausting YouTube first.
        while any(groups.values()):
            for group in sorted(groups):
                if groups[group]:
                    selected.append(groups[group].pop())
        if max_shards:
            selected = selected[:max_shards]
    return selected


def stable_source(shard_path, member_name):
    return f"{DATASET_ID}::{shard_path}::{member_name}"


def source_label(member_name, shard_path):
    parts = {part.lower(): part for part in PurePosixPath(member_name).parts}
    lower = {part.lower() for part in PurePosixPath(member_name).parts}
    if "charades" in lower:
        return "Charades"
    if "activitynet" in lower:
        return "ActivityNet"
    if "nextqa" in lower:
        return "NextQA"
    if "youcook2" in lower:
        return "YouCook2"
    if "ego4d" in lower:
        return "Ego4D"
    return Path(shard_path).parent.name


def split_for_source(source, val_ratio):
    digest = hashlib.sha1(source.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return "val" if value < val_ratio else "train"


class CacheWriter:
    def __init__(self, directory, metadata):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / "manifest.json"
        if self.manifest_path.is_file():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        else:
            self.manifest = {**metadata, "samples": []}
            atomic_write_json(self.manifest_path, self.manifest)
        self.existing = {
            (str(item["source"]), bool(item.get("static", False)))
            for item in self.manifest.get("samples", [])
        }
        indices = []
        for path in self.directory.glob("pair_*.pt"):
            try:
                indices.append(int(path.stem.split("_")[-1]))
            except ValueError:
                continue
        self.next_index = max(indices, default=-1) + 1

    def has(self, source, static):
        return (str(source), bool(static)) in self.existing

    def add(self, item, metadata):
        source = str(item["source"])
        static = bool(item.get("static", False))
        key = (source, static)
        if key in self.existing:
            return False
        output_path = self.directory / f"pair_{self.next_index:07d}.pt"
        temporary = output_path.with_name(output_path.name + ".tmp")
        torch.save(item, temporary)
        os.replace(temporary, output_path)
        entry = {"file": output_path.name, **metadata}
        self.manifest.setdefault("samples", []).append(entry)
        self.existing.add(key)
        self.next_index += 1
        atomic_write_json(self.manifest_path, self.manifest)
        return True


def extract_members(tar_path, members, staging):
    staging = Path(staging).resolve()
    staging.mkdir(parents=True, exist_ok=True)
    extracted_by_name = {}
    with tarfile.open(tar_path, "r:gz") as archive:
        # gzip tar files are not randomly seekable.  ``TarFile.extract`` on a
        # shuffled member list would decompress from the beginning for every
        # video, turning one shard into an accidental quadratic operation.
        # Read selected members in their archive offset order, then return
        # them in the deterministic selection order used by the caller.
        ordered_members = sorted(members, key=lambda member: member.offset_data)
        for member in ordered_members:
            if not member.isfile():
                continue
            target = (staging / member.name).resolve()
            if staging not in target.parents:
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            archive.extract(member, staging)
            if target.is_file():
                extracted_by_name[member.name] = (member, target)
    return [extracted_by_name[member.name] for member in members if member.name in extracted_by_name]


def build_pair_sample(video_path, source, label, split, args, extractor, static=False):
    pairs, native_fps, reader = read_video_pairs(
        video_path, pair_count=1, sample_fps=args.sample_fps
    )
    if not pairs:
        return None
    raw0, raw1, source_frame0, source_frame1 = pairs[0]
    frame0 = square_center_crop(raw0, args.image_size)
    frame1 = square_center_crop(raw1 if not static else raw0, args.image_size)
    teacher, grid_thw = extractor.extract_video_features([frame0, frame1])
    grid_thw = grid_thw.detach().cpu()
    expected_side = args.image_size // 14
    grid_values = tuple(int(value) for value in grid_thw.tolist())
    if grid_values != (1, expected_side, expected_side):
        raise RuntimeError(
            f"Expected Qwen native pair grid [1,{expected_side},{expected_side}], "
            f"got {grid_values} for {video_path}"
        )
    return {
        "frames_uint8": torch.stack(
            (pil_to_uint8_tensor(frame0), pil_to_uint8_tensor(frame1))
        ),
        "teacher_features": teacher.to(
            {
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
            }[args.teacher_dtype]
        ),
        "video_grid_thw": grid_thw,
        "static": bool(static),
        "source": source,
        "source_pair": 0,
        "source_frames": [
            int(source_frame0),
            int(source_frame0 if static else source_frame1),
        ],
        "native_fps": float(native_fps),
        "reader": reader,
        "label": label,
        "split": split,
    }


def write_sample(writer, item):
    metadata = {
        "static": bool(item["static"]),
        "source": item["source"],
        "source_pair": item["source_pair"],
        "source_frames": item["source_frames"],
        "label": item["label"],
        "split": item["split"],
        "teacher_shape": list(item["teacher_features"].shape),
        "teacher_dtype": str(item["teacher_features"].dtype).replace("torch.", ""),
        "video_grid_thw": item["video_grid_thw"].tolist(),
    }
    return writer.add(item, metadata)


def process_shard(shard, args, writers, extractor, processed_sources):
    shard_path = shard["path"]
    shard_id = hashlib.sha1(shard_path.encode("utf-8")).hexdigest()[:12]
    download_path = Path(args.download_dir) / Path(shard_path).name
    staging = Path(args.work_dir) / f"shard_{shard_id}"
    download_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)

    free_bytes = shutil.disk_usage(Path(args.cache_dir).resolve()).free
    required_bytes = int(shard["size"]) + int(args.min_free_gb * 2**30)
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"Disk safety stop before {shard_path}: free={free_bytes / 2**30:.2f} GiB, "
            f"shard={shard['size'] / 2**30:.2f} GiB, "
            f"required safety margin={args.min_free_gb:.1f} GiB"
        )
    print(
        f"Disk before shard: free={free_bytes / 2**30:.2f} GiB; "
        f"temporary shard budget={shard['size'] / 2**30:.2f} GiB"
    )

    shard_url = file_url(args.mirror, DATASET_ID, shard_path)
    if args.download_workers <= 1:
        download_with_resume(
            shard_url,
            download_path,
            expected_size=shard["size"],
        )
    else:
        download_with_parallel_ranges(
            shard_url,
            download_path,
            expected_size=shard["size"],
            workers=args.download_workers,
        )
    try:
        with tarfile.open(download_path, "r:gz") as archive:
            members = [
                member
                for member in archive.getmembers()
                if member.isfile()
                and Path(member.name).suffix.lower() in VIDEO_SUFFIXES
            ]
        rng = random.Random(args.seed + int(shard_id, 16))
        rng.shuffle(members)
        remaining = max(0, args.max_videos - len(processed_sources))
        selected = []
        for member in members:
            source = stable_source(shard_path, member.name)
            if source in processed_sources:
                continue
            selected.append(member)
            if len(selected) >= remaining:
                break
        if not selected:
            print(f"No new videos in shard {shard_path}; skipping")
            return 0
        print(
            f"Processing shard {shard_path}: selected {len(selected)} of "
            f"{len(members)} videos; cache total before={len(processed_sources)}"
        )
        if staging.exists():
            safe_remove_tree(staging, args.work_dir)
        extracted = extract_members(download_path, selected, staging)
        dynamic_items = []
        successful = 0
        for member, video_path in tqdm(
            extracted, desc=f"Reading {Path(shard_path).name}", unit="video"
        ):
            source = stable_source(shard_path, member.name)
            if source in processed_sources:
                continue
            split = split_for_source(source, args.val_ratio)
            label = source_label(member.name, shard_path)
            try:
                item = build_pair_sample(
                    video_path, source, label, split, args, extractor, static=False
                )
            except Exception as exc:
                print(f"Skipping unreadable video {member.name}: {exc!r}")
                continue
            if item is None:
                print(f"Skipping video with no pair: {member.name}")
                continue
            if write_sample(writers[split], item):
                processed_sources.add(source)
                dynamic_items.append(item)
                successful += 1

        static_count = int(
            round(successful * args.static_ratio / max(1e-8, 1.0 - args.static_ratio))
        )
        if static_count and dynamic_items:
            static_rng = random.Random(args.seed ^ int(shard_id, 16))
            static_rng.shuffle(dynamic_items)
            for dynamic in tqdm(
                dynamic_items[:static_count],
                desc=f"Caching static pairs {Path(shard_path).name}",
                unit="pair",
            ):
                source = dynamic["source"]
                if writers[dynamic["split"]].has(source, True):
                    continue
                # The uint8 frames are already center-cropped and are enough
                # to construct a static pair without retaining the original
                # video after this shard.
                pil0 = uint8_tensor_to_pil(dynamic["frames_uint8"][0])
                teacher, grid_thw = extractor.extract_video_features([pil0, pil0])
                grid_thw = grid_thw.detach().cpu()
                expected_side = args.image_size // 14
                if tuple(int(value) for value in grid_thw.tolist()) != (
                    1,
                    expected_side,
                    expected_side,
                ):
                    raise RuntimeError(
                        f"Unexpected static Qwen grid {grid_thw.tolist()} for {source}"
                    )
                static_item = {
                    **dynamic,
                    "frames_uint8": torch.stack(
                        (dynamic["frames_uint8"][0], dynamic["frames_uint8"][0])
                    ),
                    "teacher_features": teacher.to(
                        {
                            "float32": torch.float32,
                            "bfloat16": torch.bfloat16,
                            "float16": torch.float16,
                        }[args.teacher_dtype]
                    ),
                    "video_grid_thw": grid_thw.detach().cpu(),
                    "static": True,
                    "source_frames": [dynamic["source_frames"][0]] * 2,
                }
                write_sample(writers[dynamic["split"]], static_item)
        if staging.exists():
            safe_remove_tree(staging, args.work_dir)
        processed_sources.update(
            item["source"] for item in dynamic_items
        )
        return successful
    finally:
        if staging.exists():
            safe_remove_tree(staging, args.work_dir)
        if not args.keep_tar and download_path.exists():
            safe_remove_file(download_path, args.download_dir)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="data/llava_video_temporal_cache")
    parser.add_argument("--download_dir", default="data/llava_video_temporal_downloads")
    parser.add_argument("--work_dir", default="data/llava_video_temporal_work")
    parser.add_argument("--mirror", default=DEFAULT_MIRROR)
    parser.add_argument("--max_videos", type=int, default=12000)
    parser.add_argument("--max_shards", type=int, default=0)
    parser.add_argument("--shard", action="append", default=[])
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--static_ratio", type=float, default=0.10)
    parser.add_argument("--sample_fps", type=float, default=2.0)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--teacher_dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--qwen_path", default="Qwen/Qwen2.5-VL-32B-Instruct")
    parser.add_argument("--split_layer", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--keep_tar", action="store_true")
    parser.add_argument(
        "--min_free_gb",
        type=float,
        default=20.0,
        help="stop before downloading when this much free space would not remain",
    )
    parser.add_argument(
        "--download_workers",
        type=int,
        default=8,
        help="parallel HTTP Range workers for large shards; 1 uses one resumable stream",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.max_videos <= 0:
        raise SystemExit("--max_videos must be positive")
    if not 0 <= args.val_ratio < 1:
        raise SystemExit("--val_ratio must be in [0, 1)")
    if not 0 <= args.static_ratio < 1:
        raise SystemExit("--static_ratio must be in [0, 1)")
    if args.min_free_gb <= 0:
        raise SystemExit("--min_free_gb must be positive")

    metadata = {
        "dataset": DATASET_ID,
        "qwen_path": args.qwen_path,
        "split_layer": args.split_layer,
        "image_size": args.image_size,
        "sample_fps": args.sample_fps,
        "temporal_patch_size": 2,
        "teacher_dtype": args.teacher_dtype,
        "val_ratio": args.val_ratio,
        "static_ratio": args.static_ratio,
    }
    cache_root = Path(args.cache_dir)
    writers = {
        "train": CacheWriter(cache_root / "train", {**metadata, "split": "train"}),
        "val": CacheWriter(cache_root / "val", {**metadata, "split": "val"}),
    }
    state_path = cache_root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {
        "dataset": DATASET_ID,
        "processed_shards": [],
        "processed_videos": 0,
    }
    processed_sources = {
        str(item["source"])
        for writer in writers.values()
        for item in writer.manifest.get("samples", [])
        if not bool(item.get("static", False))
    }
    processed_shards = set(state.get("processed_shards", []))
    catalog = load_shard_catalog(args.mirror)
    shards = select_shards(catalog, args.seed, args.shard, args.max_shards)
    print(
        f"LLaVA-Video catalog: {len(catalog)} shards; selected {len(shards)}; "
        f"existing dynamic videos={len(processed_sources)} target={args.max_videos}"
    )
    if len(processed_sources) >= args.max_videos:
        print("Target already reached; nothing to do")
        return

    extractor = QwenFeatureExtractor(
        model_name=args.qwen_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        extract_layer=args.split_layer,
        local_files_only=args.offline,
        min_pixels=args.image_size * args.image_size,
        max_pixels=args.image_size * args.image_size,
        visual_only=True,
    ).load()
    try:
        for shard in shards:
            if shard["path"] in processed_shards:
                print(f"Already completed shard: {shard['path']}")
                continue
            if len(processed_sources) >= args.max_videos:
                break
            added = process_shard(
                shard, args, writers, extractor, processed_sources
            )
            processed_shards.add(shard["path"])
            state.update(
                {
                    "processed_shards": sorted(processed_shards),
                    "processed_videos": len(processed_sources),
                    "last_shard": shard["path"],
                    "last_added": added,
                }
            )
            atomic_write_json(state_path, state)
            print(
                f"Completed {shard['path']}: added={added}; "
                f"dynamic total={len(processed_sources)}; "
                f"free disk check should remain above the safety margin."
            )
    finally:
        del extractor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(
        f"Finished cache build: dynamic={len(processed_sources)}, "
        f"train_manifest={writers['train'].manifest_path}, "
        f"val_manifest={writers['val'].manifest_path}"
    )


if __name__ == "__main__":
    main()

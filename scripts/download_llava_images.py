#!/usr/bin/env python3
"""Download LLaVA-Pretrain images without downloading the full ZIP locally.

The official LLaVA-Pretrain repository contains about 558K LAION/CC/SBU
image-caption pairs, but its ``images.zip`` is about 27 GB.  This script does
not download that archive in full.  It downloads the small metadata JSON,
reads the remote ZIP central directory with HTTP Range requests, and fetches
only the selected image data ranges.  The default budget covers the current
complete 558K image set; ``--max_gib`` can be used for a smaller subset.

The output is a flat ``train/`` and ``val/`` image directory so it can be fed
directly to ``scripts/train_gan.py --dynamic``.  A deterministic selection
plan and a JSONL manifest make interrupted runs resumable.

Example (safe first run):

    python scripts/download_llava_images.py \
      --output_dir data/llava_pretrain_558k \
      --workers 8 \
      --images_only

The default mirror is used because the regular Hugging Face resolver can be
slow or unavailable in some environments.  Override ``--base_url`` when a
different mirror is preferred.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import struct
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from tqdm import tqdm


DEFAULT_BASE_URL = (
    "https://hf-mirror.com/datasets/liuhaotian/LLaVA-Pretrain/resolve/main"
)
DEFAULT_OUTPUT_DIR = "data/llava_pretrain_558k"
# The current archive contains 25.848 GiB of uncompressed image payload.
# Keep a small margin for a future metadata/index change while still selecting
# all 558,128 records today.
DEFAULT_MAX_GIB = 27.0
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LOCAL_FILE_HEADER = b"PK\x03\x04"
CENTRAL_FILE_HEADER = b"PK\x01\x02"
EOCD_SIGNATURE = b"PK\x05\x06"
ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"


@dataclass(frozen=True)
class ZipEntry:
    name: str
    flag: int
    method: int
    compressed_size: int
    uncompressed_size: int
    crc32: int
    header_offset: int


@dataclass(frozen=True)
class SelectedImage:
    image_id: str
    source_name: str
    output_name: str
    split: str
    url: str
    caption: str
    compressed_size: int
    uncompressed_size: int
    header_offset: int
    method: int
    flag: int
    crc32: int


class DownloadError(RuntimeError):
    pass


def _request_range(url: str, start: int, end: int, retries: int = 5) -> bytes:
    """Fetch one byte range and refuse an accidental full-archive response."""
    if start < 0 or end < start:
        raise ValueError(f"Invalid byte range: {start}-{end}")
    expected = end - start + 1
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                timeout=(30, 180),
                allow_redirects=True,
            )
            if response.status_code != 206:
                raise DownloadError(
                    f"Range request returned HTTP {response.status_code}; "
                    "refusing to download the complete ZIP"
                )
            data = response.content
            if len(data) != expected:
                raise DownloadError(
                    f"Range length mismatch for {start}-{end}: "
                    f"got {len(data)}, expected {expected}"
                )
            content_range = response.headers.get("Content-Range", "")
            if content_range and not content_range.startswith(f"bytes {start}-{end}/"):
                raise DownloadError(
                    f"Unexpected Content-Range {content_range!r} for {start}-{end}"
                )
            return data
        except Exception as exc:  # retry transient mirror/network failures
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    raise DownloadError(f"Failed to fetch range {start}-{end}: {last_error}")


def _request_large_range(
    url: str,
    start: int,
    size: int,
    chunk_size: int = 8 * 1024 * 1024,
    workers: int = 4,
) -> bytes:
    """Fetch a large central directory as parallel, bounded HTTP ranges."""
    if size <= chunk_size:
        return _request_range(url, start, start + size - 1)
    ranges = [
        (offset, min(chunk_size, size - offset))
        for offset in range(0, size, chunk_size)
    ]
    chunks = [None] * len(ranges)
    with ThreadPoolExecutor(max_workers=min(workers, len(ranges))) as executor:
        futures = {
            executor.submit(_request_range, url, start + offset, start + offset + length - 1): index
            for index, (offset, length) in enumerate(ranges)
        }
        for future in as_completed(futures):
            chunks[futures[future]] = future.result()
    return b"".join(chunks)


def _download_file(url: str, destination: Path, retries: int = 5) -> None:
    """Stream a metadata file with a resumable ``.part`` sidecar."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        try:
            with destination.open("rb") as handle:
                json.load(handle)
            return
        except Exception:
            destination.unlink()

    partial = destination.with_suffix(destination.suffix + ".part")
    start = partial.stat().st_size if partial.is_file() else 0
    for attempt in range(retries):
        headers = {"Range": f"bytes={start}-"} if start else {}
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=(30, 180),
                allow_redirects=True,
                stream=True,
            )
            if start and response.status_code != 206:
                # The server did not honor resume.  Restart safely from zero.
                start = 0
                partial.unlink(missing_ok=True)
                response.close()
                continue
            if response.status_code != (206 if start else 200):
                raise DownloadError(
                    f"Metadata request returned HTTP {response.status_code}"
                )
            mode = "ab" if start else "wb"
            with partial.open(mode) as handle, tqdm(
                total=None,
                unit="B",
                unit_scale=True,
                desc=destination.name,
                initial=start,
            ) as progress:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    progress.update(len(chunk))
            partial.replace(destination)
            with destination.open("rb") as handle:
                json.load(handle)
            return
        except Exception:
            if attempt + 1 >= retries:
                raise
            time.sleep(min(2**attempt, 8))
            start = partial.stat().st_size if partial.is_file() else 0
    raise DownloadError(f"Could not download {url}")


def _find_signature(data: bytes, signature: bytes, end: int | None = None) -> int:
    position = data.rfind(signature, 0, end)
    if position < 0:
        raise DownloadError(f"Could not find ZIP signature {signature!r}")
    return position


def _parse_zip64_eocd(data: bytes, position: int) -> tuple[int, int]:
    values = struct.unpack_from("<4sQ2H2I4Q", data, position)
    if values[0] != ZIP64_EOCD_SIGNATURE:
        raise DownloadError("Invalid ZIP64 end-of-central-directory record")
    # values: signature, record_size, version_made, version_needed, disk,
    # disk_with_cd, entries_on_disk, entries_total, cd_size, cd_offset.
    return int(values[8]), int(values[9])


def _parse_zip_index(url: str) -> dict[str, ZipEntry]:
    """Read the ZIP central directory without downloading image payloads."""
    head = requests.head(url, allow_redirects=True, timeout=(30, 60))
    size_header = head.headers.get("Content-Length")
    if not size_header:
        raise DownloadError("Remote ZIP did not provide Content-Length")
    archive_size = int(size_header)

    # The LLaVA archive's ZIP64 central directory is about 53 MiB and sits at
    # the end of the file.  Starting with a 64 MiB tail avoids a slow random
    # read from the Xet backing store while still keeping the temporary
    # metadata range small.
    tail_size = min(64 * 1024 * 1024, archive_size)
    while True:
        tail_start = archive_size - tail_size
        tail = _request_range(url, tail_start, archive_size - 1)
        try:
            eocd_position = _find_signature(tail, EOCD_SIGNATURE)
            eocd_values = struct.unpack_from("<4s4H2LH", tail, eocd_position)
            _, _, _, entries_disk, entries_total, cd_size32, cd_offset32, _ = eocd_values
            needs_zip64 = (
                entries_disk == 0xFFFF
                or entries_total == 0xFFFF
                or cd_size32 == 0xFFFFFFFF
                or cd_offset32 == 0xFFFFFFFF
            )
            if needs_zip64:
                locator_position = tail.rfind(
                    ZIP64_LOCATOR_SIGNATURE, 0, eocd_position
                )
                if locator_position < 0:
                    raise DownloadError("ZIP64 locator was not in the downloaded tail")
                locator = struct.unpack_from(
                    "<4sIQI", tail, locator_position
                )
                zip64_offset = int(locator[2])
                if tail_start <= zip64_offset < archive_size:
                    zip64_position = zip64_offset - tail_start
                    cd_size, cd_offset = _parse_zip64_eocd(tail, zip64_position)
                else:
                    record = _request_range(url, zip64_offset, zip64_offset + 55)
                    cd_size, cd_offset = _parse_zip64_eocd(record, 0)
            else:
                cd_size, cd_offset = int(cd_size32), int(cd_offset32)
            break
        except DownloadError:
            if tail_size >= min(512 * 1024 * 1024, archive_size):
                raise
            tail_size = min(tail_size * 2, 512 * 1024 * 1024, archive_size)

    if cd_size <= 0 or cd_size > 512 * 1024 * 1024:
        raise DownloadError(f"Unexpected central-directory size: {cd_size}")
    central_start = cd_offset - tail_start
    central_end = central_start + cd_size
    if 0 <= central_start and central_end <= len(tail):
        central = tail[central_start:central_end]
    else:
        central = _request_large_range(url, cd_offset, cd_size)
    entries: dict[str, ZipEntry] = {}
    position = 0
    while position + 46 <= len(central):
        if central[position : position + 4] != CENTRAL_FILE_HEADER:
            break
        values = struct.unpack_from("<4s6H3I5H2I", central, position)
        (
            _, _, _, flag, method, _, _, crc32, compressed_size, uncompressed_size,
            name_length, extra_length, comment_length, _, _, _, header_offset,
        ) = values
        name_start = position + 46
        name_end = name_start + name_length
        extra_end = name_end + extra_length
        comment_end = extra_end + comment_length
        if comment_end > len(central):
            raise DownloadError("Central-directory record extends beyond its range")
        raw_name = central[name_start:name_end]
        encoding = "utf-8" if flag & 0x800 else "cp437"
        name = raw_name.decode(encoding, errors="replace").replace("\\", "/")
        extra = central[name_end:extra_end]

        # ZIP64 extra fields replace any 32-bit sentinel values in this order.
        if (
            compressed_size == 0xFFFFFFFF
            or uncompressed_size == 0xFFFFFFFF
            or header_offset == 0xFFFFFFFF
        ):
            extra_position = 0
            while extra_position + 4 <= len(extra):
                field_id, field_size = struct.unpack_from(
                    "<HH", extra, extra_position
                )
                field_start = extra_position + 4
                field_end = field_start + field_size
                if field_end > len(extra):
                    break
                if field_id == 0x0001:
                    field = extra[field_start:field_end]
                    field_position = 0
                    if uncompressed_size == 0xFFFFFFFF:
                        uncompressed_size = struct.unpack_from(
                            "<Q", field, field_position
                        )[0]
                        field_position += 8
                    if compressed_size == 0xFFFFFFFF:
                        compressed_size = struct.unpack_from(
                            "<Q", field, field_position
                        )[0]
                        field_position += 8
                    if header_offset == 0xFFFFFFFF:
                        header_offset = struct.unpack_from(
                            "<Q", field, field_position
                        )[0]
                    break
                extra_position = field_end

        entries[name] = ZipEntry(
            name=name,
            flag=int(flag),
            method=int(method),
            compressed_size=int(compressed_size),
            uncompressed_size=int(uncompressed_size),
            crc32=int(crc32),
            header_offset=int(header_offset),
        )
        position = comment_end
    if not entries:
        raise DownloadError("ZIP central directory contained no entries")
    return entries


def _load_metadata(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return records


def _normalise_name(value: str) -> str:
    return value.replace("\\", "/").lstrip("./")


def _metadata_entries(
    records: Iterable[dict], zip_entries: dict[str, ZipEntry]
) -> list[tuple[dict, ZipEntry]]:
    result = []
    missing = 0
    for record in records:
        image_name = _normalise_name(str(record.get("image", "")))
        entry = zip_entries.get(image_name)
        if entry is None:
            entry = zip_entries.get(f"images/{image_name}")
        if entry is None or Path(entry.name).suffix.lower() not in IMAGE_SUFFIXES:
            missing += 1
            continue
        result.append((record, entry))
    if not result:
        raise DownloadError("No metadata images matched entries in images.zip")
    print(f"Matched {len(result):,} metadata images; skipped {missing:,} missing entries")
    return result


def _target_count(
    eligible: list[tuple[dict, ZipEntry]], max_bytes: int, max_images: int
) -> int:
    average = sum(entry.uncompressed_size for _, entry in eligible) / len(eligible)
    target = max(1, int(max_bytes / max(average, 1)))
    if max_images > 0:
        target = min(target, max_images)
    return min(target, len(eligible))


def _select_images(
    eligible: list[tuple[dict, ZipEntry]],
    max_bytes: int,
    max_images: int,
    seed: int,
    strata: int,
    selection: str,
    val_fraction: float,
) -> list[SelectedImage]:
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    target = _target_count(eligible, max_bytes, max_images)
    if selection == "first":
        candidates = eligible[:target]
    else:
        # Choose contiguous windows from evenly spaced metadata strata.  This
        # keeps semantic coverage broad while allowing the remote ZIP ranges
        # to remain reasonably contiguous and fast to download.
        strata = max(1, min(strata, target, len(eligible)))
        candidates = []
        for stratum in range(strata):
            start = (stratum * len(eligible)) // strata
            end = ((stratum + 1) * len(eligible)) // strata
            count = max(1, round(target * (end - start) / len(eligible)))
            count = min(count, end - start)
            rng = random.Random(seed + stratum * 1009)
            max_start = max(start, end - count)
            window_start = rng.randint(start, max_start) if max_start > start else start
            candidates.extend(eligible[window_start : window_start + count])

    selected: list[SelectedImage] = []
    seen = set()
    used_bytes = 0
    for record, entry in candidates:
        if entry.name in seen:
            continue
        if max_images > 0 and len(selected) >= max_images:
            break
        if used_bytes + entry.uncompressed_size > max_bytes:
            continue
        image_id = str(record.get("id") or Path(entry.name).stem)
        suffix = Path(entry.name).suffix.lower() or ".jpg"
        split_key = hashlib.sha1(f"{seed}:{image_id}".encode()).digest()
        split = "val" if int.from_bytes(split_key[:4], "big") / 2**32 < val_fraction else "train"
        selected.append(
            SelectedImage(
                image_id=image_id,
                source_name=entry.name,
                output_name=f"{image_id}{suffix}",
                split=split,
                url=str(record.get("url", "")),
                caption=str(record.get("blip_caption", "")),
                compressed_size=entry.compressed_size,
                uncompressed_size=entry.uncompressed_size,
                header_offset=entry.header_offset,
                method=entry.method,
                flag=entry.flag,
                crc32=entry.crc32,
            )
        )
        seen.add(entry.name)
        used_bytes += entry.uncompressed_size

    # If size variance left too much budget unused, fill from remaining
    # records.  This path is deterministic and only affects range locality
    # after the broad stratified selection has been made.
    if used_bytes < max_bytes * 0.90 and len(selected) < target:
        for record, entry in eligible:
            if entry.name in seen:
                continue
            if max_images > 0 and len(selected) >= max_images:
                break
            if used_bytes + entry.uncompressed_size > max_bytes:
                continue
            image_id = str(record.get("id") or Path(entry.name).stem)
            suffix = Path(entry.name).suffix.lower() or ".jpg"
            split_key = hashlib.sha1(f"{seed}:{image_id}".encode()).digest()
            split = "val" if int.from_bytes(split_key[:4], "big") / 2**32 < val_fraction else "train"
            selected.append(
                SelectedImage(
                    image_id=image_id,
                    source_name=entry.name,
                    output_name=f"{image_id}{suffix}",
                    split=split,
                    url=str(record.get("url", "")),
                    caption=str(record.get("blip_caption", "")),
                    compressed_size=entry.compressed_size,
                    uncompressed_size=entry.uncompressed_size,
                    header_offset=entry.header_offset,
                    method=entry.method,
                    flag=entry.flag,
                    crc32=entry.crc32,
                )
            )
            seen.add(entry.name)
            used_bytes += entry.uncompressed_size
    return selected


def _selected_from_plan(path: Path) -> list[SelectedImage]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [SelectedImage(**item) for item in payload["selected"]]


def _write_plan(path: Path, selected: list[SelectedImage], args) -> None:
    path.write_text(
        json.dumps(
            {
                "base_url": args.base_url,
                "zip_url": args.zip_url,
                "seed": args.seed,
                "selection": args.selection,
                "max_gib": args.max_gib,
                "max_images": args.max_images,
                "val_fraction": args.val_fraction,
                "selected": [item.__dict__ for item in selected],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _group_ranges(selected: list[SelectedImage], range_bytes: int) -> list[list[SelectedImage]]:
    ordered = sorted(selected, key=lambda item: item.header_offset)
    groups: list[list[SelectedImage]] = []
    current: list[SelectedImage] = []
    start = None
    for item in ordered:
        if not current:
            current = [item]
            start = item.header_offset
            continue
        if item.header_offset - int(start) <= range_bytes:
            current.append(item)
        else:
            groups.append(current)
            current = [item]
            start = item.header_offset
    if current:
        groups.append(current)
    return groups


def _decode_zip_payload(item: SelectedImage, compressed: bytes) -> bytes:
    if item.flag & 0x1:
        raise DownloadError(f"Encrypted ZIP entry is not supported: {item.source_name}")
    if item.method == 0:
        payload = compressed
    elif item.method == 8:
        payload = zlib.decompress(compressed, -15)
    else:
        raise DownloadError(
            f"Unsupported ZIP compression method {item.method} for {item.source_name}"
        )
    if len(payload) != item.uncompressed_size:
        raise DownloadError(
            f"Size mismatch for {item.source_name}: got {len(payload)}, "
            f"expected {item.uncompressed_size}"
        )
    if (zlib.crc32(payload) & 0xFFFFFFFF) != item.crc32:
        raise DownloadError(f"CRC mismatch for {item.source_name}")
    return payload


def _fetch_group(url: str, group: list[SelectedImage], range_bytes: int):
    group = sorted(group, key=lambda item: item.header_offset)
    start = group[0].header_offset
    # A local ZIP header has at most 30 bytes plus 64 KiB name/extra fields.
    end = max(
        item.header_offset + 30 + 65535 + item.compressed_size - 1
        for item in group
    )
    if end - start + 1 > range_bytes and len(group) > 1:
        # The caller normally groups by this bound; this guard avoids an
        # unexpectedly large request if a single unusual entry has a huge gap.
        split_at = max(1, len(group) // 2)
        return _fetch_group(url, group[:split_at], range_bytes) + _fetch_group(
            url, group[split_at:], range_bytes
        )
    blob = _request_range(url, start, end)
    outputs = []
    for item in group:
        relative = item.header_offset - start
        if blob[relative : relative + 4] != LOCAL_FILE_HEADER:
            raise DownloadError(f"Missing local ZIP header for {item.source_name}")
        values = struct.unpack_from("<4s5H3I2H", blob, relative)
        _, _, local_flag, local_method, _, _, _, _, _, name_length, extra_length = values
        data_start = relative + 30 + name_length + extra_length
        data_end = data_start + item.compressed_size
        if data_end > len(blob):
            raise DownloadError(
                f"Range did not include complete entry {item.source_name}; "
                "try a smaller --range_mib"
            )
        if local_flag & 0x1 or local_method != item.method:
            raise DownloadError(f"Local/central ZIP header mismatch for {item.source_name}")
        outputs.append((item, _decode_zip_payload(item, blob[data_start:data_end])))
    return outputs


def _write_image(output_dir: Path, item: SelectedImage, payload: bytes) -> Path:
    split_dir = output_dir / item.split
    split_dir.mkdir(parents=True, exist_ok=True)
    destination = split_dir / item.output_name
    if destination.is_file() and destination.stat().st_size == len(payload):
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.write_bytes(payload)
    partial.replace(destination)
    return destination


def _image_path(output_dir: Path, item: SelectedImage) -> Path:
    return output_dir / item.split / item.output_name


def _image_is_complete(output_dir: Path, item: SelectedImage) -> bool:
    destination = _image_path(output_dir, item)
    return destination.is_file() and destination.stat().st_size == item.uncompressed_size


def _verify_complete(output_dir: Path, selected: list[SelectedImage]) -> int:
    """Verify every planned image before any control metadata is removed."""
    missing = []
    actual_bytes = 0
    for item in selected:
        destination = _image_path(output_dir, item)
        if not _image_is_complete(output_dir, item):
            missing.append(str(destination.relative_to(output_dir)))
            continue
        actual_bytes += destination.stat().st_size
    if missing:
        preview = ", ".join(missing[:5])
        more = " ..." if len(missing) > 5 else ""
        raise DownloadError(
            f"Download is incomplete: {len(missing):,} images are missing or have "
            f"the wrong size ({preview}{more})"
        )
    return actual_bytes


def _load_completed(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            completed.add(json.loads(line)["source_name"])
        except Exception:
            continue
    return completed


def _cleanup_control_files(output_dir: Path) -> list[str]:
    """Remove downloader metadata, leaving only train/val image files."""
    names = {
        "blip_laion_cc_sbu_558k_meta.json",
        "selection_plan.json",
        "download_manifest.jsonl",
        "download_summary.json",
    }
    removed = []
    for path in output_dir.iterdir():
        if path.is_file() and (path.name in names or path.name.endswith(".part")):
            path.unlink()
            removed.append(str(path.relative_to(output_dir)))
    for split in ("train", "val"):
        split_dir = output_dir / split
        if not split_dir.is_dir():
            continue
        for path in split_dir.glob("*.part"):
            if path.is_file():
                path.unlink()
                removed.append(str(path.relative_to(output_dir)))
    return removed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download LLaVA-Pretrain images without downloading images.zip"
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--metadata_url", default=None)
    parser.add_argument("--zip_url", default=None)
    parser.add_argument(
        "--max_gib",
        type=float,
        default=DEFAULT_MAX_GIB,
        help="maximum uncompressed image bytes; default covers the current full 558K set",
    )
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--val_fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection",
        choices=("stratified", "first"),
        default="stratified",
        help="stratified contiguous windows preserve broad coverage and range locality",
    )
    parser.add_argument("--strata", type=int, default=64)
    parser.add_argument("--range_mib", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--reselect",
        action="store_true",
        help="discard the previous deterministic selection plan and create a new one",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="build/print the selection plan but do not download image ranges",
    )
    parser.add_argument(
        "--images_only",
        action="store_true",
        help="after a verified download, remove downloader metadata and keep only images",
    )
    args = parser.parse_args()
    if args.max_gib <= 0:
        parser.error("--max_gib must be positive")
    if args.max_images < 0:
        parser.error("--max_images must be non-negative")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.range_mib <= 0:
        parser.error("--range_mib must be positive")
    if args.dry_run and args.images_only:
        parser.error("--images_only cannot be combined with --dry_run")
    args.base_url = args.base_url.rstrip("/")
    args.metadata_url = args.metadata_url or f"{args.base_url}/blip_laion_cc_sbu_558k_meta.json"
    args.zip_url = args.zip_url or f"{args.base_url}/images.zip"
    return args


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = int(args.max_gib * 1024**3)
    metadata_path = output_dir / "blip_laion_cc_sbu_558k_meta.json"
    plan_path = output_dir / "selection_plan.json"
    manifest_path = output_dir / "download_manifest.jsonl"

    if not metadata_path.is_file():
        print(f"Downloading metadata: {args.metadata_url}")
        _download_file(args.metadata_url, metadata_path)
    records = _load_metadata(metadata_path)
    print(f"Loaded {len(records):,} LLaVA metadata records")

    if plan_path.is_file() and not args.reselect:
        selected = _selected_from_plan(plan_path)
        print(f"Resuming existing selection plan with {len(selected):,} images")
    else:
        print("Reading remote ZIP central directory (no image payload download yet)...")
        zip_entries = _parse_zip_index(args.zip_url)
        eligible = _metadata_entries(records, zip_entries)
        selected = _select_images(
            eligible=eligible,
            max_bytes=max_bytes,
            max_images=args.max_images,
            seed=args.seed,
            strata=args.strata,
            selection=args.selection,
            val_fraction=args.val_fraction,
        )
        if not selected:
            raise DownloadError("Selection produced zero images")
        _write_plan(plan_path, selected, args)

    selected_bytes = sum(item.uncompressed_size for item in selected)
    selected_compressed = sum(item.compressed_size for item in selected)
    print(
        f"Selected {len(selected):,} images: "
        f"{selected_bytes / 1024**3:.3f} GiB uncompressed, "
        f"{selected_compressed / 1024**3:.3f} GiB compressed payload"
    )
    print(
        f"Splits: train={sum(item.split == 'train' for item in selected):,}, "
        f"val={sum(item.split == 'val' for item in selected):,}"
    )
    if args.dry_run:
        return

    completed = _load_completed(manifest_path)
    pending = [
        item
        for item in selected
        if item.source_name not in completed or not _image_is_complete(output_dir, item)
    ]
    print(f"Already complete: {len(selected) - len(pending):,}; pending: {len(pending):,}")
    groups = _group_ranges(pending, args.range_mib * 1024**2)
    print(f"Remote range requests: {len(groups):,} (workers={args.workers})")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _fetch_group, args.zip_url, group, args.range_mib * 1024**2
            ): group
            for group in groups
        }
        with tqdm(total=len(futures), desc="Extracting LLaVA ranges", unit="range") as progress:
            for future in as_completed(futures):
                group = futures[future]
                outputs = future.result()
                with manifest_path.open("a", encoding="utf-8") as manifest:
                    for item, payload in outputs:
                        destination = _write_image(output_dir, item, payload)
                        manifest.write(
                            json.dumps(
                                {
                                    "id": item.image_id,
                                    "source_name": item.source_name,
                                    "path": str(destination.relative_to(output_dir)),
                                    "split": item.split,
                                    "bytes": len(payload),
                                    "url": item.url,
                                    "blip_caption": item.caption,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                progress.update(1)

    actual_bytes = _verify_complete(output_dir, selected)
    summary = {
        "dataset": "liuhaotian/LLaVA-Pretrain",
        "metadata_url": args.metadata_url,
        "zip_url": args.zip_url,
        "selected_images": len(selected),
        "train_images": sum(item.split == "train" for item in selected),
        "val_images": sum(item.split == "val" for item in selected),
        "planned_uncompressed_bytes": selected_bytes,
        "actual_image_bytes": actual_bytes,
        "max_image_bytes": max_bytes,
        "seed": args.seed,
        "selection": args.selection,
        "val_fraction": args.val_fraction,
    }
    (output_dir / "download_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.images_only:
        removed = _cleanup_control_files(output_dir)
        print(
            f"Images-only cleanup complete: removed {len(removed)} control files; "
            f"kept {summary['selected_images']:,} images under {output_dir}/train and {output_dir}/val"
        )


if __name__ == "__main__":
    main()

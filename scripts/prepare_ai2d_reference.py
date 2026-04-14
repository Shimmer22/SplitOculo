"""Prepare AI2D evaluation JSONL for EXP-002R reference protocol.

Outputs schema per line:
{
  "sample_id": "ai2d-xxx",
  "image": "/abs/path/to/image.png",
  "prompt": "<mcq prompt>",
  "question": "...",
  "options": {"A": "...", "B": "...", ...},
  "label": "B",
  "reference": "B",
  "subsets": ["ai2d", "vqa_mcq"]
}

Input modes:
1) HuggingFace datasets (recommended):
   --hf-dataset lmms-lab/ai2d --hf-split test
2) Local json/jsonl file:
   --input path/to/ai2d.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _options_from_any(options_value: Any) -> Dict[str, str]:
    if isinstance(options_value, Mapping):
        out: Dict[str, str] = {}
        for key, value in options_value.items():
            letter = normalize_text(key).upper()[:1]
            if letter and letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                out[letter] = normalize_text(value)
        return out

    if isinstance(options_value, list):
        out = {}
        for idx, value in enumerate(options_value):
            letter = chr(ord("A") + idx)
            out[letter] = normalize_text(value)
        return out

    return {}


def _normalize_label(raw_label: Any, options: Mapping[str, str]) -> str:
    label = normalize_text(raw_label)
    if not label:
        return ""

    upper = label.upper()
    if upper in options:
        return upper

    if label.isdigit():
        idx = int(label)
        if idx < len(options):
            return chr(ord("A") + idx)

    for letter, text in options.items():
        if normalize_text(text).lower() == label.lower():
            return letter

    return upper[:1] if upper else ""


def build_prompt(question: str, options: Mapping[str, str]) -> str:
    option_lines = [f"{k}. {v}" for k, v in sorted(options.items())]
    joined = "\n".join(option_lines)
    return (
        "Answer the following multiple-choice question based on the image. "
        "Return only one option letter (A/B/C/D/...).\n"
        f"Question: {question}\n"
        f"Options:\n{joined}\n"
        "Answer:"
    )


def iter_local_samples(path: Path) -> Iterable[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                yield item
            return
        if isinstance(data, dict) and isinstance(data.get("samples"), list):
            for item in data["samples"]:
                yield item
            return

    raise ValueError(f"Unsupported local input: {path}")


def maybe_save_hf_image(image_obj: Any, out_dir: Path, sample_id: str) -> Optional[str]:
    if image_obj is None:
        return None
    if isinstance(image_obj, str):
        p = Path(image_obj)
        return str(p.resolve()) if p.exists() else image_obj

    if hasattr(image_obj, "save"):
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"{sample_id}.png"
        image_obj.save(target)
        return str(target.resolve())

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AI2D jsonl for EXP-002R")
    parser.add_argument("--output", type=str, required=True, help="Output jsonl path")
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--hf-dataset", type=str, default=None, help="HF dataset name, e.g. lmms-lab/ai2d")
    parser.add_argument("--hf-split", type=str, default="test")
    parser.add_argument("--hf-cache-dir", type=str, default=None)

    parser.add_argument("--input", type=str, default=None, help="Local ai2d json/jsonl")
    parser.add_argument("--images-dir", type=str, default=None, help="Base images dir for local input")
    parser.add_argument(
        "--save-hf-images-dir",
        type=str,
        default="data/ai2d/images",
        help="Where to materialize HF image objects",
    )
    args = parser.parse_args()

    if not args.hf_dataset and not args.input:
        raise ValueError("Provide one input source: --hf-dataset or --input")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    if args.hf_dataset:
        try:
            from datasets import load_dataset
        except Exception as exc:
            raise RuntimeError(
                "datasets package is required for --hf-dataset mode. Install with: pip install datasets"
            ) from exc

        ds = load_dataset(args.hf_dataset, split=args.hf_split, cache_dir=args.hf_cache_dir)
        save_hf_dir = Path(args.save_hf_images_dir).resolve()
        for idx, sample in enumerate(ds):
            question = normalize_text(sample.get("question") or sample.get("query"))
            options = _options_from_any(sample.get("options") or sample.get("choices"))
            label = _normalize_label(sample.get("answer") or sample.get("label"), options)
            if not question or not options or not label:
                continue

            sample_id = normalize_text(sample.get("id") or sample.get("question_id") or f"ai2d-{idx:06d}")
            image_path = maybe_save_hf_image(sample.get("image"), save_hf_dir, sample_id)
            if not image_path:
                continue

            rows.append(
                {
                    "sample_id": sample_id,
                    "image": image_path,
                    "question": question,
                    "prompt": build_prompt(question, options),
                    "options": options,
                    "label": label,
                    "reference": label,
                    "subsets": ["ai2d", "vqa_mcq"],
                }
            )
            if args.max_samples is not None and len(rows) >= args.max_samples:
                break
    else:
        input_path = Path(args.input).resolve()
        image_base = Path(args.images_dir).resolve() if args.images_dir else input_path.parent
        for idx, sample in enumerate(iter_local_samples(input_path)):
            question = normalize_text(sample.get("question") or sample.get("query"))
            options = _options_from_any(sample.get("options") or sample.get("choices"))
            label = _normalize_label(sample.get("answer") or sample.get("label"), options)
            if not question or not options or not label:
                continue

            sample_id = normalize_text(sample.get("sample_id") or sample.get("id") or f"ai2d-{idx:06d}")
            image_value = sample.get("image") or sample.get("image_path")
            if not image_value:
                continue
            image_path = Path(str(image_value))
            if not image_path.is_absolute():
                image_path = (image_base / image_path).resolve()
            if not image_path.exists():
                continue

            rows.append(
                {
                    "sample_id": sample_id,
                    "image": str(image_path),
                    "question": question,
                    "prompt": sample.get("prompt") or build_prompt(question, options),
                    "options": options,
                    "label": label,
                    "reference": label,
                    "subsets": sample.get("subsets") or ["ai2d", "vqa_mcq"],
                }
            )
            if args.max_samples is not None and len(rows) >= args.max_samples:
                break

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()

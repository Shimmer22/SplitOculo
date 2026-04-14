"""Answer agreement baseline evaluator for SplitOculo edge-cloud pipeline.

Dataset format (JSON or JSONL):
  {
    "sample_id": "optional-id",
    "image": "relative/or/absolute/path.jpg",
    "prompt": "optional question",
    "reference": "full-model answer or target answer",
    "subsets": ["ocr", "digits"]
  }

Usage example:
  python scripts/eval_answer_agreement.py \
      --dataset data/eval_samples.jsonl \
      --checkpoint checkpoints/edge_weights.pth \
      --server http://127.0.0.1:8080 \
      --output-dir docs/experiments/EXP-002-answer-agreement-baseline/artifacts
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import sys

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))


_NORMALIZE_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)


@dataclass
class BudgetRun:
    name: str
    checkpoint: str


def normalize_answer(text: str) -> str:
    """Normalize text for coarse answer agreement check."""
    if text is None:
        return ""
    normalized = text.strip().lower()
    normalized = _NORMALIZE_PATTERN.sub(" ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def is_answer_agreement(prediction: str, reference: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(reference)


def summarize_samples(rows: List[Dict]) -> Dict:
    if not rows:
        return {
            "num_samples": 0,
            "agreement_rate": 0.0,
            "avg_payload_bytes": 0.0,
            "avg_payload_bits": 0.0,
            "subsets": {},
        }

    agreed = sum(1 for row in rows if is_answer_agreement(row.get("prediction", ""), row.get("reference", "")))
    total_payload_bytes = sum(float(row.get("payload_bytes", 0.0)) for row in rows)

    subset_rows: Dict[str, List[Dict]] = {}
    for row in rows:
        for subset in row.get("subsets", []):
            subset_rows.setdefault(str(subset), []).append(row)

    subset_summary = {}
    for subset, subset_items in sorted(subset_rows.items()):
        subset_agreed = sum(
            1
            for item in subset_items
            if is_answer_agreement(item.get("prediction", ""), item.get("reference", ""))
        )
        subset_summary[subset] = {
            "num_samples": len(subset_items),
            "agreement_rate": subset_agreed / len(subset_items),
        }

    return {
        "num_samples": len(rows),
        "agreement_rate": agreed / len(rows),
        "avg_payload_bytes": total_payload_bytes / len(rows),
        "avg_payload_bits": (total_payload_bytes * 8.0) / len(rows),
        "subsets": subset_summary,
    }


def parse_budget_run(raw_value: str) -> BudgetRun:
    if "=" not in raw_value:
        raise ValueError(f"Invalid --budget value '{raw_value}', expected name=checkpoint_path")
    name, checkpoint = raw_value.split("=", 1)
    name = name.strip()
    checkpoint = checkpoint.strip()
    if not name or not checkpoint:
        raise ValueError(f"Invalid --budget value '{raw_value}', empty name or checkpoint")
    return BudgetRun(name=name, checkpoint=checkpoint)


def load_dataset(dataset_path: Path, max_samples: int | None = None) -> List[Dict]:
    suffix = dataset_path.suffix.lower()
    items: List[Dict] = []

    if suffix == ".jsonl":
        with dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
    elif suffix == ".json":
        with dataset_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, list):
                items = loaded
            elif isinstance(loaded, dict) and isinstance(loaded.get("samples"), list):
                items = loaded["samples"]
            else:
                raise ValueError("JSON dataset must be a list or an object with 'samples' list")
    else:
        raise ValueError(f"Unsupported dataset suffix '{suffix}', expected .json or .jsonl")

    if max_samples is not None:
        return items[:max_samples]
    return items


def resolve_subsets(item: Dict) -> List[str]:
    subsets = item.get("subsets")
    if subsets is None:
        subset = item.get("subset")
        return [str(subset)] if subset else []
    if isinstance(subsets, str):
        return [subsets]
    if isinstance(subsets, list):
        return [str(x) for x in subsets]
    return []


def run_budget(
    budget: BudgetRun,
    dataset: List[Dict],
    dataset_dir: Path,
    server: str,
    default_prompt: str,
    device: str,
    timeout: int,
) -> Tuple[List[Dict], List[Dict]]:
    # Local import keeps utility functions testable without torch dependency.
    from scripts.edge_client import EdgeEncoder

    encoder = EdgeEncoder(checkpoint_path=budget.checkpoint, device=device)

    rows: List[Dict] = []
    failures: List[Dict] = []

    for idx, item in enumerate(dataset):
        sample_id = item.get("sample_id") or f"sample-{idx:04d}"
        image_value = item.get("image") or item.get("image_path")
        if not image_value:
            failures.append({"sample_id": sample_id, "error": "missing image/image_path"})
            continue

        image_path = Path(image_value)
        if not image_path.is_absolute():
            image_path = (dataset_dir / image_path).resolve()
        if not image_path.exists():
            failures.append({"sample_id": sample_id, "error": f"image not found: {image_path}"})
            continue

        reference = item.get("reference") or item.get("reference_answer") or item.get("target_answer")
        if reference is None:
            failures.append({"sample_id": sample_id, "error": "missing reference/reference_answer/target_answer"})
            continue

        prompt = item.get("prompt") or default_prompt

        try:
            payload, stats = encoder.encode_to_payload(str(image_path))
            payload["prompt"] = prompt

            response = requests.post(
                f"{server.rstrip('/')}/infer",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            prediction = str(data.get("response", ""))
        except Exception as exc:
            failures.append({"sample_id": sample_id, "error": str(exc)})
            continue

        rows.append(
            {
                "budget": budget.name,
                "sample_id": sample_id,
                "image": str(image_path),
                "prompt": prompt,
                "reference": str(reference),
                "prediction": prediction,
                "agreement": is_answer_agreement(prediction, str(reference)),
                "payload_bytes": int(stats.get("payload_bytes", 0)),
                "encode_time_ms": float(stats.get("encode_time_ms", 0.0)),
                "feature_shape": stats.get("feature_shape"),
                "subsets": resolve_subsets(item),
            }
        )

    return rows, failures


def to_markdown_table(summary_by_budget: Dict[str, Dict]) -> str:
    lines = [
        "| budget | samples | agreement_rate | avg_payload_bytes | avg_payload_bits |",
        "|---|---:|---:|---:|---:|",
    ]
    for budget_name in sorted(summary_by_budget.keys()):
        item = summary_by_budget[budget_name]
        lines.append(
            f"| {budget_name} | {item['num_samples']} | {item['agreement_rate']:.4f} | "
            f"{item['avg_payload_bytes']:.2f} | {item['avg_payload_bits']:.2f} |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate answer agreement under transmission budget")
    parser.add_argument("--dataset", type=str, required=True, help="Path to .json/.jsonl dataset")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8080", help="Cloud server URL")
    parser.add_argument("--checkpoint", type=str, default=None, help="Single edge checkpoint path")
    parser.add_argument(
        "--budget",
        action="append",
        default=[],
        help="Repeatable budget setting in name=checkpoint_path format",
    )
    parser.add_argument("--default-prompt", type=str, default="Describe this image.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/exp002_answer_agreement")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset).resolve()
    dataset = load_dataset(dataset_path, max_samples=args.max_samples)
    if not dataset:
        raise ValueError("Dataset is empty")

    budgets: List[BudgetRun] = []
    for raw in args.budget:
        budgets.append(parse_budget_run(raw))

    if args.checkpoint:
        budgets.append(BudgetRun(name="default", checkpoint=args.checkpoint))

    if not budgets:
        raise ValueError("Provide at least one budget via --checkpoint or --budget name=checkpoint")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []
    all_failures: List[Dict] = []
    summary_by_budget: Dict[str, Dict] = {}

    for budget in budgets:
        rows, failures = run_budget(
            budget=budget,
            dataset=dataset,
            dataset_dir=dataset_path.parent,
            server=args.server,
            default_prompt=args.default_prompt,
            device=args.device,
            timeout=args.timeout,
        )
        all_rows.extend(rows)
        all_failures.extend(
            [{"budget": budget.name, **failure} for failure in failures]
        )
        summary_by_budget[budget.name] = summarize_samples(rows)

    summary_payload = {
        "dataset": str(dataset_path),
        "server": args.server,
        "num_requested_samples": len(dataset),
        "num_rows": len(all_rows),
        "num_failures": len(all_failures),
        "summary_by_budget": summary_by_budget,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "rows.json").write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "failures.json").write_text(
        json.dumps(all_failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    markdown = ["# EXP-002 Answer Agreement Summary", "", to_markdown_table(summary_by_budget), ""]
    if all_failures:
        markdown.append("## Failures")
        markdown.append("")
        for item in all_failures[:50]:
            markdown.append(f"- [{item.get('budget', 'unknown')}] {item.get('sample_id', 'unknown')}: {item.get('error', 'unknown error')}")
    (output_dir / "summary.md").write_text("\n".join(markdown), encoding="utf-8")

    print(f"Saved summary to {output_dir / 'summary.json'}")
    print(f"Saved per-sample rows to {output_dir / 'rows.json'}")
    print(f"Saved failures to {output_dir / 'failures.json'}")
    print(f"Saved markdown report to {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()

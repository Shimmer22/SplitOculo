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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import sys

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))


_NORMALIZE_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)
_CHOICE_PATTERN_1 = re.compile(r"\(([A-Da-d])\)")
_CHOICE_PATTERN_2 = re.compile(r"\b(?:answer|option|choice|答案|选项)\s*[:：]?\s*([A-Da-d])\b", flags=re.IGNORECASE)


@dataclass
class BudgetRun:
    name: str
    checkpoint: str


def _coerce_options(options: Any) -> Dict[str, str]:
    if options is None:
        return {}
    if isinstance(options, Mapping):
        normalized: Dict[str, str] = {}
        for key, value in options.items():
            letter = str(key).strip().upper()
            if letter and letter[0] in ("A", "B", "C", "D"):
                normalized[letter[0]] = str(value)
        return normalized
    if isinstance(options, list):
        normalized = {}
        for idx, value in enumerate(options):
            if idx > 25:
                break
            letter = chr(ord("A") + idx)
            normalized[letter] = str(value)
        return normalized
    return {}


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


def extract_mcq_answer(raw_output: str, options: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    text = str(raw_output or "")
    options_map = _coerce_options(options)

    for pattern in (_CHOICE_PATTERN_1, _CHOICE_PATTERN_2):
        match = pattern.search(text)
        if match:
            choice = match.group(1).upper()
            answer = options_map.get(choice, choice)
            return {"choice": choice, "answer": str(answer), "raw": text}

    stripped = text.strip().upper()
    if stripped in ("A", "B", "C", "D"):
        answer = options_map.get(stripped, stripped)
        return {"choice": stripped, "answer": str(answer), "raw": text}

    normalized_text = normalize_answer(text)
    for choice, option_text in options_map.items():
        normalized_option = normalize_answer(option_text)
        if len(normalized_option) < 3:
            continue
        pattern = rf"(^|[\s]){re.escape(normalized_option)}($|[\s])"
        if re.search(pattern, normalized_text):
            return {"choice": choice, "answer": str(option_text), "raw": text}

    return {"choice": "", "answer": text.strip(), "raw": text}


def _canonical_label(label: Any, options: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    label_text = str(label or "").strip()
    options_map = _coerce_options(options)

    if label_text.upper() in options_map:
        letter = label_text.upper()
        return {"choice": letter, "answer": str(options_map[letter])}

    normalized_label = normalize_answer(label_text)
    for letter, option_text in options_map.items():
        if normalize_answer(option_text) == normalized_label:
            return {"choice": letter, "answer": str(option_text)}

    return {"choice": "", "answer": label_text}


def _is_correct(parsed_answer: Mapping[str, str], label_info: Mapping[str, str]) -> bool:
    label_choice = str(label_info.get("choice", "")).upper()
    pred_choice = str(parsed_answer.get("choice", "")).upper()
    if label_choice and pred_choice:
        return label_choice == pred_choice
    return is_answer_agreement(str(parsed_answer.get("answer", "")), str(label_info.get("answer", "")))


def score_answer_row(
    label: str,
    teacher_output: str,
    student_output: str,
    options: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    label_info = _canonical_label(label, options=options)
    teacher = extract_mcq_answer(teacher_output, options=options)
    student = extract_mcq_answer(student_output, options=options)

    teacher_correct = _is_correct(teacher, label_info)
    student_correct = _is_correct(student, label_info)
    distill_match = (
        teacher.get("choice", "").upper() == student.get("choice", "").upper()
        if teacher.get("choice") or student.get("choice")
        else is_answer_agreement(teacher.get("answer", ""), student.get("answer", ""))
    )

    return {
        "label": label,
        "label_choice": label_info.get("choice", ""),
        "label_answer": label_info.get("answer", ""),
        "teacher_choice": teacher.get("choice", ""),
        "teacher_answer": teacher.get("answer", ""),
        "student_choice": student.get("choice", ""),
        "student_answer": student.get("answer", ""),
        "teacher_correct": teacher_correct,
        "student_correct": student_correct,
        "teacher_label_loss": 0.0 if teacher_correct else 1.0,
        "student_label_loss": 0.0 if student_correct else 1.0,
        "distill_loss": 0.0 if distill_match else 1.0,
    }


def summarize_samples(rows: List[Dict]) -> Dict:
    if not rows:
        return {
            "num_samples": 0,
            "agreement_rate": 0.0,
            "avg_payload_bytes": 0.0,
            "avg_payload_bits": 0.0,
            "teacher_accuracy": 0.0,
            "student_accuracy": 0.0,
            "avg_teacher_label_loss": 0.0,
            "avg_student_label_loss": 0.0,
            "avg_distill_loss": 0.0,
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

    teacher_count = sum(1 for row in rows if "teacher_correct" in row)
    student_count = sum(1 for row in rows if "student_correct" in row)
    teacher_acc = (
        sum(1 for row in rows if row.get("teacher_correct") is True) / teacher_count
        if teacher_count
        else 0.0
    )
    student_acc = (
        sum(1 for row in rows if row.get("student_correct") is True) / student_count
        if student_count
        else 0.0
    )
    avg_teacher_label_loss = (
        sum(float(row.get("teacher_label_loss", 0.0)) for row in rows) / teacher_count
        if teacher_count
        else 0.0
    )
    avg_student_label_loss = (
        sum(float(row.get("student_label_loss", 0.0)) for row in rows) / student_count
        if student_count
        else 0.0
    )
    avg_distill_loss = (
        sum(float(row.get("distill_loss", 0.0)) for row in rows) / student_count
        if student_count
        else 0.0
    )

    return {
        "num_samples": len(rows),
        "agreement_rate": agreed / len(rows),
        "avg_payload_bytes": total_payload_bytes / len(rows),
        "avg_payload_bits": (total_payload_bytes * 8.0) / len(rows),
        "teacher_accuracy": teacher_acc,
        "student_accuracy": student_acc,
        "avg_teacher_label_loss": avg_teacher_label_loss,
        "avg_student_label_loss": avg_student_label_loss,
        "avg_distill_loss": avg_distill_loss,
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
    require_teacher_output: bool = False,
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
        label = item.get("label", reference)
        options = _coerce_options(item.get("options"))

        prompt = item.get("prompt") or default_prompt
        teacher_output = str(
            item.get("teacher_output")
            or item.get("teacher_answer")
            or item.get("teacher_prediction")
            or ""
        )
        if require_teacher_output and not teacher_output:
            failures.append(
                {"sample_id": sample_id, "error": "missing teacher_output/teacher_answer/teacher_prediction"}
            )
            continue

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

        score = score_answer_row(
            label=str(label),
            teacher_output=teacher_output,
            student_output=prediction,
            options=options,
        )

        rows.append(
            {
                "budget": budget.name,
                "sample_id": sample_id,
                "image": str(image_path),
                "prompt": prompt,
                "reference": str(reference),
                "label": str(label),
                "options": options,
                "teacher_output": teacher_output,
                "prediction": prediction,
                "agreement": is_answer_agreement(prediction, str(reference)),
                "payload_bytes": int(stats.get("payload_bytes", 0)),
                "encode_time_ms": float(stats.get("encode_time_ms", 0.0)),
                "feature_shape": stats.get("feature_shape"),
                "subsets": resolve_subsets(item),
                **score,
            }
        )

    return rows, failures


def to_markdown_table(summary_by_budget: Dict[str, Dict]) -> str:
    lines = [
        "| budget | samples | agreement_rate | teacher_acc | student_acc | t_loss | s_loss | distill_loss | avg_payload_bytes | avg_payload_bits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget_name in sorted(summary_by_budget.keys()):
        item = summary_by_budget[budget_name]
        lines.append(
            f"| {budget_name} | {item['num_samples']} | {item['agreement_rate']:.4f} | "
            f"{item.get('teacher_accuracy', 0.0):.4f} | {item.get('student_accuracy', 0.0):.4f} | "
            f"{item.get('avg_teacher_label_loss', 0.0):.4f} | {item.get('avg_student_label_loss', 0.0):.4f} | "
            f"{item.get('avg_distill_loss', 0.0):.4f} | {item['avg_payload_bytes']:.2f} | {item['avg_payload_bits']:.2f} |"
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
    parser.add_argument(
        "--require-teacher-output",
        action="store_true",
        help="Fail samples missing teacher_output/teacher_answer/teacher_prediction",
    )
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
            require_teacher_output=args.require_teacher_output,
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

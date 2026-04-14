"""Temporary probe for teacher/student raw output format on VQA-style samples.

Use this script to inspect raw generations and parsed MCQ choices before bulk eval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval_answer_agreement import extract_mcq_answer


def load_jsonl(path: Path, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if max_samples is not None and len(items) >= max_samples:
                break
    return items


class TeacherEngine:
    def __init__(self, model_name: str, device: str = "cpu", local_only: bool = False):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = device
        model_dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
            device_map=None,
            trust_remote_code=True,
            local_files_only=local_only,
        )
        self.model.to(device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=local_only,
        )

    def infer(self, image_path: str, prompt: str, max_new_tokens: int = 64) -> str:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return str(response)


def run_student(
    encoder: Any,
    server: str,
    image_path: str,
    prompt: str,
    timeout: int,
) -> str:
    payload, _ = encoder.encode_to_payload(image_path)
    payload["prompt"] = prompt
    response = requests.post(
        f"{server.rstrip('/')}/infer",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"student infer failed: status={response.status_code}, body={response.text[:500]}")
    return str(response.json().get("response", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe VQA output format for teacher/student")
    parser.add_argument("--dataset", type=str, required=True, help="Prepared AI2D jsonl")
    parser.add_argument("--output", type=str, required=True, help="Probe json output path")
    parser.add_argument("--max-samples", type=int, default=5)

    parser.add_argument("--teacher-model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--teacher-device", type=str, default="cpu")
    parser.add_argument("--teacher-local-only", action="store_true")
    parser.add_argument("--skip-teacher", action="store_true")

    parser.add_argument("--student-checkpoint", type=str, required=True)
    parser.add_argument("--student-device", type=str, default="cpu")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8080")
    parser.add_argument("--timeout", type=int, default=300)

    parser.add_argument(
        "--write-teacher-jsonl",
        type=str,
        default=None,
        help="Optional: write dataset jsonl with teacher_output field",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    rows = load_jsonl(dataset_path, max_samples=args.max_samples)
    if not rows:
        raise ValueError("Empty dataset")

    teacher = None
    if not args.skip_teacher:
        teacher = TeacherEngine(
            model_name=args.teacher_model,
            device=args.teacher_device,
            local_only=args.teacher_local_only,
        )

    from scripts.edge_client import EdgeEncoder

    encoder = EdgeEncoder(checkpoint_path=args.student_checkpoint, device=args.student_device)

    outputs = []
    enriched = []
    for item in rows:
        image_path = str(item["image"])
        prompt = str(item.get("prompt") or "Answer with option letter only.")
        options = item.get("options") or {}

        teacher_raw = ""
        if teacher is not None:
            teacher_raw = teacher.infer(image_path=image_path, prompt=prompt)

        student_raw = run_student(
            encoder=encoder,
            server=args.server,
            image_path=image_path,
            prompt=prompt,
            timeout=args.timeout,
        )

        teacher_parsed = extract_mcq_answer(teacher_raw, options=options) if teacher_raw else {}
        student_parsed = extract_mcq_answer(student_raw, options=options)

        outputs.append(
            {
                "sample_id": item.get("sample_id"),
                "label": item.get("label"),
                "options": options,
                "teacher_raw": teacher_raw,
                "teacher_parsed": teacher_parsed,
                "student_raw": student_raw,
                "student_parsed": student_parsed,
            }
        )

        enriched_item = dict(item)
        if teacher_raw:
            enriched_item["teacher_output"] = teacher_raw
        enriched.append(enriched_item)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved probe outputs to {output_path}")

    if args.write_teacher_jsonl:
        teacher_out = Path(args.write_teacher_jsonl).resolve()
        teacher_out.parent.mkdir(parents=True, exist_ok=True)
        with teacher_out.open("w", encoding="utf-8") as f:
            for item in enriched:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved teacher-enriched dataset to {teacher_out}")


if __name__ == "__main__":
    main()

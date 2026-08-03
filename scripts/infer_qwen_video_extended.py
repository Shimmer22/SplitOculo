#!/usr/bin/env python3
"""Qwen Native video inference with extended frame counts (2-256).
Uses ffmpeg to uniformly sample frames across the full video duration.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from PIL import Image

# numpy compat patches
for _alias, _target in [("long", np.int64), ("longlong", np.int64),
                          ("ulong", np.uint64), ("ulonglong", np.uint64),
                          ("uintc", np.uint32)]:
    if not hasattr(np, _alias):
        setattr(np, _alias, _target)

_tensor = torch.tensor
_dtype_map = {np.uint8: torch.uint8, np.int8: torch.int8, np.int16: torch.int16,
              np.int32: torch.int32, np.int64: torch.int64, np.float16: torch.float16,
              np.float32: torch.float32, np.float64: torch.float64, np.bool_: torch.bool}
torch.from_numpy = lambda ndarray: _tensor(ndarray.tolist(), dtype=_dtype_map.get(ndarray.dtype.type, torch.float32))

from core.qwen_extractor import QwenFeatureExtractor
from transformers import TextIteratorStreamer

PROMPT = (
    "These are uniformly sampled first-person supermarket video frames. "
    "Question: During the middle part of the video, the wearer selected or "
    "picked items in front of which product area? "
    "Answer in one short Chinese noun phrase. "
    "Describe only visible product category or shelf area."
)
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
MAX_NEW_TOKENS = 32
FRAME_COUNTS = [2, 4, 8, 16, 32, 64, 128, 256]


def read_video_uniform(video_path, num_frames):
    """Read uniformly sampled frames from video using ffmpeg select filter."""
    video_path = str(Path(video_path).resolve())

    # Get total frames
    probe = subprocess.run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_frames,r_frame_rate,width,height,duration',
        '-of', 'default=noprint_wrappers=1', video_path
    ], capture_output=True, text=True)
    info = {}
    for line in probe.stdout.strip().split('\n'):
        if '=' in line:
            k, v = line.split('=', 1)
            info[k] = v
    total_frames = int(info.get('nb_frames', 0))
    num, denom = info.get('r_frame_rate', '30/1').split('/')
    fps = float(num) / float(denom)
    w, h = int(info.get('width', 1920)), int(info.get('height', 1080))

    if num_frames >= total_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

    # Use ffmpeg select filter
    select_expr = '+'.join(f'eq(n,{i})' for i in indices)
    cmd = [
        'ffmpeg', '-v', 'error', '-i', video_path,
        '-vf', f"select='{select_expr}'",
        '-vsync', '0',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_size = w * h * 3
    frames = []
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        img = Image.frombytes('RGB', (w, h), raw)
        frames.append(img)

    proc.wait()
    return frames, fps, total_frames


def batch_to_device(batch, device):
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


@torch.no_grad()
def generate_with_timing(extractor, frames, max_new_tokens):
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": "sampled_frames"},
            {"type": "text", "text": PROMPT},
        ],
    }]
    text = extractor.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = extractor.processor(text=[text], videos=[frames], return_tensors="pt", padding=True)
    inputs = batch_to_device(inputs, extractor.device)

    streamer = TextIteratorStreamer(
        extractor.processor.tokenizer, skip_prompt=True,
        skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
    output_holder = {}
    def _run():
        output_holder["out"] = extractor.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, streamer=streamer
        )

    gen_start = time.perf_counter()
    t = threading.Thread(target=_run)
    t.start()

    chunks = []
    ft = None
    for chunk in streamer:
        if chunk and ft is None:
            ft = time.perf_counter() - gen_start
        chunks.append(chunk)
    t.join()
    gen_s = time.perf_counter() - gen_start
    answer = "".join(chunks).strip()

    in_len = inputs["input_ids"].shape[1]
    out_len = output_holder["out"].shape[1]
    generated_tokens = out_len - in_len

    return answer, {
        "first_token_seconds": ft,
        "generation_seconds": gen_s,
        "generated_tokens": generated_tokens,
        "average_tps": generated_tokens / gen_s if gen_s > 0 else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output_json", default="./qwen_native_results.json")
    parser.add_argument("--output_md", default="./qwen_native_results.md")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    results = []

    extractor = QwenFeatureExtractor(
        model_name=MODEL_NAME, device=args.device, extract_layer=4, local_files_only=True,
        min_pixels=224*224, max_pixels=448*448,
    ).load()

    for nf in FRAME_COUNTS:
        print(f"\n=== {nf} frames ===")
        frames, fps, total = read_video_uniform(args.video, nf)
        print(f"Sampled {len(frames)} frames from {total} total @ {fps:.2f} fps")

        t0 = time.perf_counter()
        answer, timing = generate_with_timing(extractor, frames, MAX_NEW_TOKENS)
        total_s = time.perf_counter() - t0

        r = {
            "frames": nf,
            "sampled_actual": len(frames),
            "answer": answer,
            "first_token_s": timing["first_token_seconds"],
            "total_s": total_s,
            "gen_s": timing["generation_seconds"],
            "generated_tokens": timing["generated_tokens"],
            "average_tps": timing["average_tps"],
        }
        results.append(r)
        print(f"Answer: {answer}, FT={timing['first_token_seconds']:.3f}s, Total={total_s:.3f}s")

    # Save JSON
    metadata = {
        "video": args.video,
        "total_frames": total,
        "fps": fps,
        "prompt": PROMPT,
        "ground_truth": "乳制品",
        "model": MODEL_NAME,
        "max_new_tokens": MAX_NEW_TOKENS,
        "results": results,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.output_json}")

    # Save MD
    lines = [
        "# Qwen Native Video Inference — Extended Frame Test",
        "",
        f"- Video: `{args.video}` ({total} frames, {fps:.2f} fps)",
        f"- Model: {MODEL_NAME}",
        f"- Ground truth: **乳制品**",
        f"- Prompt: {PROMPT}",
        "",
        "| Frames | Answer | FT(s) | Total(s) | Gen(s) | Tokens | TPS |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['frames']} | {r['answer']} | {r['first_token_s']:.3f} | "
            f"{r['total_s']:.3f} | {r['gen_s']:.3f} | {r['generated_tokens']} | "
            f"{r['average_tps']:.1f} |"
        )
    lines.extend([
        "",
        "## RAW JSON",
        f"See `{args.output_json}`",
    ])
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {args.output_md}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Video inference benchmark v2 — uniform frame sampling across full video.
Qwen Native + SplitOculo matrix, same frames for both."""
import argparse
import json
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
# numpy compat
for _a, _t in [("long", np.int64), ("longlong", np.int64), ("ulong", np.uint64),
                ("ulonglong", np.uint64), ("uintc", np.uint32)]:
    if not hasattr(np, _a): setattr(np, _a, _t)

import torch
_tensor = torch.tensor
_dtype_map = {np.uint8: torch.uint8, np.int8: torch.int8, np.int16: torch.int16,
              np.int32: torch.int32, np.int64: torch.int64, np.float16: torch.float16,
              np.float32: torch.float32, np.float64: torch.float64, np.bool_: torch.bool}
torch.from_numpy = lambda ndarray: _tensor(ndarray.tolist(), dtype=_dtype_map.get(ndarray.dtype.type, torch.float32))

from PIL import Image
from core.qwen_extractor import QwenFeatureExtractor

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
PAYLOAD_LEVELS = ["49x64", "49x128", "196x64", "196x128"]
GROUND_TRUTH = "乳制品"


def read_video_uniform(video_path, num_frames):
    """Read uniformly sampled frames across full video via ffmpeg."""
    video_path = str(Path(video_path).resolve())
    probe = subprocess.run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_frames,r_frame_rate,width,height',
        '-of', 'default=noprint_wrappers=1', video_path
    ], capture_output=True, text=True)
    info = {}
    for line in probe.stdout.strip().split('\n'):
        if '=' in line:
            k, v = line.split('=', 1)
            info[k] = v
    total = int(info.get('nb_frames', 0))
    num, denom = info.get('r_frame_rate', '30/1').split('/')
    fps = float(num) / float(denom)
    w, h = int(info.get('width', 1920)), int(info.get('height', 1080))

    if num_frames >= total:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()

    select_expr = '+'.join(f'eq(n,{i})' for i in indices)
    cmd = [
        'ffmpeg', '-v', 'error', '-i', video_path,
        '-vf', f"select='{select_expr}'", '-vsync', '0',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_size = w * h * 3
    frames = []
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        frames.append(Image.frombytes('RGB', (w, h), raw))
    proc.wait()
    return frames, fps, total


def batch_to_device(batch, device):
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def bench_qwen_native(extractor, frames):
    from transformers import TextIteratorStreamer
    messages = [{"role": "user", "content": [
        {"type": "video", "video": "sampled_frames"},
        {"type": "text", "text": PROMPT},
    ]}]
    text = extractor.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = extractor.processor(text=[text], videos=[frames], return_tensors="pt", padding=True)
    inputs = batch_to_device(inputs, extractor.device)

    streamer = TextIteratorStreamer(
        extractor.processor.tokenizer, skip_prompt=True, skip_special_tokens=True,
        clean_up_tokenization_spaces=False)
    output_holder = {}
    def _run():
        output_holder["out"] = extractor.model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, streamer=streamer)
    gen_start = time.perf_counter()
    t = threading.Thread(target=_run); t.start()
    chunks = []; ft = None
    for chunk in streamer:
        if chunk and ft is None: ft = time.perf_counter() - gen_start
        chunks.append(chunk)
    t.join()
    gen_s = time.perf_counter() - gen_start
    answer = "".join(chunks).strip()
    in_len = inputs["input_ids"].shape[1]
    out_len = output_holder["out"].shape[1]
    gen_tokens = out_len - in_len
    return answer, {
        "first_token_seconds": ft, "generation_seconds": gen_s,
        "generated_tokens": gen_tokens,
        "average_tps": gen_tokens / gen_s if gen_s > 0 else None,
    }


def bench_so(edge, cloud, frames, payload_level_str):
    from models.multilevel import parse_payload_levels
    level = parse_payload_levels(payload_level_str)[0]
    encoded = []
    for frame in frames:
        feats, _ = edge.encode_pil_level(frame, level)
        encoded.append(feats.squeeze(0).detach())
    compressed = torch.stack(encoded, dim=0)
    response, cm = cloud.infer_video_from_frame_features_with_timing(
        compressed.to(edge.device), prompt=PROMPT, max_new_tokens=MAX_NEW_TOKENS,
        multilevel_payload=True)
    payload_int8 = compressed.numel()
    return response, {
        "first_token_seconds": cm.get("first_token_seconds"),
        "generation_seconds": cm.get("generation_seconds"),
        "generated_tokens": cm.get("generated_tokens"),
        "average_tps": cm.get("average_tps"),
        "payload_int8_bytes": payload_int8,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--edge_checkpoint", required=True)
    parser.add_argument("--cloud_checkpoint", required=True)
    parser.add_argument("--output_md", default="./video_bench_v2.md")
    parser.add_argument("--output_json", default="./video_bench_v2.json")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    results = []

    # Load Qwen Native extractor (shared)
    print("Loading Qwen Native extractor...")
    extractor = QwenFeatureExtractor(
        model_name=MODEL_NAME, device=args.device, extract_layer=4,
        local_files_only=True, min_pixels=224*224, max_pixels=448*448,
    ).load()

    # Qwen Native baseline
    print("\n=== Qwen Native (Uniform Sampling) ===")
    for nf in FRAME_COUNTS:
        frames, fps, total = read_video_uniform(args.video, nf)
        print(f"\nQ{nf}: {len(frames)} frames from {total} total")
        t0 = time.perf_counter()
        answer, timing = bench_qwen_native(extractor, frames)
        total_s = time.perf_counter() - t0
        r = {"method": f"Q{nf}", "frames": nf, "answer": answer, "first_token_s": timing["first_token_seconds"],
             "total_s": total_s, "gen_s": timing["generation_seconds"],
             "generated_tokens": timing["generated_tokens"], "average_tps": timing["average_tps"]}
        results.append(r)
        print(f"  Answer: {answer}, FT={timing['first_token_seconds']:.3f}s, Total={total_s:.3f}s")

    # Load SO models
    print("\nLoading SO edge+cloud...")
    from scripts.edge_client import EdgeEncoder
    from scripts.cloud_server import CloudInferenceEngine
    edge = EdgeEncoder(args.edge_checkpoint, device=args.device)
    cloud = CloudInferenceEngine(args.cloud_checkpoint, device=args.device, split_layer=4)
    cloud.qwen_path = MODEL_NAME
    cloud.offline_mode = args.offline

    # SplitOculo with multi-level payload
    print("\n=== SplitOculo (Uniform Sampling) ===")
    for nf in FRAME_COUNTS:
        frames, _, _ = read_video_uniform(args.video, nf)
        for pl in PAYLOAD_LEVELS:
            tokens_str, dim_str = pl.split("x")
            label = f"S{tokens_str}{dim_str}-{nf}"
            print(f"\n{label}: {nf} frames, payload {pl}")
            t0 = time.perf_counter()
            try:
                answer, timing = bench_so(edge, cloud, frames, pl)
                total_s = time.perf_counter() - t0
                r = {"method": label, "frames": nf, "payload_level": pl,
                     "answer": answer, "first_token_s": timing["first_token_seconds"],
                     "total_s": total_s, "gen_s": timing["generation_seconds"],
                     "generated_tokens": timing["generated_tokens"],
                     "average_tps": timing["average_tps"],
                     "payload_int8_bytes": timing["payload_int8_bytes"]}
                results.append(r)
                print(f"  Answer: {answer}, FT={timing['first_token_seconds']:.3f}s, Total={total_s:.3f}s")
            except Exception as e:
                print(f"  ERROR: {traceback.format_exc()}")
                results.append({"method": label, "frames": nf, "payload_level": pl, "error": str(e)})

    # Save
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"video": args.video, "ground_truth": GROUND_TRUTH, "results": results}, f, indent=2, ensure_ascii=False)

    lines = [
        "# Video Inference v2 — Uniform Frame Sampling",
        "",
        f"- Video: `{args.video}`",
        f"- Ground truth: **{GROUND_TRUTH}**",
        f"- Prompt: {PROMPT}",
        f"- Model: {MODEL_NAME}, max_pixels=448x448",
        f"- Max new tokens: {MAX_NEW_TOKENS}",
        "",
        "## Qwen Native (Uniform Sampling)",
        "",
        "| Frames | Answer | FT(s) | Total(s) | Gen(s) | Tokens | TPS |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r["method"].startswith("Q") and not r["method"].startswith("QC"):
            lines.append(f"| {r['frames']} | {r['answer']} | {r['first_token_s']:.3f} | {r['total_s']:.3f} | {r['gen_s']:.3f} | {r['generated_tokens']} | {r['average_tps']:.1f} |")

    for pl in PAYLOAD_LEVELS:
        lines.extend([
            f"\n## SplitOculo — Payload {pl}",
            "",
            "| Frames | Answer | FT(s) | Total(s) | Payload(B) |",
            "|---|---|---|---|---|",
        ])
        t_str, d_str = pl.split("x")
        prefix = f"S{t_str}{d_str}-"
        for r in results:
            if r["method"].startswith(prefix):
                lb = r.get("payload_int8_bytes", "-")
                if isinstance(lb, int): lb = f"{lb:,}"
                lines.append(f"| {r['frames']} | {r['answer']} | {r['first_token_s']:.3f} | {r['total_s']:.3f} | {lb} |")

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved: {args.output_md}, {args.output_json}")


if __name__ == "__main__":
    main()

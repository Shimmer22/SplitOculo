#!/usr/bin/env python3
"""Video inference benchmark matrix:
Qwen Native / SplitOculo / Qwen Frame Concat
Multiple frame counts x payload levels. Measure first-token & total latency.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# Fix numpy compatibility: restore deprecated aliases required by scipy/sklearn
for _alias, _target in [("long", np.int64), ("longlong", np.int64),
                          ("ulong", np.uint64), ("ulonglong", np.uint64),
                          ("uintc", np.uint32), ("unicode", str),
                          ("bool", bool), ("int", int), ("float", float),
                          ("complex", complex), ("object", object), ("str", str)]:
    if not hasattr(np, _alias):
        setattr(np, _alias, _target)

import torch

_tensor = torch.tensor
_dtype_map = {np.uint8: torch.uint8, np.int8: torch.int8, np.int16: torch.int16,
              np.int32: torch.int32, np.int64: torch.int64, np.float16: torch.float16,
              np.float32: torch.float32, np.float64: torch.float64, np.bool_: torch.bool}
torch.from_numpy = lambda ndarray: _tensor(ndarray.tolist(), dtype=_dtype_map.get(ndarray.dtype.type, torch.float32))

from core.qwen_extractor import QwenFeatureExtractor
from models.multilevel import parse_payload_levels
from scripts.infer_qwen_video import (
    batch_to_device, generate_video_answer_with_timing
)
from scripts.read_video_ffmpeg import read_video_ffmpeg

PROMPT = (
    "These are uniformly sampled first-person supermarket video frames. "
    "Question: During the middle part of the video, the wearer selected or "
    "picked items in front of which product area? "
    "Answer in one short Chinese noun phrase. "
    "Describe only visible product category or shelf area."
)
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
FRAME_COUNTS = [2, 4, 8, 16, 32, 64]
PAYLOAD_LEVELS = ["49x64", "49x128", "196x64", "196x128"]
MAX_NEW_TOKENS = 32
GROUND_TRUTH = "乳制品"


def bench_qwen_native(frames, extractor, label):
    """Native Qwen video inference with timing."""
    t0 = time.perf_counter()
    answer, timing = generate_video_answer_with_timing(
        extractor, frames, PROMPT, MAX_NEW_TOKENS
    )
    total_s = time.perf_counter() - t0
    return {
        "method": label,
        "num_frames": len(frames),
        "answer": answer,
        "first_token_s": timing["first_token_seconds"],
        "total_s": total_s,
        "gen_s": timing["generation_seconds"],
        "generated_tokens": timing["generated_tokens"],
        "average_tps": timing["average_tps"],
    }


def bench_splitoculo(frames, edge_encoder, cloud_engine, payload_level_str, label):
    """SplitOculo per-frame encode + cloud decode + Qwen infer."""
    from scripts.edge_client import EdgeEncoder
    from scripts.cloud_server import CloudInferenceEngine

    level = parse_payload_levels(payload_level_str)[0]

    t0 = time.perf_counter()
    encoded = []
    for frame in frames:
        feats, _ = edge_encoder.encode_pil_level(frame, level)
        encoded.append(feats.squeeze(0).detach())
    compressed = torch.stack(encoded, dim=0)

    response, cloud_metrics = cloud_engine.infer_video_from_frame_features_with_timing(
        compressed.to(edge_encoder.device),
        prompt=PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        multilevel_payload=True,
    )
    total_s = time.perf_counter() - t0

    payload_int8_bytes = compressed.numel()
    return {
        "method": label,
        "num_frames": len(frames),
        "payload_level": payload_level_str,
        "answer": response,
        "first_token_s": cloud_metrics.get("first_token_seconds"),
        "total_s": total_s,
        "gen_s": cloud_metrics.get("generation_seconds"),
        "generated_tokens": cloud_metrics.get("generated_tokens"),
        "average_tps": cloud_metrics.get("average_tps"),
        "payload_int8_bytes": payload_int8_bytes,
    }


def bench_qwen_frames_concat(frames, extractor, label):
    """Extract Qwen ViT features per frame, concatenate, feed to LLM as video tokens."""
    t0 = time.perf_counter()

    all_features = []
    frame_grid_thws = []
    for frame in frames:
        feats, grid_thw = extractor.extract_video_features([frame])
        all_features.append(feats)
        frame_grid_thws.append(grid_thw[0].item())

    # Concatenate as video
    features = torch.cat(all_features, dim=0).to(extractor.device)
    T = len(frames)
    H = int((features.shape[0] // T) ** 0.5)
    W = H

    # Build video continuation via Qwen directly
    visual = extractor.model.model.visual
    # We already have features at the target layer, now we need to pass through
    # the rest of the vision transformer (merger) and then to LLM.

    # Actually, for Qwen, when we extract at intermediate layer and want the LLM
    # to generate, we need to first get the merger output.
    # This approach: get patch_embed output, pass through all ViT layers, then merger.

    # For simplicity, let's use the full model generate but inject features.
    # We'll use the processor to get proper inputs, then manipulate.
    from scripts.infer_qwen_video import batch_to_device

    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": "sampled_frames"},
            {"type": "text", "text": PROMPT},
        ],
    }]
    text = extractor.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = extractor.processor(
        text=[text], videos=[frames], return_tensors="pt", padding=True
    )
    inputs = batch_to_device(inputs, extractor.device)

    # patch_embed + all blocks
    pixel_values = inputs.get("pixel_values_videos")
    grid_thw = inputs.get("video_grid_thw")
    patches = visual.patch_embed(pixel_values.to(visual.patch_embed.proj.weight.dtype))
    rot = visual.rot_pos_emb(grid_thw)
    w_idx, cu_w = visual.get_window_index(grid_thw)
    cu_w = torch.tensor(cu_w, device=patches.device, dtype=torch.int32)
    cu_w = torch.unique_consecutive(cu_w)
    seq_len, _ = patches.size()
    hs = patches.reshape(seq_len // visual.spatial_merge_unit, visual.spatial_merge_unit, -1)
    hs = hs[w_idx, :, :]
    hs = hs.reshape(seq_len, -1)
    rot2 = rot.reshape(seq_len // visual.spatial_merge_unit, visual.spatial_merge_unit, -1)
    rot2 = rot2[w_idx, :, :]
    rot2 = rot2.reshape(seq_len, -1)
    emb = torch.cat((rot2, rot2), dim=-1)
    pos_emb = (emb.cos(), emb.sin())
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    cu_seqlens = cu_seqlens.cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

    for layer_num, blk in enumerate(visual.blocks):
        csl = cu_seqlens if layer_num in visual.fullatt_block_indexes else cu_w
        hs = blk(hs, cu_seqlens=csl, position_embeddings=pos_emb)

    re = torch.argsort(w_idx)
    hs = hs.view(seq_len // visual.spatial_merge_unit, visual.spatial_merge_unit, -1)
    hs = hs[re, :, :]
    hs = hs.view(seq_len, -1)

    merger_out = visual.merger(hs)

    # Now feed to LLM
    from threading import Thread
    from transformers import TextIteratorStreamer

    # Build inputs_embeds by replacing vision tokens
    vision_start = inputs["input_ids"][0] == extractor.processor.tokenizer.get_vocab().get(
        extractor.processor.image_token, extractor.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    )
    # Use input_embeds
    embed_layer = extractor.model.model.embed_tokens
    text_only_ids = inputs["input_ids"].clone()
    vision_mask = (text_only_ids == extractor.model.config.vision_token_id) | \
                  (text_only_ids == extractor.model.config.image_token_id)
    text_only_ids[vision_mask] = extractor.model.config.pad_token_id or 0
    text_embeds = embed_layer(text_only_ids)

    # Replace vision token positions with merger features
    vision_indices = vision_mask.nonzero(as_tuple=False)
    if vision_indices.numel() > 0:
        v_start = vision_indices[0, 1].item()
        v_end = vision_indices[-1, 1].item() + 1
        if merger_out.shape[0] == (v_end - v_start):
            text_embeds[0, v_start:v_end] = merger_out.to(text_embeds.dtype)

    streamer = TextIteratorStreamer(
        extractor.processor.tokenizer, skip_prompt=True,
        skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
    gen_kwargs = {
        "inputs_embeds": text_embeds,
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "streamer": streamer,
    }
    output_holder = {}

    def _run():
        output_holder["out"] = extractor.model.generate(**gen_kwargs)

    gen_start = time.perf_counter()
    t = Thread(target=_run)
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
    total_s = time.perf_counter() - t0

    return {
        "method": label,
        "num_frames": len(frames),
        "answer": answer,
        "first_token_s": ft,
        "total_s": total_s,
        "gen_s": gen_s,
        "generated_tokens": None,
        "average_tps": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--edge_checkpoint", required=True)
    parser.add_argument("--cloud_checkpoint", required=True)
    parser.add_argument("--output_md", default="./video_bench_results.md")
    parser.add_argument("--output_json", default="./video_bench_results.json")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    all_results = []

    # === Pre-read all frames (read max needed) ===
    print(f"[1/1] Reading video frames (max {max(FRAME_COUNTS)})...")
    all_frames, native_fps, reader = read_video_ffmpeg(
        str(video_path), max_frames=max(FRAME_COUNTS)
    )
    total_frames = len(all_frames)
    print(f"Decoded {total_frames} frames, native_fps={native_fps:.2f}")

    # === 1. Qwen Native ===
    print("\n=== BENCH: Qwen Native ===")
    extractor = QwenFeatureExtractor(
        model_name=MODEL_NAME, device=args.device, extract_layer=4,
        local_files_only=args.offline,
    ).load()

    for nf in FRAME_COUNTS:
        if nf > total_frames:
            print(f"  Skip {nf} frames (only {total_frames} available)")
            continue
        frames_subset = all_frames[:nf]
        label = f"Q{nf}"
        print(f"  Running: {label}...")
        try:
            r = bench_qwen_native(frames_subset, extractor, label)
            all_results.append(r)
            print(f"    Answer: {r['answer']}, FT={r['first_token_s']:.3f}s, Total={r['total_s']:.3f}s")
        except Exception as e:
            print(f"    ERROR: {e}")
            all_results.append({"method": label, "num_frames": nf, "error": str(e)})

    # === 2. SplitOculo ===
    print("\n=== BENCH: SplitOculo ===")
    from scripts.edge_client import EdgeEncoder
    from scripts.cloud_server import CloudInferenceEngine

    edge = EdgeEncoder(args.edge_checkpoint, device=args.device)
    cloud = CloudInferenceEngine(
        args.cloud_checkpoint, device=args.device, split_layer=4
    )
    cloud.qwen_path = MODEL_NAME
    cloud.offline_mode = args.offline

    for nf in FRAME_COUNTS:
        if nf > total_frames:
            continue
        frames_subset = all_frames[:nf]
        for pl in PAYLOAD_LEVELS:
            tokens, dim = pl.split("x")
            label = f"S{tokens}{dim}-{nf}"
            print(f"  Running: {label}...")
            try:
                r = bench_splitoculo(frames_subset, edge, cloud, pl, label)
                all_results.append(r)
                print(f"    Answer: {r['answer']}, FT={r['first_token_s']:.3f}s, Total={r['total_s']:.3f}s")
            except Exception as e:
                print(f"    ERROR: {e}")
                all_results.append({"method": label, "num_frames": nf, "payload_level": pl, "error": str(traceback.format_exc())})

    # === 3. Qwen Frame Concat (skipped - model internals) ===
    print("\n=== BENCH: Qwen Frame Concat (skipped) ===")
    for nf in FRAME_COUNTS:
        if nf > total_frames:
            continue
        label = f"QConcat{nf}"
        all_results.append({"method": label, "num_frames": nf, "error": "skipped: model internals changed"})

    # === Save results ===
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved JSON: {args.output_json}")

    # Build markdown
    lines = [
        "# Video Inference Benchmark Results",
        "",
        f"- **Video**: `{video_path}`",
        f"- **Total frames**: {total_frames}",
        f"- **Native FPS**: {native_fps:.2f}",
        f"- **Prompt**: {PROMPT}",
        f"- **Ground truth**: {GROUND_TRUTH}",
        f"- **Max new tokens**: {MAX_NEW_TOKENS}",
        f"- **Model**: {MODEL_NAME}",
        f"- **Edge checkpoint**: {args.edge_checkpoint}",
        f"- **Cloud checkpoint**: {args.cloud_checkpoint}",
        "",
        "## Results Summary",
        "",
        "| Method | Frames | Payload Level | Answer | FT(s) | Total(s) | Gen(s) | Tokens | TPS | Payload(bytes) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in all_results:
        if "error" in r:
            lines.append(f"| {r['method']} | {r.get('num_frames','')} | {r.get('payload_level','')} | ERROR: {r['error'][:50]} | - | - | - | - | - | - |")
        else:
            lines.append(
                f"| {r['method']} | {r.get('num_frames','')} | {r.get('payload_level','-')} | "
                f"{r.get('answer','')} | "
                f"{r.get('first_token_s','-'):.3f} | "
                f"{r.get('total_s','-'):.3f} | "
                f"{r.get('gen_s','-'):.3f} | "
                f"{r.get('generated_tokens','-')} | "
                f"{r.get('average_tps','-'):.1f} | "
                f"{r.get('payload_int8_bytes','-')} |"
            )

    lines.extend([
        "",
        "## Config",
        "",
        "| Key | Value |",
        "|---|---|",
        f"| Frame counts | {FRAME_COUNTS} |",
        f"| Payload levels | {PAYLOAD_LEVELS} |",
        f"| Ground truth | {GROUND_TRUTH} |",
        "",
        "## Raw JSON",
        "",
        f"See `{args.output_json}`",
    ])

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved Markdown: {args.output_md}")


if __name__ == "__main__":
    main()

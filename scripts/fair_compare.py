#!/usr/bin/env python3
"""Fair comparison: Qwen Native Video vs SplitOculo per-frame concat.
Both use same frames, same resolution, fair timing from image→answer."""
import subprocess, sys, time, threading, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
for _a,_t in [('long',np.int64),('longlong',np.int64),('ulong',np.uint64),('ulonglong',np.uint64),('uintc',np.uint32)]:
    if not hasattr(np,_a): setattr(np,_a,_t)
import torch
_t=torch.tensor; _dm={np.uint8:torch.uint8,np.int8:torch.int8,np.int16:torch.int16,np.int32:torch.int32,
    np.int64:torch.int64,np.float16:torch.float16,np.float32:torch.float32,np.float64:torch.float64,np.bool_:torch.bool}
torch.from_numpy=lambda x:_t(x.tolist(),dtype=_dm.get(x.dtype.type,torch.float32))

from PIL import Image
from core.qwen_extractor import QwenFeatureExtractor
from transformers import TextIteratorStreamer

VIDEO_PATH = "/data/downloads/20260115_0001_dajiangaction5pro_lingshouxiaofei_chaoshi_0001.MP4"
EDGE_CKPT = "/workspace/SplitOculo/checkpoints/ckpt/split/edge_weights.pth"
CLOUD_CKPT = "/workspace/SplitOculo/checkpoints/ckpt/split/cloud_weights.pth"
PROMPT = "These are uniformly sampled first-person supermarket video frames. Question: During the middle part of the video, the wearer selected or picked items in front of which product area? Answer in one short Chinese noun phrase. Describe only visible product category or shelf area."
FRAME_COUNTS = [2, 4, 8, 16, 32, 64, 128]
PAYLOAD_LEVELS = ["49x64", "49x128", "196x64", "196x128"]
MAX_NEW_TOKENS = 32


def read_uniform_frames(video_path, nf):
    video_path = str(Path(video_path).resolve())
    probe = subprocess.run(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=nb_frames,r_frame_rate,width,height','-of','default=noprint_wrappers=1',video_path],
        capture_output=True,text=True)
    info = {}
    for line in probe.stdout.strip().split('\n'):
        if '=' in line: k,v = line.split('=',1); info[k]=v
    total = int(info.get('nb_frames',0))
    w,h = int(info.get('width',1920)), int(info.get('height',1080))
    num,denom = info.get('r_frame_rate','30/1').split('/'); fps=float(num)/float(denom)
    indices = np.linspace(0,total-1,nf,dtype=int).tolist()
    sel = '+'.join(f'eq(n,{i})' for i in indices)
    proc = subprocess.Popen(['ffmpeg','-v','error','-i',video_path,'-vf',f"select='{sel}'",'-vsync','0',
        '-f','rawvideo','-pix_fmt','rgb24','-'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    fs = w*h*3; frames=[]
    while True:
        raw=proc.stdout.read(fs)
        if len(raw)<fs: break
        frames.append(Image.frombytes('RGB',(w,h),raw))
    proc.wait()
    return frames, fps, total


def batch_to_device(batch, device):
    return {k: v.to(device) if hasattr(v,"to") else v for k,v in batch.items()}


def make_qwen_inputs(ext, frames, prompt):
    messages = [{"role":"user","content":[{"type":"video","video":"sampled_frames"},{"type":"text","text":prompt}]}]
    text = ext.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return ext.processor(text=[text], videos=[frames], return_tensors="pt", padding=True)


def qwen_native_infer(ext, frames):
    inputs = make_qwen_inputs(ext, frames, PROMPT)
    inputs = batch_to_device(inputs, ext.device)
    t0 = time.perf_counter()
    streamer = TextIteratorStreamer(ext.processor.tokenizer, skip_prompt=True, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    out_holder = {}
    def _run(): out_holder["out"] = ext.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, streamer=streamer)
    t = threading.Thread(target=_run); t.start()
    chunks=[]; ft=None
    for chunk in streamer:
        if chunk and ft is None: ft = time.perf_counter()-t0
        chunks.append(chunk)
    t.join(); end = time.perf_counter()
    total = end-t0; gen_s = end-t0 if ft is None else end-(t0+ft)
    ans = "".join(chunks).strip()
    ilen = inputs["input_ids"].shape[1]; olen = out_holder["out"].shape[1]
    return ans, {"first_token_s": ft, "total_s": total, "gen_tokens": olen-ilen,
                  "total_start_to_end": end-t0, "input_size": inputs.get("pixel_values_videos",torch.zeros(1)).shape}


# ===== Main =====
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output_md",default="/workspace/SplitOculo/fair_compare.md")
    parser.add_argument("--output_json",default="/workspace/SplitOculo/fair_compare.json")
    args=parser.parse_args()

    all_results = []
    total_frames_cache = {}

    # Pre-load Qwen Native model
    print("Pre-loading Qwen Native...")
    qext = QwenFeatureExtractor(min_pixels=224*224, max_pixels=448*448, device="cuda", local_files_only=True, extract_layer=4).load()

    # Pre-load SO models
    print("Pre-loading SplitOculo Edge+Cloud...")
    from scripts.edge_client import EdgeEncoder
    from scripts.cloud_server import CloudInferenceEngine
    from models.multilevel import parse_payload_levels
    edge = EdgeEncoder(EDGE_CKPT, device="cuda")
    cloud = CloudInferenceEngine(CLOUD_CKPT, device="cuda", split_layer=4)
    cloud.qwen_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    cloud.offline_mode = True

    print("\n=== Running benchmarks (models pre-loaded, no one-time costs) ===\n")

    for nf in FRAME_COUNTS:
        print(f"\n{'='*60}")
        print(f"FRAMES: {nf}")
        print(f"{'='*60}")

        if nf not in total_frames_cache:
            frames, fps, total = read_uniform_frames(VIDEO_PATH, nf)
            total_frames_cache[nf] = frames
            print(f"Decoded {len(frames)} frames from {total} total")
        else:
            frames = total_frames_cache[nf]

        # === Qwen Native (VIDEO API) ===
        print(f"\n  Qwen Native VIDEO API:")
        ans, tim = qwen_native_infer(qext, frames)
        print(f"    Answer: {ans}")
        print(f"    FT={tim['first_token_s']:.3f}s, Total={tim['total_s']:.3f}s, Tokens={tim['gen_tokens']}")
        print(f"    Input shape: {tim['input_size']}")
        all_results.append({"method":"Q"+str(nf),"frames":nf,"answer":ans, **tim})

        # === SplitOculo per-frame encoding + concat ===
        for pl in PAYLOAD_LEVELS:
            label = f"S{pl.split('x')[0]}{pl.split('x')[1]}-{nf}"
            level = parse_payload_levels(pl)[0]

            # Edge encode all frames
            t0 = time.perf_counter()
            encoded = []
            for frame in frames:
                feats, _ = edge.encode_pil_level(frame, level)
                encoded.append(feats.squeeze(0).detach())
            compressed = torch.stack(encoded, dim=0)
            edge_time = time.perf_counter()-t0

            # Cloud infer
            cloud_t0 = time.perf_counter()
            resp, cm = cloud.infer_video_from_frame_features_with_timing(
                compressed.to("cuda"), prompt=PROMPT, max_new_tokens=MAX_NEW_TOKENS,
                multilevel_payload=True)
            cloud_time = time.perf_counter()-cloud_t0
            total_time = edge_time + cloud_time

            print(f"\n  {label}: Payload {pl}")
            print(f"    Edge encode: {edge_time:.4f}s | Cloud infer: {cloud_time:.4f}s | Total: {total_time:.4f}s")
            print(f"    Answer: {resp}")
            print(f"    FT={cm.get('first_token_seconds',0):.3f}s, Gen={cm.get('generation_seconds',0):.3f}s")
            all_results.append({
                "method": label, "frames": nf, "payload": pl, "answer": resp,
                "edge_time_s": edge_time, "cloud_time_s": cloud_time, "total_s": total_time,
                "first_token_s": cm.get("first_token_seconds"),
                "generation_s": cm.get("generation_seconds"),
                "generated_tokens": cm.get("generated_tokens"),
                "payload_bytes": compressed.numel(),
            })

    # Save
    with open(args.output_json,"w") as f: json.dump(all_results,f,indent=2,ensure_ascii=False)

    # MD
    lines=["# Fair Comparison: Qwen Native Video API vs SplitOculo","",
           f"- All models pre-loaded. Timing: frame decode (excluded, identical) → image → first_token / end",
           f"- Prompt: {PROMPT}",
           f"- Ground truth: 乳制品 / Dairy",
           f"- Resolution: Qwen max=448×448, SO edge=224×224",
           ""]

    # Qwen table
    lines.append("## Qwen Native (Video API)")
    lines.append("| Frames | Answer | Match | FT(s) | Total(s) | Gen(s) | Tokens |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in all_results:
        if r["method"].startswith("Q") and not r["method"].startswith("QC"):
            match = "✅" if "乳制" in r["answer"] or "Dairy" in r["answer"] or "dairy" in r["answer"] else "❌"
            lines.append(f"| {r['frames']} | {r['answer']} | {match} | {r['first_token_s']:.3f} | {r['total_s']:.3f} | {r.get('generation_s',r['total_s']):.3f} | {r.get('gen_tokens','')} |")

    # SO tables per payload
    for pl in PAYLOAD_LEVELS:
        lines.append(f"\n## SplitOculo — {pl}")
        lines.append("| Frames | Answer | Match | Edge(s) | Cloud(s) | Total(s) | FT(s) | Payload |")
        lines.append("|---|---|---|---|---|---|---|---|")
        prefix = f"S{pl.split('x')[0]}{pl.split('x')[1]}-"
        for r in all_results:
            if r["method"].startswith(prefix):
                match = "✅" if "Dairy" in r["answer"] or "dairy" in r["answer"] else ("✅" if "乳制" in r["answer"] else "❌")
                lines.append(f"| {r['frames']} | {r['answer']} | {match} | {r['edge_time_s']:.3f} | {r['cloud_time_s']:.3f} | {r['total_s']:.3f} | {r['first_token_s']:.3f} | {r['payload_bytes']:,} |")

    # Speedup table
    lines.append("\n## Speedup vs Qwen Native (Total Latency)")
    lines.append("| Frames | Qwen(s) | S196128(s) | Speedup | Qwen Acc | SO Acc |")
    lines.append("|---|---|---|---|---|---|")
    for nf in FRAME_COUNTS:
        q_row = next((r for r in all_results if r["method"]==f"Q{nf}"), None)
        s_row = next((r for r in all_results if r["method"]==f"S196128-{nf}"), None)
        if q_row and s_row:
            sp = q_row["total_s"]/s_row["total_s"]
            qm = "✅" if "乳制" in q_row["answer"] or "Dairy" in q_row["answer"] or "dairy" in q_row["answer"] else "❌"
            sm = "✅" if "Dairy" in s_row["answer"] or "dairy" in s_row["answer"] else "❌"
            lines.append(f"| {nf} | {q_row['total_s']:.1f} | {s_row['total_s']:.1f} | {sp:.1f}× | {qm} | {sm} |")

    with open(args.output_md,"w") as f: f.write("\n".join(lines)+"\n")
    print(f"\nSaved: {args.output_md}")

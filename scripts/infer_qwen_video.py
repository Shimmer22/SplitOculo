"""Run Qwen2.5-VL video inference and optionally save ViT-layer video tokens.

Examples:
    python scripts/infer_qwen_video.py --video ./data/test.mp4

    python scripts/infer_qwen_video.py \
      --video ./data/test.mp4 \
      --prompt "Describe the key events in this video." \
      --save_features ./data/qwen_video_features/test_layer4.pt \
      --layer 4 \
      --max_frames 16 \
      --offline
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from PIL import Image
from transformers import TextIteratorStreamer

from core.qwen_extractor import QwenFeatureExtractor


def _uniform_indices(total_frames, max_frames):
    if total_frames <= 0:
        return []
    if max_frames is None or total_frames <= max_frames:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, max_frames).round().astype(int).tolist()


def _fps_indices(total_frames, native_fps, sample_fps, max_frames):
    if not sample_fps or not native_fps or native_fps <= 0:
        return _uniform_indices(total_frames, max_frames)
    step = max(1, int(round(native_fps / sample_fps)))
    indices = list(range(0, total_frames, step))
    if max_frames is not None:
        indices = indices[:max_frames]
    return indices


def _read_video_torchvision(video_path, max_frames=None, sample_fps=None):
    from torchvision.io import read_video

    video, _, info = read_video(str(video_path), pts_unit="sec", output_format="THWC")
    total_frames = int(video.shape[0])
    native_fps = float(info.get("video_fps", 0.0) or 0.0)
    indices = _fps_indices(total_frames, native_fps, sample_fps, max_frames)
    frames = [Image.fromarray(video[i].numpy()).convert("RGB") for i in indices]
    return frames, native_fps


def _read_video_cv2(video_path, max_frames=None, sample_fps=None):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = set(_fps_indices(total_frames, native_fps, sample_fps, max_frames))
    last_index = max(indices) if indices else -1
    frames = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if last_index >= 0 and frame_idx > last_index:
            break
        if frame_idx in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame).convert("RGB"))
        frame_idx += 1
    cap.release()
    return frames, native_fps


def read_video_frames(video_path, max_frames=None, sample_fps=None):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    if video_path.is_file() and video_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        frame = Image.open(video_path).convert("RGB")
        return [frame], 0.0, "image_file"

    if video_path.is_dir():
        image_paths = sorted(
            p for p in video_path.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if max_frames is not None:
            image_paths = image_paths[:max_frames]
        frames = [Image.open(p).convert("RGB") for p in image_paths]
        if frames:
            return frames, 0.0, "image_dir"
        raise RuntimeError(f"decoded zero frames from image directory: {video_path}")

    errors = []
    readers = (_read_video_cv2, _read_video_torchvision) if sample_fps else (_read_video_torchvision, _read_video_cv2)
    for reader in readers:
        try:
            frames, native_fps = reader(video_path, max_frames=max_frames, sample_fps=sample_fps)
            if frames:
                return frames, native_fps, reader.__name__.replace("_read_video_", "")
            errors.append(f"{reader.__name__}: decoded zero frames")
        except Exception as exc:
            errors.append(f"{reader.__name__}: {exc}")

    raise RuntimeError(
        "Video decode failed. Install a working torchvision video backend or opencv-python. "
        + " | ".join(errors)
    )


def batch_to_device(batch, device):
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


@torch.no_grad()
def generate_video_answer(extractor, frames, prompt, max_new_tokens):
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": "sampled_frames"},
            {"type": "text", "text": prompt},
        ],
    }]
    text = extractor.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = extractor.processor(
        text=[text],
        videos=[frames],
        return_tensors="pt",
        padding=True,
    )
    inputs = batch_to_device(inputs, extractor.device)

    output_ids = extractor.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], output_ids)
    ]
    return extractor.processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


@torch.no_grad()
def generate_video_answer_with_timing(extractor, frames, prompt, max_new_tokens):
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": "sampled_frames"},
            {"type": "text", "text": prompt},
        ],
    }]
    text = extractor.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = extractor.processor(
        text=[text],
        videos=[frames],
        return_tensors="pt",
        padding=True,
    )
    inputs = batch_to_device(inputs, extractor.device)

    streamer = TextIteratorStreamer(
        extractor.processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "streamer": streamer,
    }
    output_holder = {}
    error_holder = {}

    def _run_generate():
        try:
            output_holder["output_ids"] = extractor.model.generate(**generation_kwargs)
        except Exception as exc:
            error_holder["error"] = exc

    generation_start = time.perf_counter()
    thread = threading.Thread(target=_run_generate)
    thread.start()

    chunks = []
    first_chunk_seconds = None
    for chunk in streamer:
        if chunk and first_chunk_seconds is None:
            first_chunk_seconds = time.perf_counter() - generation_start
        chunks.append(chunk)

    thread.join()
    generation_seconds = time.perf_counter() - generation_start
    if "error" in error_holder:
        raise error_holder["error"]

    output_ids = output_holder["output_ids"]
    generated_ids = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], output_ids)
    ]
    generated_tokens = int(generated_ids[0].numel()) if generated_ids else 0
    answer = "".join(chunks).strip()
    if not answer and generated_ids:
        answer = extractor.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    first_token_seconds = first_chunk_seconds
    decode_seconds_after_first = None
    average_tps = None
    if generated_tokens > 0 and generation_seconds > 0:
        average_tps = generated_tokens / generation_seconds
    if first_token_seconds is not None:
        decode_seconds_after_first = max(0.0, generation_seconds - first_token_seconds)

    return answer, {
        "first_token_seconds": first_token_seconds,
        "generation_seconds": generation_seconds,
        "generated_tokens": generated_tokens,
        "average_tps": average_tps,
        "decode_seconds_after_first": decode_seconds_after_first,
    }


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL video inference smoke test")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--prompt", type=str, default="Describe the key events in this video.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--layer", type=int, default=4, help="ViT layer to save when --save_features is set")
    parser.add_argument("--max_frames", type=int, default=16)
    parser.add_argument("--sample_fps", type=float, default=None)
    parser.add_argument("--min_pixels", type=int, default=224 * 224)
    parser.add_argument("--max_pixels", type=int, default=448 * 448)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--save_features", type=str, default=None)
    parser.add_argument("--metadata", type=str, default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    frames, native_fps, reader = read_video_frames(
        args.video,
        max_frames=args.max_frames,
        sample_fps=args.sample_fps,
    )
    print(f"Decoded {len(frames)} frames with {reader} (native_fps={native_fps:.3f})")

    extractor = QwenFeatureExtractor(
        model_name=args.model_name,
        device=args.device,
        extract_layer=args.layer,
        local_files_only=args.offline,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    ).load()

    answer = generate_video_answer(
        extractor=extractor,
        frames=frames,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )
    print("\nResponse:")
    print(answer)

    feature_info = None
    if args.save_features:
        features, video_grid_thw = extractor.extract_video_features(frames)
        output_path = Path(args.save_features)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "features": features,
            "path": str(Path(args.video)),
            "num_tokens": int(features.shape[0]),
            "hidden_size": int(features.shape[1]),
            "video_grid_thw": video_grid_thw.tolist(),
            "frames_sampled": len(frames),
            "native_fps": native_fps,
            "sample_fps": args.sample_fps,
            "model_name": args.model_name,
            "extract_layer": args.layer,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
        }
        torch.save(payload, output_path)
        feature_info = {
            "feature_path": str(output_path),
            "feature_shape": list(features.shape),
            "video_grid_thw": video_grid_thw.tolist(),
        }
        print(f"\nSaved features: {output_path}")
        print(f"Feature shape: {tuple(features.shape)}, video_grid_thw={video_grid_thw.tolist()}")

    if args.metadata:
        metadata = {
            "video": str(Path(args.video)),
            "prompt": args.prompt,
            "response": answer,
            "frames_sampled": len(frames),
            "native_fps": native_fps,
            "reader": reader,
            "model_name": args.model_name,
            "feature_info": feature_info,
        }
        metadata_path = Path(args.metadata)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()

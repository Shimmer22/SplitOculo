"""Local Electron demo client for pure-Qwen and SplitOculo inference.

The baseline sends sampled RGB frames to the cloud and runs the complete Qwen
vision encoder plus language model. SplitOculo variants send quantised feature
tensors produced by the edge encoder.
"""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.codec_video_reader import read_video_records_with_mvs
from models.codec_accelerator import CodecWarpAccelerator, DecoderMotionVectorAccelerator
from models.multilevel import parse_payload_levels
from models.temporal_pair import load_temporal_pair_fusion
from scripts.edge_client import EdgeEncoder, load_cloud_checkpoint
from scripts.infer_qwen_video import read_video_frames


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
RAW_EXTENSIONS = {".raw", ".rgb", ".bgr", ".yuv"}


class RoundInterrupted(RuntimeError):
    def __init__(self, partial_text="", result=None):
        super().__init__("round inference deadline reached")
        self.partial_text = partial_text
        self.result = dict(result or {})


def _raise_for_cloud_error(response):
    """Turn a cloud HTTP failure into a useful demo error."""
    if response.ok:
        return
    message = ""
    try:
        message = str((response.json() or {}).get("error", "")).strip()
    except (ValueError, TypeError):
        pass
    if not message:
        message = response.reason or "cloud request failed"
    raise requests.HTTPError(
        f"Cloud HTTP {response.status_code}: {message}",
        response=response,
    )


def _post_cloud_stream(url, payload, args, on_text):
    interrupt_after = getattr(args, "current_interrupt_after_seconds", None)
    interrupt_deadline = getattr(args, "current_interrupt_deadline", None)
    request_started = time.monotonic()
    request_timeout = args.timeout
    if interrupt_deadline is not None:
        remaining = interrupt_deadline - request_started
        if remaining <= 0:
            raise RoundInterrupted()
        request_timeout = (min(float(args.timeout), 10.0), remaining)
    elif interrupt_after is not None:
        interrupt_deadline = request_started + float(interrupt_after)
        request_timeout = (min(float(args.timeout), 10.0), float(interrupt_after))
    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"},
            auth=(args.username, args.password) if args.username else None,
            timeout=request_timeout,
            stream=True,
        )
    except requests.Timeout as exc:
        if interrupt_after is not None:
            raise RoundInterrupted() from exc
        raise
    _raise_for_cloud_error(response)
    response.encoding = "utf-8"
    result = None
    chunks = []
    interrupted = threading.Event()
    timer = None
    if interrupt_deadline is not None:
        remaining = max(0.0, interrupt_deadline - time.monotonic())

        def _interrupt_response():
            interrupted.set()
            close_response = getattr(response, "close", None)
            if close_response is not None:
                close_response()

        timer = threading.Timer(remaining, _interrupt_response)
        timer.daemon = True
        timer.start()
    try:
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            event = json.loads(line)
            event_type = event.get("type")
            if event_type == "delta":
                text = str(event.get("text") or "")
                if text:
                    chunks.append(text)
                    on_text(text)
            elif event_type == "result":
                result = event.get("result")
            elif event_type == "error":
                raise RuntimeError(str(event.get("error") or "cloud streaming failed"))
    except Exception as exc:
        if interrupted.is_set():
            raise RoundInterrupted("".join(chunks)) from exc
        raise
    finally:
        if timer is not None:
            timer.cancel()
        close_response = getattr(response, "close", None)
        if close_response is not None:
            close_response()
    if interrupted.is_set():
        raise RoundInterrupted("".join(chunks))
    if not isinstance(result, dict):
        raise RuntimeError("cloud stream ended without a result event")
    return result


def _raw_frames(path: Path, width: int, height: int, pixel_format: str):
    if width <= 0 or height <= 0:
        raise ValueError("Raw frame width and height must be positive")
    channels = 1 if pixel_format == "gray8" else 3
    frame_bytes = width * height * channels
    data = path.read_bytes()
    if len(data) % frame_bytes:
        raise ValueError(
            f"Raw input size {len(data)} is not divisible by one {width}x{height} "
            f"{pixel_format} frame ({frame_bytes} bytes)"
        )
    frames = []
    for offset in range(0, len(data), frame_bytes):
        array = np.frombuffer(data[offset : offset + frame_bytes], dtype=np.uint8)
        if pixel_format == "gray8":
            array = np.repeat(array.reshape(height, width, 1), 3, axis=2)
        else:
            array = array.reshape(height, width, 3)
            if pixel_format == "bgr24":
                array = array[:, :, ::-1]
        frames.append(Image.fromarray(array.copy(), mode="RGB"))
    return frames[:]


def _read_input(path: Path, args, use_decoder_mvs: bool):
    start_time = max(0.0, float(args.start_time))
    if path.is_dir():
        images = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError(f"No image frames found in directory: {path}")
        start_index = int(round(start_time * float(args.sample_fps or 1.0)))
        images = images[start_index:]
        frames = [Image.open(item).convert("RGB") for item in images]
        if args.max_frames:
            frames = frames[: args.max_frames]
        return frames, 0.0, "image_dir", None

    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return [Image.open(path).convert("RGB")], 0.0, "image_file", None

    if path.suffix.lower() in RAW_EXTENSIONS:
        frames = _raw_frames(path, args.raw_width, args.raw_height, args.raw_format)
        start_index = int(round(start_time * float(args.raw_fps)))
        frames = frames[start_index:]
        if args.max_frames:
            frames = frames[: args.max_frames]
        return frames, args.raw_fps, "raw_frames", None

    if use_decoder_mvs:
        records, native_fps, reader = read_video_records_with_mvs(
            path,
            max_frames=args.max_frames,
            sample_fps=args.sample_fps,
            selection_policy=args.codec_selection_policy,
            start_time=start_time,
        )
        return [record["image"] for record in records if record["selected"]], native_fps, reader, records

    frames, native_fps, reader = read_video_frames(
        path,
        max_frames=args.max_frames,
        sample_fps=args.sample_fps,
        start_time=start_time,
    )
    return frames, native_fps, reader, None


def _quantize(features: torch.Tensor):
    array = features.detach().cpu().numpy()
    f_min, f_max = float(array.min()), float(array.max())
    scale = (f_max - f_min) / 255.0
    if scale == 0.0:
        scale = 1.0
    zero_point = -f_min / scale
    quantized = np.clip(np.round(array / scale + zero_point), 0, 255).astype(np.uint8)
    return quantized, scale, zero_point


def _payload(features: torch.Tensor, modality: str, prompt: str, level=None):
    quantized, scale, zero_point = _quantize(features)
    result = {
        "features": base64.b64encode(quantized.tobytes()).decode("ascii"),
        "scale": scale,
        "zero_point": zero_point,
        "feature_shape": list(quantized.shape),
        "modality": modality,
        "prompt": prompt,
    }
    if level:
        result["payload_tokens"], result["payload_dim"] = level
    return result, int(quantized.nbytes)


PROJECTS = {
    "baseline": ("纯 Qwen Baseline", False, False, False, True),
    "so": ("逐帧 SplitOculo", True, False, False, False),
    "temporal": ("SplitOculo + Qwen 时序融合", True, False, True, False),
    "codec": ("Codec + Qwen 时序融合", True, True, True, False),
}


def _variant_specs(args):
    """Keep project order and duplicates exactly as requested by the UI."""
    names = [item.strip().lower() for item in args.projects.split(",") if item.strip()]
    unknown = [item for item in names if item not in PROJECTS]
    if unknown:
        raise ValueError(f"Unknown project(s): {', '.join(unknown)}")
    if not names:
        raise ValueError("At least one project is required")
    return [PROJECTS[name] for name in names]


@torch.inference_mode()
def _full_backbone_features(encoder, frames):
    image_tensor = torch.stack(
        [encoder.transform(frame.convert("RGB")) for frame in frames], dim=0
    )
    return encoder.student(encoder.input_to_device(image_tensor))[-1]


@torch.inference_mode()
def _temporal_payload(encoder, temporal_fusion, backbone_features):
    if temporal_fusion is None:
        raise ValueError(
            "Temporal demo project requires --temporal_pair_checkpoint"
        )
    if backbone_features.shape[0] % 2:
        backbone_features = torch.cat(
            (backbone_features, backbone_features[-1:]), dim=0
        )
    fused = temporal_fusion(
        backbone_features[0::2], backbone_features[1::2]
    )
    tokens = encoder.projector(fused)
    if encoder.bottleneck is not None:
        tokens = encoder.bottleneck.encode(tokens)
    return tokens


def _encode_variant(
    encoder,
    temporal_fusion,
    path: Path,
    args,
    spatial_acc: bool,
    codec_acc: bool,
    temporal_acc: bool,
):
    read_start = time.perf_counter()
    frames, native_fps, reader, records = _read_input(path, args, codec_acc)
    decode_ms = (time.perf_counter() - read_start) * 1000
    if not frames:
        raise ValueError("Input produced no frames")

    level = parse_payload_levels(args.spatial_level)[0] if spatial_acc else None
    # Temporal-pair training targets the checkpoint's complete 49x64 payload.
    # Arbitrary multilevel truncation was not part of that training objective.
    if temporal_acc and level != (49, 64):
        raise ValueError(
            "Temporal demo currently requires --spatial_level 49x64"
        )
    encode_start = time.perf_counter()
    codec_records = []
    features = []
    source_frame_count = len(frames)

    if codec_acc and records is not None:
        accelerator = DecoderMotionVectorAccelerator(
            encoder,
            flow_impl=args.codec_flow_impl,
            reference_mode=args.codec_reference_mode,
            min_coverage=args.codec_mv_min_coverage,
            max_p_chain=args.codec_max_p_chain,
        )
        for record in records:
            encoded, _, info = accelerator.encode_record(
                record,
                payload_level=None if temporal_acc else level,
                defer_payload=temporal_acc,
            )
            codec_records.append(info)
            if record["selected"]:
                features.append(encoded.squeeze(0).detach())
        if temporal_acc:
            backbone = torch.stack(features, dim=0)
            tensor = _temporal_payload(
                encoder, temporal_fusion, backbone
            )
            features = []
    elif codec_acc and not temporal_acc:
        accelerator = CodecWarpAccelerator(encoder, gop_frames=args.codec_gop_frames)
        for frame in frames:
            encoded, _, info = accelerator.encode_pil(frame, payload_level=level)
            codec_records.append(info)
            features.append(encoded.squeeze(0).detach())
    elif temporal_acc:
        backbone = _full_backbone_features(encoder, frames)
        tensor = _temporal_payload(encoder, temporal_fusion, backbone)
    else:
        if level:
            for frame in frames:
                encoded, _ = encoder.encode_pil_level(frame, level)
                features.append(encoded.squeeze(0).detach())
        else:
            encoded, _ = encoder.encode_pil_batch(frames)
            features.extend(encoded[index].detach() for index in range(encoded.shape[0]))

    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
    encode_ms = (time.perf_counter() - encode_start) * 1000
    if not temporal_acc:
        tensor = (
            features[0].unsqueeze(0)
            if len(features) == 1
            else torch.stack(features, dim=0)
        )
    modality = "image" if source_frame_count == 1 else "video"
    payload, payload_bytes = _payload(
        tensor,
        modality,
        args.prompt,
        None if temporal_acc else level,
    )
    return payload, {
        "modality": modality,
        "decode_ms": decode_ms,
        "encode_ms": encode_ms,
        "payload_bytes": payload_bytes,
        "feature_shape": list(tensor.shape),
        "frames": source_frame_count,
        "temporal_grids": int(tensor.shape[0]),
        "temporal_pair_fusion": bool(temporal_acc),
        "native_fps": native_fps,
        "reader": reader,
        "codec_frames": codec_records,
    }


def _jpeg_frames_payload(
    frames,
    prompt,
    jpeg_quality,
    input_size=224,
    video_fps=2.0,
):
    encoded_frames = []
    binary_bytes = 0
    for frame in frames:
        frame = TF.resize(
            frame.convert("RGB"),
            input_size,
            interpolation=InterpolationMode.BICUBIC,
        )
        frame = TF.center_crop(frame, [input_size, input_size])
        buffer = BytesIO()
        frame.save(buffer, format="JPEG", quality=jpeg_quality)
        value = buffer.getvalue()
        binary_bytes += len(value)
        encoded_frames.append(base64.b64encode(value).decode("ascii"))
    return {
        "frames": encoded_frames,
        "frame_format": "jpeg",
        "prompt": prompt,
        "max_new_tokens": 256,
        "video_pixel_budget": int(input_size * input_size),
        "input_size": int(input_size),
        "video_fps": float(video_fps),
    }, binary_bytes


def _run_pure_qwen(path, args, label, on_text=None):
    started = time.perf_counter()
    read_start = time.perf_counter()
    frames, native_fps, reader, _ = _read_input(path, args, False)
    decode_ms = (time.perf_counter() - read_start) * 1000
    if not frames:
        raise ValueError("Input produced no frames")

    encode_start = time.perf_counter()
    payload, payload_bytes = _jpeg_frames_payload(
        frames,
        args.prompt,
        args.baseline_jpeg_quality,
        args.baseline_input_size,
        args.sample_fps,
    )
    encode_ms = (time.perf_counter() - encode_start) * 1000
    payload_size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    bandwidth_delay_ms = 0.0
    if args.bandwidth_kb_s and args.bandwidth_kb_s > 0:
        bandwidth_delay_ms = payload_size / (args.bandwidth_kb_s * 1024) * 1000
        time.sleep(bandwidth_delay_ms / 1000)

    request_started = time.perf_counter()
    if on_text is not None:
        try:
            cloud = _post_cloud_stream(
                f"{args.server.rstrip('/')}/infer_qwen_stream",
                payload,
                args,
                on_text,
            )
        except RoundInterrupted as exc:
            exc.result.update(
                {
                    "frames": len(frames),
                    "edge_encode_ms": encode_ms,
                    "edge_decode_ms": decode_ms,
                    "payload_bytes": payload_bytes,
                    "request_bytes": payload_size,
                    "upload_delay_ms": bandwidth_delay_ms,
                    "full_response_ms": (time.perf_counter() - started) * 1000,
                }
            )
            raise
        response = None
    else:
        response = requests.post(
            f"{args.server.rstrip('/')}/infer_qwen",
            json=payload,
            headers={"Content-Type": "application/json"},
            auth=(args.username, args.password) if args.username else None,
            timeout=args.timeout,
        )
        _raise_for_cloud_error(response)
        cloud = response.json()
    http_roundtrip_ms = (time.perf_counter() - request_started) * 1000
    cloud_process_ms = cloud.get("cloud_process_ms", cloud.get("latency_ms", 0.0))
    inference_metrics = cloud.get("inference_metrics", {})
    cloud_ttft_ms = cloud.get(
        "cloud_ttft_ms",
        float(cloud.get("inference_metrics", {}).get("ttft_seconds", 0.0)) * 1000,
    )
    network_overhead_ms = max(0.0, http_roundtrip_ms - cloud_process_ms)
    modality = "image" if len(frames) == 1 else "video"
    return {
        "label": label,
        "response": cloud.get("response", ""),
        "pure_qwen": True,
        "spatial_acceleration": False,
        "temporal_redundancy_acceleration": False,
        "edge_encode_ms": encode_ms,
        "edge_decode_ms": decode_ms,
        "cloud_process_ms": cloud_process_ms,
        "cloud_decode_ms": cloud.get("cloud_decode_ms", 0.0),
        "cloud_inference_ms": cloud.get("cloud_inference_ms", cloud_process_ms),
        "http_roundtrip_ms": http_roundtrip_ms,
        "network_overhead_ms": network_overhead_ms,
        "cloud_ttft_ms": cloud_ttft_ms,
        "upload_delay_ms": bandwidth_delay_ms,
        "bandwidth_kb_s": args.bandwidth_kb_s,
        "payload_bytes": payload_bytes,
        "payload_scope": f"{len(frames)} JPEG frame(s) total",
        "payload_per_frame_bytes": payload_bytes / max(len(frames), 1),
        "request_bytes": payload_size,
        "feature_shape": [],
        "native_video_grid_thw": inference_metrics.get("native_grid_thw"),
        "native_visual_tokens": inference_metrics.get("native_visual_tokens"),
        "frames": len(frames),
        "sample_fps": args.sample_fps if modality == "video" else None,
        "sampled_prefix_seconds": (
            len(frames) / args.sample_fps
            if modality == "video" and args.sample_fps and args.sample_fps > 0
            else 0.0
        ),
        "native_fps": native_fps,
        "reader": reader,
        "codec_cnn_frames": 0,
        "codec_warp_frames": 0,
        "codec_source_frames_processed": len(frames),
        "end_to_end_ttft_ms": (
            decode_ms
            + encode_ms
            + bandwidth_delay_ms
            + network_overhead_ms
            + cloud.get("cloud_decode_ms", 0.0)
            + cloud_ttft_ms
        ),
        "ttft_without_network_ms": (
            decode_ms
            + encode_ms
            + cloud.get("cloud_decode_ms", 0.0)
            + cloud_ttft_ms
        ),
        "full_response_ms": (time.perf_counter() - started) * 1000,
    }


def _run_variant(
    encoder,
    temporal_fusion,
    path,
    args,
    label,
    spatial_acc,
    codec_acc,
    temporal_acc,
    pure_qwen,
    on_text=None,
):
    if pure_qwen:
        return _run_pure_qwen(path, args, label, on_text=on_text)
    if encoder is None:
        raise ValueError("SplitOculo projects require --edge_checkpoint")

    started = time.perf_counter()
    payload, edge_metrics = _encode_variant(
        encoder,
        temporal_fusion,
        path,
        args,
        spatial_acc,
        codec_acc,
        temporal_acc,
    )
    modality = edge_metrics["modality"]
    payload_size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    bandwidth_delay_ms = 0.0
    if args.bandwidth_kb_s and args.bandwidth_kb_s > 0:
        bandwidth_delay_ms = payload_size / (args.bandwidth_kb_s * 1024) * 1000
        time.sleep(bandwidth_delay_ms / 1000)

    request_started = time.perf_counter()
    if on_text is not None:
        try:
            cloud = _post_cloud_stream(
                f"{args.server.rstrip('/')}/infer_stream",
                payload,
                args,
                on_text,
            )
        except RoundInterrupted as exc:
            exc.result.update(
                {
                    "frames": edge_metrics["frames"],
                    "edge_encode_ms": edge_metrics["encode_ms"],
                    "edge_decode_ms": edge_metrics["decode_ms"],
                    "payload_bytes": edge_metrics["payload_bytes"],
                    "request_bytes": payload_size,
                    "upload_delay_ms": bandwidth_delay_ms,
                    "full_response_ms": (time.perf_counter() - started) * 1000,
                }
            )
            raise
        response = None
    else:
        response = requests.post(
            f"{args.server.rstrip('/')}/infer",
            json=payload,
            headers={"Content-Type": "application/json"},
            auth=(args.username, args.password) if args.username else None,
            timeout=args.timeout,
        )
        _raise_for_cloud_error(response)
        cloud = response.json()
    http_roundtrip_ms = (time.perf_counter() - request_started) * 1000
    full_response_ms = (time.perf_counter() - started) * 1000
    cloud_process_ms = cloud.get("cloud_process_ms", cloud.get("latency_ms", 0.0))
    cloud_ttft_ms = cloud.get("cloud_ttft_ms", float(cloud.get("inference_metrics", {}).get("ttft_seconds", 0.0)) * 1000)
    network_overhead_ms = max(0.0, http_roundtrip_ms - cloud_process_ms)
    end_to_end_ttft_ms = (
        edge_metrics["decode_ms"]
        + edge_metrics["encode_ms"]
        + bandwidth_delay_ms
        + network_overhead_ms
        + cloud.get("cloud_decode_ms", 0.0)
        + cloud_ttft_ms
    )
    return {
        "label": label,
        "response": cloud.get("response", ""),
        "pure_qwen": False,
        "spatial_acceleration": bool(spatial_acc),
        "temporal_redundancy_acceleration": bool(codec_acc),
        "temporal_pair_fusion": bool(temporal_acc),
        "edge_encode_ms": edge_metrics["encode_ms"],
        "edge_decode_ms": edge_metrics["decode_ms"],
        "cloud_process_ms": cloud_process_ms,
        "cloud_decode_ms": cloud.get("cloud_decode_ms", 0.0),
        "cloud_inference_ms": cloud.get("cloud_inference_ms", cloud.get("latency_ms", 0.0)),
        "http_roundtrip_ms": http_roundtrip_ms,
        "network_overhead_ms": network_overhead_ms,
        "cloud_ttft_ms": cloud_ttft_ms,
        "upload_delay_ms": bandwidth_delay_ms,
        "bandwidth_kb_s": args.bandwidth_kb_s,
        "payload_bytes": edge_metrics["payload_bytes"],
        "payload_scope": (
            f"{edge_metrics['temporal_grids']} temporal grid(s) from "
            f"{edge_metrics['frames']} frame(s)"
        ),
        "payload_per_frame_bytes": edge_metrics["payload_bytes"] / max(edge_metrics["frames"], 1),
        "request_bytes": payload_size,
        "feature_shape": edge_metrics["feature_shape"],
        "frames": edge_metrics["frames"],
        "temporal_grids": edge_metrics["temporal_grids"],
        "sample_fps": args.sample_fps if modality == "video" else None,
        "sampled_prefix_seconds": (
            edge_metrics["frames"] / args.sample_fps
            if modality == "video" and args.sample_fps and args.sample_fps > 0
            else 0.0
        ),
        "native_fps": edge_metrics["native_fps"],
        "reader": edge_metrics["reader"],
        "codec_cnn_frames": sum(item.get("cnn_executed", False) for item in edge_metrics["codec_frames"]),
        "codec_warp_frames": sum(item.get("warp_executed", False) for item in edge_metrics["codec_frames"]),
        "codec_source_frames_processed": len(edge_metrics["codec_frames"]) if codec_acc else edge_metrics["frames"],
        "codec_selected_frame_types": [
            item.get("codec_frame_type")
            for item in edge_metrics["codec_frames"]
            if item.get("selected")
        ],
        "codec_selected_source_indices": [
            item.get("source_index")
            for item in edge_metrics["codec_frames"]
            if item.get("selected")
        ],
        "codec_output_timestamps": [
            item.get("time_seconds")
            for item in edge_metrics["codec_frames"]
            if item.get("selected")
        ],
        "codec_processing_realtime_factor": (
            edge_metrics["encode_ms"]
            / max(
                1.0,
                (
                    max(
                        (
                            item.get("time_seconds") or 0.0
                            for item in edge_metrics["codec_frames"]
                        ),
                        default=0.0,
                    )
                    * 1000
                ),
            )
            if codec_acc
            else None
        ),
        "codec_total_realtime_factor": (
            (edge_metrics["decode_ms"] + edge_metrics["encode_ms"])
            / max(
                1.0,
                (
                    max(
                        (
                            item.get("time_seconds") or 0.0
                            for item in edge_metrics["codec_frames"]
                        ),
                        default=0.0,
                    )
                    * 1000
                ),
            )
            if codec_acc
            else None
        ),
        "end_to_end_ttft_ms": end_to_end_ttft_ms,
        "ttft_without_network_ms": (
            end_to_end_ttft_ms - network_overhead_ms - bandwidth_delay_ms
        ),
        "full_response_ms": full_response_ms,
    }


def main():
    parser = argparse.ArgumentParser(description="SplitOculo Electron demo client")
    parser.add_argument("--input", required=True)
    parser.add_argument("--edge_checkpoint", default=None)
    parser.add_argument(
        "--cloud_checkpoint",
        default=None,
        help="Optional checkpoint path/URL visible to the cloud server; load before SplitOculo requests",
    )
    parser.add_argument("--temporal_pair_checkpoint", default=None)
    parser.add_argument("--server", default="http://localhost:8080")
    parser.add_argument("--username", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--projects",
        default="baseline",
        help="Comma-separated project ids: baseline,so,temporal,codec",
    )
    parser.add_argument("--bandwidth_kb_s", type=float, default=0.0)
    parser.add_argument("--spatial_level", default="49x64")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--sample_fps", type=float, default=2.0)
    parser.add_argument("--start_time", type=float, default=0.0)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--round_step_seconds", type=float, default=2.0)
    parser.add_argument("--interrupt_after_seconds", type=float, default=None)
    parser.add_argument("--codec_flow_impl", choices=("feature_grid", "feature_grid_center", "dense"), default="feature_grid")
    parser.add_argument(
        "--codec_selection_policy",
        choices=("fixed", "best_effort_ip"),
        default="best_effort_ip",
    )
    parser.add_argument("--codec_reference_mode", choices=("recursive", "keyframe"), default="recursive")
    parser.add_argument("--codec_mv_min_coverage", type=float, default=0.0)
    parser.add_argument("--codec_max_p_chain", type=int, default=0)
    parser.add_argument("--codec_gop_frames", type=int, default=4)
    parser.add_argument("--raw_width", type=int, default=224)
    parser.add_argument("--raw_height", type=int, default=224)
    parser.add_argument("--raw_fps", type=float, default=10.0)
    parser.add_argument("--raw_format", choices=("rgb24", "bgr24", "gray8"), default="rgb24")
    parser.add_argument("--baseline_jpeg_quality", type=int, default=90)
    parser.add_argument("--baseline_input_size", type=int, default=224)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not 1 <= args.baseline_jpeg_quality <= 100:
        raise ValueError("--baseline_jpeg_quality must be in [1, 100]")
    if args.baseline_input_size <= 0:
        raise ValueError("--baseline_input_size must be positive")
    if args.rounds <= 0 or args.round_step_seconds <= 0:
        raise ValueError("--rounds and --round_step_seconds must be positive")
    if args.interrupt_after_seconds is not None and args.interrupt_after_seconds <= 0:
        raise ValueError("--interrupt_after_seconds must be positive")
    path = Path(args.input)
    project_names = [
        item.strip().lower() for item in args.projects.split(",") if item.strip()
    ]
    project_specs = _variant_specs(args)
    if args.cloud_checkpoint and any(not spec[4] for spec in project_specs):
        try:
            checkpoint_result = load_cloud_checkpoint(
                args.server,
                args.cloud_checkpoint,
                timeout=args.timeout,
                username=args.username,
                password=args.password,
            )
            print(
                "Cloud checkpoint ready: "
                f"{checkpoint_result.get('checkpoint_path', args.cloud_checkpoint)}",
                flush=True,
            )
        except Exception as exc:
            print(f"Cloud checkpoint load failed: {exc}", file=sys.stderr, flush=True)
            return 1
    encoder = None
    temporal_fusion = None
    temporal_metadata = None
    load_ms = 0.0
    results = []
    initial_start_time = float(args.start_time)
    for project, spec in zip(project_names, project_specs):
        label, spatial_acc, codec_acc, temporal_acc, pure_qwen = spec
        try:
            if not pure_qwen and encoder is None:
                if not args.edge_checkpoint:
                    raise ValueError(
                        "SplitOculo projects require --edge_checkpoint"
                    )
                load_started = time.perf_counter()
                encoder = EdgeEncoder(
                    args.edge_checkpoint, device=args.device
                )
                load_ms += (time.perf_counter() - load_started) * 1000
            if temporal_acc and temporal_fusion is None:
                if not args.temporal_pair_checkpoint:
                    raise ValueError(
                        "Temporal demo project requires "
                        "--temporal_pair_checkpoint"
                    )
                load_started = time.perf_counter()
                temporal_fusion, temporal_metadata = (
                    load_temporal_pair_fusion(
                        args.temporal_pair_checkpoint,
                        device=args.device,
                    )
                )
                temporal_fusion.eval()
                load_ms += (time.perf_counter() - load_started) * 1000
            schedule_started = time.monotonic()
            for round_index in range(1, args.rounds + 1):
                scheduled_at = (
                    schedule_started + (round_index - 1) * args.round_step_seconds
                )
                remaining = scheduled_at - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                elif round_index > 1 and -remaining >= 0.01:
                    print(
                        f"{label} · 第 {round_index}/{args.rounds} 轮"
                        f"已落后采样 {-remaining:.2f} 秒",
                        flush=True,
                    )
                window_start = (
                    initial_start_time
                    + (round_index - 1) * args.round_step_seconds
                )
                args.start_time = window_start
                args.current_interrupt_after_seconds = (
                    args.interrupt_after_seconds
                    if round_index < args.rounds
                    else None
                )
                args.current_interrupt_deadline = (
                    schedule_started + round_index * args.round_step_seconds
                    if round_index < args.rounds
                    and args.interrupt_after_seconds is not None
                    else None
                )
                event_context = {
                    "project": project,
                    "label": label,
                    "round": round_index,
                    "rounds": args.rounds,
                    "window_start_seconds": window_start,
                }
                try:
                    if args.stream:
                        print(
                            "DEMO_STREAM_START="
                            + json.dumps(event_context, ensure_ascii=False),
                            flush=True,
                        )

                    def _emit_text(text):
                        print(
                            "DEMO_STREAM_DELTA="
                            + json.dumps(
                                {**event_context, "text": text},
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )

                    row = _run_variant(
                        encoder,
                        temporal_fusion,
                        path,
                        args,
                        label,
                        spatial_acc,
                        codec_acc,
                        temporal_acc,
                        pure_qwen,
                        on_text=_emit_text if args.stream else None,
                    )
                    if temporal_acc:
                        row["temporal_checkpoint"] = args.temporal_pair_checkpoint
                        row["temporal_patch_size"] = int(
                            temporal_metadata.get("temporal_patch_size", 2)
                        )
                    row.update(event_context)
                    results.append(row)
                    print(
                        "DEMO_RESULT_ITEM="
                        + json.dumps(row, ensure_ascii=False),
                        flush=True,
                    )
                except RoundInterrupted as exc:
                    row = {
                        **exc.result,
                        **event_context,
                        "pure_qwen": bool(pure_qwen),
                        "spatial_acceleration": bool(spatial_acc),
                        "temporal_redundancy_acceleration": bool(codec_acc),
                        "temporal_pair_fusion": bool(temporal_acc),
                        "interrupted": True,
                        "response": exc.partial_text,
                    }
                    results.append(row)
                    print(
                        "DEMO_RESULT_ITEM="
                        + json.dumps(row, ensure_ascii=False),
                        flush=True,
                    )
                except Exception as exc:
                    row = {
                        **event_context,
                        "pure_qwen": bool(pure_qwen),
                        "spatial_acceleration": bool(spatial_acc),
                        "temporal_redundancy_acceleration": bool(codec_acc),
                        "temporal_pair_fusion": bool(temporal_acc),
                        "error": str(exc),
                    }
                    results.append(row)
                    print(
                        "DEMO_RESULT_ITEM="
                        + json.dumps(row, ensure_ascii=False),
                        flush=True,
                    )
        except Exception as exc:
            row = {
                "project": project,
                "label": label,
                "round": 1,
                "rounds": args.rounds,
                "window_start_seconds": initial_start_time,
                "pure_qwen": bool(pure_qwen),
                "spatial_acceleration": bool(spatial_acc),
                "temporal_redundancy_acceleration": bool(codec_acc),
                "temporal_pair_fusion": bool(temporal_acc),
                "error": str(exc),
            }
            results.append(row)
            print("DEMO_RESULT_ITEM=" + json.dumps(row, ensure_ascii=False), flush=True)
    print("DEMO_RESULT_JSON=" + json.dumps({"model_load_ms": load_ms, "results": results}, ensure_ascii=False))
    if any("error" in item for item in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

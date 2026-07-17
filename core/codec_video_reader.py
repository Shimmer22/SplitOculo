"""PyAV video reader that preserves decoder motion-vector side data."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _uniform_indices(total_frames, max_frames):
    if total_frames <= 0:
        return []
    if max_frames is None or total_frames <= max_frames:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, max_frames).round().astype(int).tolist()


def _sample_indices(total_frames, native_fps, sample_fps, max_frames):
    if not sample_fps or not native_fps or native_fps <= 0:
        return _uniform_indices(total_frames, max_frames)
    step = max(1, int(round(native_fps / sample_fps)))
    indices = list(range(0, total_frames, step))
    if max_frames is not None:
        indices = indices[:max_frames]
    return indices


def _motion_vectors(frame):
    for side_data in frame.side_data:
        if "MOTION" in str(side_data.type).upper():
            return side_data.to_ndarray().copy()
    return None


def _picture_type(frame):
    value = int(frame.pict_type)
    return {
        0: "NONE",
        1: "I",
        2: "P",
        3: "B",
        4: "S",
        5: "SI",
        6: "SP",
        7: "BI",
    }.get(value, f"UNKNOWN_{value}")


def read_video_records_with_mvs(video_path, max_frames=None, sample_fps=None):
    """Decode through the last sampled frame and retain AVMotionVector arrays.

    All source frames are returned because reference feature state must advance
    through intervening codec frames even when only sparse frames enter the VLM.
    """
    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            "Decoder-MV mode requires PyAV; install the 'av' package"
        ) from exc

    video_path = Path(video_path)
    if not video_path.is_file():
        raise ValueError("Decoder-MV mode requires a compressed video file")

    container = av.open(str(video_path))
    if not container.streams.video:
        container.close()
        raise RuntimeError(f"No video stream found: {video_path}")
    stream = container.streams.video[0]
    options = dict(stream.codec_context.options or {})
    options["flags2"] = "+export_mvs"
    stream.codec_context.options = options

    native_fps = float(stream.average_rate or 0.0)
    total_frames = int(stream.frames or 0)
    if total_frames <= 0:
        # Decode count once so sparse/uniform selection has an exact endpoint.
        container.close()
        probe = av.open(str(video_path))
        total_frames = sum(1 for _ in probe.decode(video=0))
        probe.close()
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        options = dict(stream.codec_context.options or {})
        options["flags2"] = "+export_mvs"
        stream.codec_context.options = options

    selected_indices = _sample_indices(total_frames, native_fps, sample_fps, max_frames)
    if not selected_indices:
        container.close()
        raise RuntimeError(f"No sample indices selected from {video_path}")
    selected = set(selected_indices)
    last_selected = selected_indices[-1]

    records = []
    for source_index, frame in enumerate(container.decode(stream)):
        if source_index > last_selected:
            break
        records.append({
            "source_index": source_index,
            "pts": int(frame.pts) if frame.pts is not None else None,
            "time_seconds": float(frame.time) if frame.time is not None else None,
            "pict_type": _picture_type(frame),
            "selected": source_index in selected,
            "image": frame.to_image().convert("RGB"),
            "motion_vectors": _motion_vectors(frame),
        })
    container.close()

    decoded_selected = [record["source_index"] for record in records if record["selected"]]
    if decoded_selected != selected_indices[: len(decoded_selected)] or len(decoded_selected) != len(selected_indices):
        raise RuntimeError(
            f"Decoded samples {decoded_selected} do not match requested indices {selected_indices}"
        )
    return records, native_fps, "pyav_export_mvs"

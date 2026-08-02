"""Read video frames via ffmpeg pipe (avoids numpy/opencv version conflicts)."""
import subprocess
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import io


def read_video_ffmpeg(video_path, max_frames=None):
    """Pipe video through ffmpeg and yield PIL frames."""
    video_path = str(Path(video_path).resolve())

    # Get video info
    probe_cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_frames,r_frame_rate,duration',
        '-of', 'default=noprint_wrappers=1', video_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    info = {}
    for line in result.stdout.strip().split('\n'):
        if '=' in line:
            k, v = line.split('=', 1)
            info[k] = v

    num, denom = info.get('r_frame_rate', '30/1').split('/')
    fps = float(num) / float(denom)
    total_frames = int(info.get('nb_frames', 0))

    # Build ffmpeg command to output raw RGB frames
    cmd = [
        'ffmpeg', '-v', 'error', '-i', video_path,
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Probe for video dimensions
    probe_cmd2 = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'default=noprint_wrappers=1', video_path
    ]
    r2 = subprocess.run(probe_cmd2, capture_output=True, text=True)
    w = h = 0
    for line in r2.stdout.strip().split('\n'):
        if line.startswith('width='):
            w = int(line.split('=')[1])
        if line.startswith('height='):
            h = int(line.split('=')[1])

    frame_size = w * h * 3
    frames = []
    frame_idx = 0
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        if max_frames is not None and len(frames) >= max_frames:
            proc.kill()
            break
        img = Image.frombytes('RGB', (w, h), raw)
        frames.append(img)
        frame_idx += 1

    proc.wait()
    return frames, fps, f'ffmpeg_pipe ({len(frames)} frames)'

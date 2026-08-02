#!/usr/bin/env python3
"""Record GPU utilization while a training process is running.

Examples:

    python scripts/monitor_gpu.py --pid 12345 \
        --interval 2 --output checkpoints/gan_dynamic_49x64/gpu.csv

The monitor intentionally uses ``nvidia-smi`` instead of polling through
PyTorch, so it can observe the training process from a separate process and
does not affect CUDA context or allocator state.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from pathlib import Path


QUERY_FIELDS = (
    "timestamp,index,name,memory.used,memory.total,"
    "utilization.gpu,utilization.memory,temperature.gpu,power.draw"
)
OUTPUT_FIELDS = (
    "sample_time",
    "elapsed_seconds",
    "pid",
    "pid_alive",
    "timestamp",
    "index",
    "name",
    "memory_used_mib",
    "memory_total_mib",
    "utilization_gpu_percent",
    "utilization_memory_percent",
    "temperature_c",
    "power_w",
)


def _pid_alive(pid: int | None) -> bool:
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _query_gpus() -> list[dict[str, str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={QUERY_FIELDS}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    rows = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(QUERY_FIELDS.split(",")):
            continue
        rows.append(dict(zip(QUERY_FIELDS.split(","), values)))
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPU rows")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="stop after this process exits; omit to monitor until Ctrl-C",
    )
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/gpu_monitor.csv"))
    parser.add_argument("--once", action="store_true", help="write one sample and exit")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    if args.pid is not None and args.pid <= 0:
        parser.error("--pid must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.output.exists() or args.output.stat().st_size == 0
    started = time.monotonic()

    with args.output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        if new_file:
            writer.writeheader()

        while True:
            alive = _pid_alive(args.pid)
            sample_time = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            try:
                gpu_rows = _query_gpus()
            except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                print(f"GPU query failed: {exc}", flush=True)
                gpu_rows = []

            for gpu in gpu_rows:
                row = {
                    "sample_time": sample_time,
                    "elapsed_seconds": f"{time.monotonic() - started:.3f}",
                    "pid": "" if args.pid is None else str(args.pid),
                    "pid_alive": str(alive),
                    "timestamp": gpu["timestamp"],
                    "index": gpu["index"],
                    "name": gpu["name"],
                    "memory_used_mib": gpu["memory.used"],
                    "memory_total_mib": gpu["memory.total"],
                    "utilization_gpu_percent": gpu["utilization.gpu"],
                    "utilization_memory_percent": gpu["utilization.memory"],
                    "temperature_c": gpu["temperature.gpu"],
                    "power_w": gpu["power.draw"],
                }
                writer.writerow(row)
                print(
                    f"gpu={gpu['index']} util={gpu['utilization.gpu']}% "
                    f"mem={gpu['memory.used']}/{gpu['memory.total']} MiB "
                    f"power={gpu['power.draw']} W",
                    flush=True,
                )
            handle.flush()

            if args.once or (args.pid is not None and not alive):
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()

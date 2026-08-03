"""Run the complete SplitOculo demo from an SSH terminal.

The command keeps the existing edge/cloud HTTP boundary, but both processes run
on the same machine over localhost.  It starts the cloud service when needed,
streams one readable result per demo variant, and only stops services that it
started itself.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any
import unicodedata

import requests


ROOT = Path(__file__).resolve().parents[1]
PROJECT_IDS = ("baseline", "so", "temporal", "codec")
RESULT_ITEM_PREFIX = "DEMO_RESULT_ITEM="
RESULT_JSON_PREFIX = "DEMO_RESULT_JSON="
STREAM_START_PREFIX = "DEMO_STREAM_START="
STREAM_DELTA_PREFIX = "DEMO_STREAM_DELTA="
REQUIRED_HEALTH_FIELDS = {
    "qwen_model_name",
    "checkpoint_path",
    "checkpoint_hidden_size",
    "checkpoint_bottleneck_dim",
    "checkpoint_transmission_tokens",
    "checkpoint_target_tokens",
}
CONFIG_FIELDS = (
    "input",
    "cloud_checkpoint",
    "edge_checkpoint",
    "temporal_pair_checkpoint",
    "qwen_path",
    "projects",
    "prompt",
    "device",
    "port",
    "timeout",
    "startup_timeout",
    "offline",
    "bandwidth_kb_s",
    "spatial_level",
    "max_frames",
    "sample_fps",
    "rounds",
    "round_step_seconds",
    "interrupt_on_next_round",
    "codec_flow_impl",
    "codec_selection_policy",
    "codec_reference_mode",
    "codec_mv_min_coverage",
    "codec_max_p_chain",
    "codec_gop_frames",
    "raw_width",
    "raw_height",
    "raw_fps",
    "raw_format",
    "baseline_jpeg_quality",
    "baseline_input_size",
    "username",
)


def config_path() -> Path:
    override = os.environ.get("SPLITOCULO_DEMO_CONFIG")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "splitoculo" / "terminal_demo.json"


def model_profiles() -> dict[str, dict[str, str]]:
    checkpoint_root = ROOT / "checkpoints"
    local_split = checkpoint_root / "qwen_vit_h1280_layer4_224_b64_t256" / "split_imported"
    server_split = checkpoint_root / "llava558k_32b_49x64_rolling" / "split"
    return {
        "3b": {
            "qwen_path": "Qwen/Qwen2.5-VL-3B-Instruct",
            "cloud_checkpoint": str(local_split / "cloud_weights.pth"),
            "edge_checkpoint": str(local_split / "edge_weights.pth"),
            "temporal_pair_checkpoint": str(
                checkpoint_root / "temporal_pair_ucf101" / "temporal_pair_best.pth"
            ),
        },
        "32b": {
            "qwen_path": "Qwen/Qwen2.5-VL-32B-Instruct",
            "cloud_checkpoint": str(server_split / "cloud_weights.pth"),
            "edge_checkpoint": str(server_split / "edge_weights.pth"),
            "temporal_pair_checkpoint": str(
                checkpoint_root / "llava558k_Qwen32B" / "temporal_pair_best.pth"
            ),
        },
    }


def default_config() -> dict[str, Any]:
    profiles = model_profiles()
    profile = profiles["3b"]
    if not Path(profile["cloud_checkpoint"]).is_file():
        profile = profiles["32b"]
    return {
        "input": "",
        **profile,
        "projects": "baseline,so,temporal",
        "prompt": "请简短地描述视频。",
        "device": "cuda",
        "port": 8080,
        "timeout": 300,
        "startup_timeout": 1200,
        "offline": True,
        "bandwidth_kb_s": 0.0,
        "spatial_level": "49x64",
        "max_frames": 8,
        "sample_fps": 2.0,
        "rounds": 1,
        "round_step_seconds": 2.0,
        "interrupt_on_next_round": False,
        "codec_flow_impl": "feature_grid",
        "codec_selection_policy": "best_effort_ip",
        "codec_reference_mode": "recursive",
        "codec_mv_min_coverage": 0.0,
        "codec_max_p_chain": 0,
        "codec_gop_frames": 4,
        "raw_width": 224,
        "raw_height": 224,
        "raw_fps": 10.0,
        "raw_format": "rgb24",
        "baseline_jpeg_quality": 90,
        "baseline_input_size": 224,
        "username": None,
    }


def load_config(path: Path | None = None) -> dict[str, Any]:
    values = default_config()
    target = path or config_path()
    try:
        loaded = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return values
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot read saved config {target}: {exc}") from exc
    if isinstance(loaded, dict):
        values.update({key: loaded[key] for key in CONFIG_FIELDS if key in loaded})
    return values


def save_config(args: argparse.Namespace, path: Path | None = None) -> Path:
    target = path or config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    values = {key: getattr(args, key) for key in CONFIG_FIELDS}
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(target)
    return target


def build_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    defaults = {**default_config(), **(defaults or {})}
    parser = argparse.ArgumentParser(
        description="SplitOculo localhost terminal demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--interactive", action="store_true", help="Open the arrow-key terminal menu")
    parser.add_argument("--input", default=defaults["input"], help="Image, video, frame directory, or raw file")
    parser.add_argument("--cloud-checkpoint", default=defaults["cloud_checkpoint"])
    parser.add_argument("--edge-checkpoint", default=defaults["edge_checkpoint"])
    parser.add_argument("--temporal-pair-checkpoint", default=defaults["temporal_pair_checkpoint"])
    parser.add_argument("--qwen-path", default=defaults["qwen_path"])
    parser.add_argument("--projects", default=defaults["projects"])
    parser.add_argument("--prompt", default=defaults["prompt"])
    parser.add_argument("--device", default=defaults["device"])
    parser.add_argument("--port", type=int, default=defaults["port"])
    parser.add_argument("--timeout", type=int, default=defaults["timeout"], help="Inference request timeout in seconds")
    parser.add_argument("--startup-timeout", type=int, default=defaults["startup_timeout"])
    parser.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=defaults["offline"],
        help="Only use locally cached Qwen files",
    )
    parser.add_argument("--bandwidth-kb-s", type=float, default=defaults["bandwidth_kb_s"])
    parser.add_argument("--spatial-level", default=defaults["spatial_level"])
    parser.add_argument("--max-frames", type=int, default=defaults["max_frames"])
    parser.add_argument("--sample-fps", type=float, default=defaults["sample_fps"])
    parser.add_argument("--rounds", type=int, default=defaults["rounds"], help="Total sliding-window inference rounds")
    parser.add_argument(
        "--round-step-seconds",
        type=float,
        default=defaults["round_step_seconds"],
        help="Window start-time increment and round cadence in seconds",
    )
    parser.add_argument(
        "--interrupt-on-next-round",
        action=argparse.BooleanOptionalAction,
        default=defaults["interrupt_on_next_round"],
        help="Interrupt an unfinished round when the next sampling time arrives",
    )
    parser.add_argument(
        "--codec-flow-impl",
        choices=("feature_grid", "feature_grid_center", "dense"),
        default=defaults["codec_flow_impl"],
    )
    parser.add_argument(
        "--codec-selection-policy",
        choices=("fixed", "best_effort_ip"),
        default=defaults["codec_selection_policy"],
    )
    parser.add_argument(
        "--codec-reference-mode",
        choices=("recursive", "keyframe"),
        default=defaults["codec_reference_mode"],
    )
    parser.add_argument("--codec-mv-min-coverage", type=float, default=defaults["codec_mv_min_coverage"])
    parser.add_argument("--codec-max-p-chain", type=int, default=defaults["codec_max_p_chain"])
    parser.add_argument("--codec-gop-frames", type=int, default=defaults["codec_gop_frames"])
    parser.add_argument("--raw-width", type=int, default=defaults["raw_width"])
    parser.add_argument("--raw-height", type=int, default=defaults["raw_height"])
    parser.add_argument("--raw-fps", type=float, default=defaults["raw_fps"])
    parser.add_argument(
        "--raw-format", choices=("rgb24", "bgr24", "gray8"), default=defaults["raw_format"]
    )
    parser.add_argument("--baseline-jpeg-quality", type=int, default=defaults["baseline_jpeg_quality"])
    parser.add_argument("--baseline-input-size", type=int, default=defaults["baseline_input_size"])
    parser.add_argument("--username", default=defaults["username"], help="Optional HTTP Basic Auth username")
    parser.add_argument("--password", help="Optional HTTP Basic Auth password")
    return parser


def _clear_screen() -> None:
    if not sys.stdout.isatty():
        print()
        return
    if os.name == "nt":
        os.system("cls")
    else:
        print("\033[2J\033[H", end="", flush=True)


def _read_key() -> str:
    if os.name == "nt":
        import msvcrt

        value = msvcrt.getwch()
        if value in {"\x00", "\xe0"}:
            return {"H": "up", "P": "down"}.get(msvcrt.getwch(), "")
        return {"\r": "enter", " ": "space"}.get(value, value.lower())

    import termios
    import tty

    descriptor = sys.stdin.fileno()
    previous = termios.tcgetattr(descriptor)
    try:
        tty.setraw(descriptor)
        value = sys.stdin.read(1)
        if value == "\x1b":
            suffix = sys.stdin.read(2)
            return {"[A": "up", "[B": "down"}.get(suffix, "")
        return {"\r": "enter", "\n": "enter", " ": "space"}.get(
            value, value.lower()
        )
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)


def _select(title: str, options: list[str], selected: int = 0) -> int | None:
    if not sys.stdin.isatty():
        print(title)
        for index, option in enumerate(options, 1):
            print(f"  {index}. {option}")
        raw = input("选择编号（q 退出）: ").strip().lower()
        if raw == "q":
            return None
        return max(0, min(len(options) - 1, int(raw) - 1))

    selected = max(0, min(len(options) - 1, selected))
    while True:
        _clear_screen()
        print(f"{title}\n")
        for index, option in enumerate(options):
            if index == selected:
                print(f"  \033[7m> {option}\033[0m")
            else:
                print(f"    {option}")
        print("\n↑/↓ 移动  Enter 选择  q 返回", flush=True)
        key = _read_key()
        if key == "up":
            selected = (selected - 1) % len(options)
        elif key == "down":
            selected = (selected + 1) % len(options)
        elif key == "enter":
            return selected
        elif key == "q":
            return None


def _multi_select_projects(current: str) -> str:
    selected = set(parse_projects(current))
    cursor = 0
    labels = {
        "baseline": "纯 Qwen Baseline",
        "so": "逐帧端云协同",
        "temporal": "Qwen 时序融合",
        "codec": "Codec + 时序融合",
    }
    if not sys.stdin.isatty():
        raw = input(f"项目（逗号分隔）[{current}]: ").strip()
        return ",".join(parse_projects(raw or current))
    while True:
        _clear_screen()
        print("选择运行项目\n")
        for index, project in enumerate(PROJECT_IDS):
            mark = "x" if project in selected else " "
            line = f"[{mark}] {labels[project]}"
            if index == cursor:
                print(f"  \033[7m> {line}\033[0m")
            else:
                print(f"    {line}")
        print("\n↑/↓ 移动  Space 勾选  Enter 保存  q 取消", flush=True)
        key = _read_key()
        if key == "up":
            cursor = (cursor - 1) % len(PROJECT_IDS)
        elif key == "down":
            cursor = (cursor + 1) % len(PROJECT_IDS)
        elif key == "space":
            project = PROJECT_IDS[cursor]
            if project in selected:
                selected.remove(project)
            else:
                selected.add(project)
        elif key == "enter" and selected:
            return ",".join(project for project in PROJECT_IDS if project in selected)
        elif key == "q":
            return current


def _input_value(label: str, current: Any, cast=str) -> Any:
    _clear_screen()
    print(f"{label}\n当前值: {current}\n")
    raw = input("新值（直接回车保持，输入 - 清空）: ").strip()
    if not raw:
        return current
    if raw == "-":
        return "" if cast is str else current
    try:
        return cast(raw)
    except ValueError:
        input("输入格式无效，按 Enter 返回。")
        return current


def _short(value: Any, width: int = 58) -> str:
    text = str(value or "未设置")
    return text if len(text) <= width else "..." + text[-(width - 3) :]


def _settings_menu(args: argparse.Namespace) -> None:
    cursor = 0
    while True:
        options = [
            f"最大帧数                 {args.max_frames}",
            f"采样 FPS                 {args.sample_fps}",
            f"推理轮数                 {args.rounds}",
            f"每轮窗口滑动             {args.round_step_seconds:g} 秒",
            f"到点中断上一轮           {'是' if args.interrupt_on_next_round else '否'}",
            f"网络时延模拟              {_network_label(args.bandwidth_kb_s)}",
            f"离线模型                  {'是' if args.offline else '否'}",
            f"设备                      {args.device}",
            f"端口                      {args.port}",
            "返回",
        ]
        choice = _select("运行设置", options, cursor)
        if choice is None or choice == 9:
            return
        cursor = choice
        if choice == 0:
            args.max_frames = _input_value("最大帧数", args.max_frames, int)
        elif choice == 1:
            args.sample_fps = _input_value("采样 FPS", args.sample_fps, float)
        elif choice == 2:
            args.rounds = _input_value("推理轮数", args.rounds, int)
        elif choice == 3:
            args.round_step_seconds = _input_value(
                "每轮窗口滑动秒数", args.round_step_seconds, float
            )
        elif choice == 4:
            args.interrupt_on_next_round = not args.interrupt_on_next_round
        elif choice == 5:
            _network_menu(args)
        elif choice == 6:
            args.offline = not args.offline
        elif choice == 7:
            args.device = ["cuda", "cpu"][_select("选择设备", ["CUDA", "CPU"]) or 0]
        elif choice == 8:
            args.port = _input_value("localhost 端口", args.port, int)


def _network_label(value: float) -> str:
    presets = {
        0.0: "关闭",
        62.5: "BLE 62.5 KB/s",
        125.0: "BLE 125 KB/s",
        1000.0: "1 MB/s",
    }
    return presets.get(float(value), f"自定义 {value:g} KB/s")


def _network_menu(args: argparse.Namespace) -> None:
    options = ["关闭", "BLE · 62.5 KB/s", "BLE · 125 KB/s", "1 MB/s", "自定义"]
    choice = _select("网络/上传时延模拟", options)
    if choice is None:
        return
    if choice < 4:
        args.bandwidth_kb_s = (0.0, 62.5, 125.0, 1000.0)[choice]
    else:
        args.bandwidth_kb_s = _input_value(
            "自定义带宽 KB/s", args.bandwidth_kb_s, float
        )


def _model_menu(args: argparse.Namespace) -> None:
    choice = _select("选择 Qwen 模型", ["Qwen2.5-VL 3B", "Qwen2.5-VL 32B", "自定义"])
    if choice is None:
        return
    if choice < 2:
        profile = model_profiles()["3b" if choice == 0 else "32b"]
        for key, value in profile.items():
            setattr(args, key, value)
    else:
        args.qwen_path = _input_value("Qwen 模型路径", args.qwen_path)


def interactive_menu(args: argparse.Namespace) -> bool:
    cursor = 0
    while True:
        options = [
            "运行 Demo",
            f"输入文件/目录             {_short(args.input)}",
            f"运行项目                  {args.projects}",
            f"Qwen 模型                 {_short(args.qwen_path)}",
            f"云端 checkpoint           {_short(args.cloud_checkpoint)}",
            f"端侧 checkpoint           {_short(args.edge_checkpoint)}",
            f"时序 checkpoint           {_short(args.temporal_pair_checkpoint)}",
            f"Prompt                    {_short(args.prompt)}",
            f"运行设置                  {args.max_frames} 帧 @ {args.sample_fps} FPS · {args.rounds} 轮/{args.round_step_seconds:g}s",
            "退出",
        ]
        choice = _select("SplitOculo 终端 Demo", options, cursor)
        if choice is None or choice == 9:
            return False
        cursor = choice
        if choice == 0:
            try:
                validate_args(args)
                save_config(args)
                return True
            except (ValueError, OSError) as exc:
                _clear_screen()
                input(f"配置有误：{exc}\n\n按 Enter 返回设置。")
        elif choice == 1:
            args.input = _input_value("输入文件或目录", args.input)
        elif choice == 2:
            args.projects = _multi_select_projects(args.projects)
        elif choice == 3:
            _model_menu(args)
        elif choice == 4:
            args.cloud_checkpoint = _input_value("云端 checkpoint", args.cloud_checkpoint)
        elif choice == 5:
            args.edge_checkpoint = _input_value("端侧 checkpoint", args.edge_checkpoint)
        elif choice == 6:
            args.temporal_pair_checkpoint = _input_value(
                "时序 checkpoint", args.temporal_pair_checkpoint
            )
        elif choice == 7:
            args.prompt = _input_value("Prompt", args.prompt)
        elif choice == 8:
            _settings_menu(args)
        try:
            save_config(args)
        except OSError as exc:
            _clear_screen()
            input(f"配置保存失败：{exc}\n\n按 Enter 返回设置。")


def parse_projects(value: str) -> list[str]:
    projects = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not projects:
        raise ValueError("--projects must contain at least one project")
    unknown = [item for item in projects if item not in PROJECT_IDS]
    if unknown:
        raise ValueError(f"unknown project(s): {', '.join(unknown)}")
    return projects


def validate_args(args: argparse.Namespace) -> list[str]:
    projects = parse_projects(args.projects)
    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        raise ValueError(f"input does not exist: {input_path}")
    cloud_checkpoint = Path(args.cloud_checkpoint).expanduser()
    if not cloud_checkpoint.is_file():
        raise ValueError(f"cloud checkpoint does not exist: {cloud_checkpoint}")
    if any(project != "baseline" for project in projects):
        if not args.edge_checkpoint:
            raise ValueError("SplitOculo projects require --edge-checkpoint")
        if not Path(args.edge_checkpoint).expanduser().is_file():
            raise ValueError(f"edge checkpoint does not exist: {args.edge_checkpoint}")
    if any(project in {"temporal", "codec"} for project in projects):
        if not args.temporal_pair_checkpoint:
            raise ValueError(
                "temporal and codec projects require --temporal-pair-checkpoint"
            )
        if not Path(args.temporal_pair_checkpoint).expanduser().is_file():
            raise ValueError(
                "temporal pair checkpoint does not exist: "
                f"{args.temporal_pair_checkpoint}"
            )
        if args.spatial_level.lower() != "49x64":
            raise ValueError("temporal and codec projects currently require --spatial-level 49x64")
    if not 1 <= args.baseline_jpeg_quality <= 100:
        raise ValueError("--baseline-jpeg-quality must be in [1, 100]")
    if args.max_frames <= 0 or args.sample_fps <= 0:
        raise ValueError("--max-frames and --sample-fps must be positive")
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")
    if args.round_step_seconds <= 0:
        raise ValueError("--round-step-seconds must be positive")
    if args.port <= 0 or args.port > 65535:
        raise ValueError("--port must be in [1, 65535]")
    return projects


def _auth(args: argparse.Namespace):
    return (args.username, args.password or "") if args.username else None


def fetch_health(server_url: str, timeout: float = 2.0, auth=None) -> dict[str, Any] | None:
    try:
        response = requests.get(f"{server_url}/health", timeout=timeout, auth=auth)
    except requests.ConnectionError:
        return None
    except requests.RequestException as exc:
        raise RuntimeError(f"health check failed: {exc}") from exc
    if not response.ok:
        raise RuntimeError(f"health check returned HTTP {response.status_code}")
    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError("health check returned non-JSON data") from exc
    if body.get("status") != "ok":
        raise RuntimeError(f"unhealthy cloud response: {body}")
    return body


def _normalized_path(value: str) -> str:
    return os.path.normcase(str(Path(value).expanduser().resolve()))


def validate_health(
    health: dict[str, Any], cloud_checkpoint: str, qwen_path: str
) -> None:
    missing = sorted(REQUIRED_HEALTH_FIELDS.difference(health))
    if missing:
        raise RuntimeError(
            "localhost is running an outdated cloud_server.py; missing /health "
            f"fields: {', '.join(missing)}. Update the server checkout first."
        )
    actual_checkpoint = health.get("checkpoint_path")
    if not actual_checkpoint or _normalized_path(actual_checkpoint) != _normalized_path(
        cloud_checkpoint
    ):
        raise RuntimeError(
            "localhost checkpoint mismatch: "
            f"running={actual_checkpoint!r}, requested={str(Path(cloud_checkpoint).resolve())!r}"
        )
    running_qwen = health.get("qwen_model_name") or health.get("qwen_path")
    if running_qwen != qwen_path:
        raise RuntimeError(
            "localhost Qwen mismatch: "
            f"running={running_qwen!r}, requested={qwen_path!r}"
        )


def build_cloud_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "cloud_server.py"),
        "--checkpoint",
        str(Path(args.cloud_checkpoint).expanduser().resolve()),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--device",
        args.device,
        "--qwen_path",
        args.qwen_path,
        "--preload_qwen",
    ]
    if args.offline:
        command.append("--offline")
    return command


def build_demo_command(
    args: argparse.Namespace, projects: list[str], start_time: float = 0.0
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "demo_client.py"),
        "--input",
        str(Path(args.input).expanduser().resolve()),
        "--server",
        f"http://127.0.0.1:{args.port}",
        "--prompt",
        args.prompt,
        "--device",
        args.device,
        "--timeout",
        str(args.timeout),
        "--projects",
        ",".join(projects),
        "--bandwidth_kb_s",
        str(args.bandwidth_kb_s),
        "--spatial_level",
        args.spatial_level,
        "--max_frames",
        str(args.max_frames),
        "--sample_fps",
        str(args.sample_fps),
        "--start_time",
        str(start_time),
        "--codec_flow_impl",
        args.codec_flow_impl,
        "--codec_selection_policy",
        args.codec_selection_policy,
        "--codec_reference_mode",
        args.codec_reference_mode,
        "--codec_mv_min_coverage",
        str(args.codec_mv_min_coverage),
        "--codec_max_p_chain",
        str(args.codec_max_p_chain),
        "--codec_gop_frames",
        str(args.codec_gop_frames),
        "--raw_width",
        str(args.raw_width),
        "--raw_height",
        str(args.raw_height),
        "--raw_fps",
        str(args.raw_fps),
        "--raw_format",
        args.raw_format,
        "--baseline_jpeg_quality",
        str(args.baseline_jpeg_quality),
        "--baseline_input_size",
        str(args.baseline_input_size),
    ]
    if args.edge_checkpoint:
        command.extend(
            ["--edge_checkpoint", str(Path(args.edge_checkpoint).expanduser().resolve())]
        )
    if args.temporal_pair_checkpoint:
        command.extend(
            [
                "--temporal_pair_checkpoint",
                str(Path(args.temporal_pair_checkpoint).expanduser().resolve()),
            ]
        )
    if args.username:
        command.extend(["--username", args.username])
    if args.password:
        command.extend(["--password", args.password])
    command.append("--stream")
    return command


def _tail(path: Path, lines: int = 30) -> str:
    try:
        values = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(values[-lines:])


_STATUS_VISIBLE = False
_STATUS_WIDTH = 0


def _status(message: str) -> None:
    global _STATUS_VISIBLE, _STATUS_WIDTH
    compact = " ".join(str(message).split())
    if len(compact) > 110:
        compact = "..." + compact[-107:]
    rendered = f"状态: {compact}"
    padding = " " * max(0, _STATUS_WIDTH - len(rendered))
    print(f"\r{rendered}{padding}", end="", flush=True)
    _STATUS_VISIBLE = True
    _STATUS_WIDTH = len(rendered)


def _clear_status() -> None:
    global _STATUS_VISIBLE, _STATUS_WIDTH
    if _STATUS_VISIBLE:
        print("\r" + (" " * _STATUS_WIDTH) + "\r", end="", flush=True)
        _STATUS_VISIBLE = False
        _STATUS_WIDTH = 0


def start_cloud(args: argparse.Namespace) -> tuple[subprocess.Popen, Path, Any]:
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix="splitoculo-cloud-", suffix=".log", delete=False
    )
    log_path = Path(handle.name)
    process = subprocess.Popen(
        build_cloud_command(args),
        cwd=ROOT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process, log_path, handle


def wait_for_cloud(
    process: subprocess.Popen,
    server_url: str,
    args: argparse.Namespace,
    log_path: Path,
) -> dict[str, Any]:
    deadline = time.monotonic() + args.startup_timeout
    next_message = 0.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"cloud server exited with code {process.returncode}\n{_tail(log_path)}"
            )
        health = fetch_health(server_url, timeout=2.0, auth=_auth(args))
        if health and health.get("model_loaded") and health.get("qwen_loaded"):
            validate_health(health, args.cloud_checkpoint, args.qwen_path)
            return health
        if time.monotonic() >= next_message:
            _status("正在加载 cloud checkpoint 与 Qwen 模型...")
            next_message = time.monotonic() + 10.0
        time.sleep(1.0)
    raise RuntimeError(
        f"cloud startup timed out after {args.startup_timeout}s\n{_tail(log_path)}"
    )


def warm_existing_cloud(server_url: str, args: argparse.Namespace) -> dict[str, Any]:
    response = requests.post(
        f"{server_url}/warmup",
        timeout=args.startup_timeout,
        auth=_auth(args),
    )
    if not response.ok:
        raise RuntimeError(f"cloud warmup returned HTTP {response.status_code}: {response.text[:500]}")
    health = fetch_health(server_url, timeout=5.0, auth=_auth(args))
    if not health or not health.get("qwen_loaded"):
        raise RuntimeError("cloud warmup completed but qwen_loaded is false")
    return health


def _milliseconds(value: Any) -> str:
    return "-" if value is None else f"{float(value):.1f} ms"


def _kilobytes(value: Any) -> str:
    return "-" if value is None else f"{float(value) / 1024.0:.2f} KB"


def _display_width(value: str) -> int:
    return sum(
        2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
        for character in value
    )


def _fit_cell(value: Any, width: int, align: str = "left") -> str:
    text = str(value)
    result = ""
    used = 0
    for character in text:
        size = 2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
        if used + size > width:
            break
        result += character
        used += size
    padding = max(0, width - used)
    if align == "center":
        left = padding // 2
        return (" " * left) + result + (" " * (padding - left))
    return result + (" " * padding)


def _wrap_cells(value: Any, width: int) -> list[str]:
    lines: list[str] = []
    for source_line in str(value).splitlines() or [""]:
        current = ""
        used = 0
        for character in source_line:
            size = 2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
            if current and used + size > width:
                lines.append(_fit_cell(current, width))
                current = ""
                used = 0
            current += character
            used += size
        lines.append(_fit_cell(current, width))
    return lines or [" " * width]


AVERAGED_RESULT_FIELDS = (
    "edge_encode_ms",
    "upload_delay_ms",
    "ttft_without_network_ms",
    "relative_speed",
)


def update_aggregate_result(
    aggregates: dict[str, dict[str, Any]],
    project: str,
    row: dict[str, Any],
    round_index: int,
    total_rounds: int,
    start_time: float,
) -> dict[str, Any]:
    aggregate = aggregates.setdefault(
        project,
        {
            "label": row.get("label") or project,
            "completed_rounds": 0,
            "rounds": total_rounds,
            "frames": 0,
            "request_bytes": 0,
            "round_outputs": [],
            "_metric_totals": {},
            "_metric_counts": {},
        },
    )
    aggregate["label"] = row.get("label") or aggregate["label"]
    aggregate["completed_rounds"] += 1
    aggregate["rounds"] = total_rounds
    aggregate["frames"] += int(row.get("frames") or 0)
    aggregate["request_bytes"] += int(
        row.get("request_bytes", row.get("payload_bytes")) or 0
    )
    for field in AVERAGED_RESULT_FIELDS:
        value = row.get(field)
        if value is None:
            continue
        totals = aggregate["_metric_totals"]
        counts = aggregate["_metric_counts"]
        totals[field] = float(totals.get(field, 0.0)) + float(value)
        counts[field] = int(counts.get(field, 0)) + 1
        aggregate[field] = totals[field] / counts[field]
    aggregate["round_outputs"].append(
        {
            "round": round_index,
            "window_start_seconds": start_time,
            "response": row.get("response"),
            "error": row.get("error"),
            "interrupted": bool(row.get("interrupted")),
        }
    )
    return aggregate


def _card_body(row: dict[str, Any], inner_width: int) -> tuple[list[str], list[str]]:
    aggregate_mode = "round_outputs" in row
    label = str(row.get("round_label") or row.get("label") or "Result")
    edge = _milliseconds(row.get("edge_encode_ms"))
    simulated = _milliseconds(row.get("upload_delay_ms"))
    ttft = _milliseconds(row.get("ttft_without_network_ms"))
    frames = row.get("frames")
    frame_text = "-" if frames is None else str(frames)
    payload = _kilobytes(row.get("request_bytes", row.get("payload_bytes")))
    relative_speed = row.get("relative_speed")
    speed_text = "-" if relative_speed is None else f"{float(relative_speed):.2f}×"
    header = [
        _fit_cell(label, inner_width, "center"),
    ]
    if aggregate_mode:
        header.append(
            _fit_cell(
                f"已完成轮次: {row.get('completed_rounds', 0)}/{row.get('rounds', 0)}",
                inner_width,
                "center",
            )
        )
    header.extend([
        _fit_cell(f"总输入帧数: {frame_text}", inner_width, "center"),
        _fit_cell(f"总负载大小: {payload}", inner_width, "center"),
        _fit_cell(f"{'平均' if aggregate_mode else ''}相对速度: {speed_text}", inner_width, "center"),
        _fit_cell(f"{'平均' if aggregate_mode else ''}端侧编码: {edge}", inner_width, "center"),
        _fit_cell(f"{'平均' if aggregate_mode else ''}模拟时延: {simulated}", inner_width, "center"),
        _fit_cell(f"平均 TTFT: {ttft}" if aggregate_mode else f"TTFT: {ttft}", inner_width, "center"),
    ])
    if aggregate_mode:
        response = []
        for output in row.get("round_outputs") or []:
            prefix = (
                f"第 {output['round']} 轮"
                f"（{float(output['window_start_seconds']):g}s）: "
            )
            if output.get("interrupted"):
                value = prefix + "已到点中断"
            elif output.get("error"):
                value = prefix + f"失败：{output['error']}"
            else:
                value = prefix + str(output.get("response") or "（无响应）")
            response.extend(_wrap_cells(value, inner_width))
        completed = int(row.get("completed_rounds") or 0)
        total = int(row.get("rounds") or 0)
        footer = "已完成" if completed >= total else "进行中"
    elif row.get("error"):
        response = _wrap_cells(f"失败：{row['error']}", inner_width)
        footer = "失败"
    else:
        response = _wrap_cells(row.get("response") or "（无响应）", inner_width)
        footer = "已完成"
    return header, [*response, _fit_cell(footer, inner_width, "center")]


def render_comparison(
    rows: list[dict[str, Any]], terminal_width: int | None = None
) -> str:
    if not rows:
        return "（没有结果）"
    available = terminal_width or shutil.get_terminal_size((160, 40)).columns
    gap = 2
    columns = min(len(rows), 4, max(1, (available + gap) // (44 + gap)))
    output: list[str] = []
    for offset in range(0, len(rows), columns):
        group = rows[offset : offset + columns]
        card_width = max(34, (available - gap * (len(group) - 1)) // len(group))
        inner = card_width - 2
        cards = [_card_body(row, inner) for row in group]
        max_header = max(len(header) for header, _ in cards)
        max_response = max(len(response) - 1 for _, response in cards)
        rendered_cards: list[list[str]] = []
        for header, response_and_footer in cards:
            response = response_and_footer[:-1]
            footer = response_and_footer[-1]
            lines = ["┌" + ("─" * inner) + "┐"]
            lines.extend("│" + line + "│" for line in header)
            lines.extend("│" + (" " * inner) + "│" for _ in range(max_header - len(header)))
            lines.append("├" + ("─" * inner) + "┤")
            lines.extend("│" + line + "│" for line in response)
            lines.extend("│" + (" " * inner) + "│" for _ in range(max_response - len(response)))
            lines.append("├" + ("─" * inner) + "┤")
            lines.append("│" + footer + "│")
            lines.append("└" + ("─" * inner) + "┘")
            rendered_cards.append(lines)
        if output:
            output.append("")
        for line_index in range(len(rendered_cards[0])):
            output.append((" " * gap).join(card[line_index] for card in rendered_cards))
    return "\n".join(output)


def render_result(row: dict[str, Any]) -> str:
    """Backward-compatible single-card renderer used by tests and callers."""
    return render_comparison([row], terminal_width=88)


def run_demo(
    args: argparse.Namespace,
    projects: list[str],
    round_index: int = 1,
    total_rounds: int = 1,
    start_time: float = 0.0,
    deadline: float | None = None,
    aggregate_results: dict[str, dict[str, Any]] | None = None,
) -> int:
    if len(projects) != 1:
        raise ValueError("run_demo expects exactly one project in multi-round mode")
    project = projects[0]
    aggregates = aggregate_results if aggregate_results is not None else {}
    process = subprocess.Popen(
        build_demo_command(args, projects, start_time=start_time),
        cwd=ROOT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    final_result = None
    received_result = False
    live_terminal = sys.stdout.isatty()
    stream_active = False
    round_name = f"第 {round_index}/{total_rounds} 轮"
    assert process.stdout is not None
    output_queue: queue.Queue[str | None] = queue.Queue()

    def _read_output() -> None:
        try:
            for raw_line in process.stdout:
                output_queue.put(raw_line)
        finally:
            output_queue.put(None)

    reader = threading.Thread(target=_read_output, daemon=True)
    reader.start()
    while True:
        if (
            deadline is not None
            and time.monotonic() >= deadline
            and process.poll() is None
        ):
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            _clear_status()
            update_aggregate_result(
                aggregates,
                project,
                {"label": aggregates.get(project, {}).get("label", project), "interrupted": True},
                round_index,
                total_rounds,
                start_time,
            )
            if live_terminal:
                _clear_screen()
            print(render_comparison(list(aggregates.values())), flush=True)
            print(f"{round_name}到达下一轮采样时刻，已中断。", file=sys.stderr, flush=True)
            return 124
        wait_seconds = 0.2
        if deadline is not None:
            wait_seconds = max(0.01, min(wait_seconds, deadline - time.monotonic()))
        try:
            raw_line = output_queue.get(timeout=wait_seconds)
        except queue.Empty:
            continue
        if raw_line is None:
            break
        line = raw_line.rstrip("\r\n")
        if line.startswith(STREAM_START_PREFIX):
            try:
                event = json.loads(line[len(STREAM_START_PREFIX) :])
                label = str(event.get("label") or "项目")
            except json.JSONDecodeError:
                label = "项目"
            _clear_status()
            stream_active = True
            if live_terminal:
                _clear_screen()
                if aggregates:
                    print(render_comparison(list(aggregates.values())))
                    print()
                print(f"正在生成：{round_name} · {label}\n")
                print(f"{round_name}回答：", end="", flush=True)
            else:
                _status(f"{round_name}正在生成 {label}...")
        elif line.startswith(STREAM_DELTA_PREFIX):
            try:
                event = json.loads(line[len(STREAM_DELTA_PREFIX) :])
                text = str(event.get("text") or "")
            except json.JSONDecodeError:
                text = ""
            if text and live_terminal:
                print(text, end="", flush=True)
        elif line.startswith(RESULT_ITEM_PREFIX):
            try:
                row = json.loads(line[len(RESULT_ITEM_PREFIX) :])
                full_response_ms = row.get("full_response_ms")
                if full_response_ms is not None and float(full_response_ms) > 0:
                    row["relative_speed"] = (
                        args.round_step_seconds * 1000.0 / float(full_response_ms)
                    )
                update_aggregate_result(
                    aggregates,
                    project,
                    row,
                    round_index,
                    total_rounds,
                    start_time,
                )
                received_result = True
                if stream_active and live_terminal:
                    print()
                    _clear_screen()
                    print(render_comparison(list(aggregates.values())), flush=True)
                else:
                    _status(f"{round_name} · {row.get('label') or '项目'} 已完成")
                stream_active = False
            except json.JSONDecodeError:
                _status("客户端返回了无效结果事件")
        elif line.startswith(RESULT_JSON_PREFIX):
            try:
                final_result = json.loads(line[len(RESULT_JSON_PREFIX) :])
            except json.JSONDecodeError:
                _status("客户端返回了无效汇总结果")
        elif line:
            _status(line)
    returncode = process.wait()
    _clear_status()
    if not received_result and final_result is not None:
        fallback_rows = list(final_result.get("results") or [])
        if fallback_rows:
            row = fallback_rows[0]
            update_aggregate_result(
                aggregates, project, row, round_index, total_rounds, start_time
            )
            received_result = True
    if received_result and not live_terminal:
        print(render_comparison(list(aggregates.values())), flush=True)
    if returncode != 0:
        print(f"Demo client exited with code {returncode}", file=sys.stderr)
    return returncode


def stop_cloud(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def new_cloud_session() -> dict[str, Any]:
    return {"process": None, "log_path": None, "log_handle": None}


def close_cloud_session(session: dict[str, Any], show_status: bool = False) -> None:
    process = session.get("process")
    if show_status and process is not None and process.poll() is None:
        _status("正在关闭本次会话启动的 cloud 服务...")
    stop_cloud(process)
    handle = session.get("log_handle")
    if handle is not None:
        handle.close()
    path = session.get("log_path")
    if path is not None:
        try:
            Path(path).unlink()
        except OSError:
            pass
    session.update(process=None, log_path=None, log_handle=None)
    if show_status:
        _clear_status()


def _start_session_cloud(
    args: argparse.Namespace,
    server_url: str,
    session: dict[str, Any],
) -> dict[str, Any]:
    process, log_path, log_handle = start_cloud(args)
    session.update(process=process, log_path=log_path, log_handle=log_handle)
    return wait_for_cloud(process, server_url, args, log_path)


def execute(
    args: argparse.Namespace,
    projects: list[str],
    cloud_session: dict[str, Any] | None = None,
) -> int:
    server_url = f"http://127.0.0.1:{args.port}"
    owns_session = cloud_session is None
    session = cloud_session if cloud_session is not None else new_cloud_session()
    try:
        _status(f"检查 localhost 服务 · {args.qwen_path} · {','.join(projects)}")
        health = fetch_health(server_url, auth=_auth(args))
        if health is None:
            if session.get("process") is not None:
                close_cloud_session(session)
            _status("未发现本地服务，正在启动 cloud_server.py...")
            health = _start_session_cloud(args, server_url, session)
        else:
            _status("复用现有 localhost 服务，正在核对模型与 checkpoint...")
            try:
                validate_health(health, args.cloud_checkpoint, args.qwen_path)
            except RuntimeError:
                process = session.get("process")
                if process is None or process.poll() is not None:
                    raise
                _status("模型或 checkpoint 已更改，正在重启本次会话服务...")
                close_cloud_session(session)
                health = _start_session_cloud(args, server_url, session)
            if not health.get("qwen_loaded"):
                _status("正在预热现有 Qwen 服务...")
                health = warm_existing_cloud(server_url, args)
                validate_health(health, args.cloud_checkpoint, args.qwen_path)
        _status(
            "云端就绪 · "
            f"checkpoint={health.get('checkpoint_transmission_tokens')}x"
            f"{health.get('checkpoint_bottleneck_dim')}, "
            f"target={health.get('checkpoint_target_tokens')}x"
            f"{health.get('checkpoint_hidden_size')}"
        )
        aggregate_results: dict[str, dict[str, Any]] = {}
        last_returncode = 0
        for project in projects:
            schedule_started = time.monotonic()
            for round_index in range(1, args.rounds + 1):
                scheduled_at = schedule_started + (round_index - 1) * args.round_step_seconds
                remaining = scheduled_at - time.monotonic()
                if remaining > 0:
                    _status(
                        f"{project} · 等待第 {round_index}/{args.rounds} 轮采样时刻..."
                    )
                    time.sleep(remaining)
                else:
                    lag = -remaining
                    if round_index > 1 and lag >= 0.01:
                        _status(
                            f"{project} · 第 {round_index}/{args.rounds} 轮"
                            f"已落后采样 {lag:.2f} 秒"
                        )
                window_start = (round_index - 1) * args.round_step_seconds
                deadline = None
                if args.interrupt_on_next_round and round_index < args.rounds:
                    deadline = scheduled_at + args.round_step_seconds
                last_returncode = run_demo(
                    args,
                    [project],
                    round_index=round_index,
                    total_rounds=args.rounds,
                    start_time=window_start,
                    deadline=deadline,
                    aggregate_results=aggregate_results,
                )
                if last_returncode not in {0, 124}:
                    return last_returncode
        return 0 if last_returncode == 124 else last_returncode
    except KeyboardInterrupt:
        _clear_status()
        print("已中断。", file=sys.stderr)
        return 130
    except Exception as exc:
        _clear_status()
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if owns_session:
            close_cloud_session(session, show_status=True)
        _clear_status()


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        saved = load_config()
    except RuntimeError as exc:
        print(f"WARNING: {exc}", file=sys.stderr)
        saved = default_config()
    parser = build_parser(saved)
    args = parser.parse_args(raw_argv)
    use_menu = args.interactive or not raw_argv

    if not use_menu:
        try:
            projects = validate_args(args)
        except ValueError as exc:
            parser.error(str(exc))
        return execute(args, projects)

    cloud_session = new_cloud_session()
    returncode = 0
    try:
        while interactive_menu(args):
            projects = validate_args(args)
            _clear_screen()
            returncode = execute(args, projects, cloud_session=cloud_session)
            print("\n运行完成。Enter 返回菜单，输入 q 后回车退出。")
            if input("> ").strip().lower() == "q":
                return returncode
        _clear_screen()
        return returncode
    finally:
        close_cloud_session(cloud_session, show_status=True)


if __name__ == "__main__":
    raise SystemExit(main())

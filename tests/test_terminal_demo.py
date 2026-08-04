from argparse import Namespace
from pathlib import Path
import subprocess
from unittest import mock

from scripts import terminal_demo


def _args(tmp_path: Path, **overrides):
    input_path = tmp_path / "frames"
    input_path.mkdir()
    cloud = tmp_path / "cloud.pth"
    edge = tmp_path / "edge.pth"
    temporal = tmp_path / "temporal.pth"
    for path in (cloud, edge, temporal):
        path.touch()
    values = {
        "input": str(input_path),
        "cloud_checkpoint": str(cloud),
        "edge_checkpoint": str(edge),
        "temporal_pair_checkpoint": str(temporal),
        "qwen_path": "Qwen/Qwen2.5-VL-3B-Instruct",
        "projects": "baseline,so,temporal,codec",
        "prompt": "describe",
        "device": "cpu",
        "port": 8080,
        "timeout": 30,
        "startup_timeout": 60,
        "offline": True,
        "bandwidth_kb_s": 62.5,
        "spatial_level": "49x64",
        "max_frames": 4,
        "sample_fps": 2.0,
        "rounds": 1,
        "round_step_seconds": 2.0,
        "interrupt_on_next_round": False,
        "codec_flow_impl": "feature_grid",
        "codec_selection_policy": "best_effort_ip",
        "codec_reference_mode": "recursive",
        "codec_mv_min_coverage": 0.0,
        "codec_max_p_chain": 4,
        "codec_gop_frames": 4,
        "raw_width": 224,
        "raw_height": 224,
        "raw_fps": 10.0,
        "raw_format": "rgb24",
        "baseline_jpeg_quality": 90,
        "baseline_input_size": 224,
        "username": None,
        "password": None,
    }
    values.update(overrides)
    return Namespace(**values)


def _health(args):
    return {
        "status": "ok",
        "model_loaded": True,
        "qwen_loaded": True,
        "qwen_model_name": args.qwen_path,
        "checkpoint_path": str(Path(args.cloud_checkpoint).resolve()),
        "checkpoint_hidden_size": 1280,
        "checkpoint_bottleneck_dim": 64,
        "checkpoint_transmission_tokens": 49,
        "checkpoint_target_tokens": 256,
        "compute_warmup_supported": True,
        "feature_metrics_supported": True,
    }


def _assert_raises(error_type, pattern, callback):
    try:
        callback()
    except error_type as exc:
        assert pattern in str(exc)
        return
    raise AssertionError(f"expected {error_type.__name__}")


def test_parse_projects_preserves_order_and_duplicates():
    assert terminal_demo.parse_projects("codec,baseline,codec") == [
        "codec",
        "baseline",
        "codec",
    ]


def test_validate_args_requires_temporal_checkpoint(tmp_path):
    args = _args(tmp_path, projects="temporal", temporal_pair_checkpoint=None)
    _assert_raises(
        ValueError,
        "temporal-pair-checkpoint",
        lambda: terminal_demo.validate_args(args),
    )


def test_commands_use_localhost_and_current_python(tmp_path):
    args = _args(tmp_path)
    projects = terminal_demo.validate_args(args)
    cloud = terminal_demo.build_cloud_command(args)
    demo = terminal_demo.build_demo_command(args, projects)
    assert cloud[0] == terminal_demo.sys.executable
    assert "127.0.0.1" in cloud
    assert "--offline" in cloud
    assert "http://127.0.0.1:8080" in demo
    assert demo[demo.index("--projects") + 1] == "baseline,so,temporal,codec"
    assert demo[demo.index("--start_time") + 1] == "0.0"


def test_demo_command_sets_sliding_window_start(tmp_path):
    args = _args(tmp_path)
    demo = terminal_demo.build_demo_command(
        args, ["baseline"], start_time=4.5, client_rounds=3
    )
    assert demo[demo.index("--start_time") + 1] == "4.5"
    assert demo[demo.index("--rounds") + 1] == "3"
    assert demo[demo.index("--round_step_seconds") + 1] == "2.0"


def test_demo_command_passes_persistent_interrupt_deadline(tmp_path):
    args = _args(tmp_path, interrupt_on_next_round=True, round_step_seconds=5.0)
    demo = terminal_demo.build_demo_command(args, ["baseline"], client_rounds=3)
    assert demo[demo.index("--interrupt_after_seconds") + 1] == "5.0"


def test_saved_config_round_trip_excludes_password(tmp_path):
    args = _args(tmp_path)
    args.password = "do-not-save"
    target = tmp_path / "terminal_demo.json"
    terminal_demo.save_config(args, target)
    saved_text = target.read_text(encoding="utf-8")
    loaded = terminal_demo.load_config(target)
    assert loaded["projects"] == "baseline,so,temporal,codec"
    assert loaded["input"] == args.input
    assert "do-not-save" not in saved_text


def test_parser_uses_persisted_defaults():
    parser = terminal_demo.build_parser(
        {"input": "/data/demo.mp4", "max_frames": 12, "offline": False}
    )
    args = parser.parse_args([])
    assert args.input == "/data/demo.mp4"
    assert args.max_frames == 12
    assert args.offline is False
    assert args.rounds == 1
    assert args.round_step_seconds == 2.0


def test_arrow_menu_moves_selection():
    fake_stdin = mock.Mock()
    fake_stdin.isatty.return_value = True
    with (
        mock.patch.object(terminal_demo.sys, "stdin", fake_stdin),
        mock.patch.object(terminal_demo, "_clear_screen"),
        mock.patch.object(terminal_demo, "_read_key", side_effect=["down", "enter"]),
    ):
        assert terminal_demo._select("menu", ["first", "second"]) == 1


def test_network_delay_preset_selects_ble():
    args = mock.Mock(bandwidth_kb_s=0.0)
    with mock.patch.object(terminal_demo, "_select", return_value=1):
        terminal_demo._network_menu(args)
    assert args.bandwidth_kb_s == 62.5


def test_health_rejects_old_server(tmp_path):
    args = _args(tmp_path)
    _assert_raises(
        RuntimeError,
        "outdated cloud_server",
        lambda: terminal_demo.validate_health(
            {"status": "ok", "model_loaded": True},
            args.cloud_checkpoint,
            args.qwen_path,
        ),
    )


def test_health_accepts_matching_service(tmp_path):
    args = _args(tmp_path)
    terminal_demo.validate_health(_health(args), args.cloud_checkpoint, args.qwen_path)


def test_health_accepts_requested_qwen_before_warmup(tmp_path):
    args = _args(tmp_path)
    health = _health(args)
    health["qwen_model_name"] = None
    health["qwen_path"] = args.qwen_path
    terminal_demo.validate_health(health, args.cloud_checkpoint, args.qwen_path)


def test_terminal_requests_compute_warmup_for_selected_projects(tmp_path):
    args = _args(tmp_path, max_frames=8, baseline_input_size=224, sample_fps=2.0)
    health = _health(args)
    response = mock.Mock(ok=True)
    response.json.return_value = {
        "status": "ok",
        "compute_warmed": True,
        "warmup_ms": 1234,
        "paths": {"native_qwen": {"cached": False}},
    }
    with (
        mock.patch.object(terminal_demo.requests, "post", return_value=response) as post,
        mock.patch.object(terminal_demo, "fetch_health", return_value=health),
    ):
        result = terminal_demo.warm_existing_cloud(
            "http://127.0.0.1:8080", args, ["baseline"]
        )

    assert result["warmup_result"]["compute_warmed"] is True
    assert post.call_args.kwargs["json"] == {
        "projects": ["baseline"],
        "max_frames": 8,
        "video_pixel_budget": 224 * 224,
        "video_fps": 2.0,
    }


def test_render_result_contains_core_and_codec_metrics():
    rendered = terminal_demo.render_result(
        {
            "label": "Codec",
            "response": "A person is walking.",
            "edge_encode_ms": 12.5,
            "cloud_process_ms": 25,
            "network_overhead_ms": 1,
            "upload_delay_ms": 3,
            "cloud_ttft_ms": 35,
            "first_response_ms": 38,
            "end_to_end_ttft_ms": 42,
            "ttft_without_network_ms": 38,
            "full_response_ms": 55,
            "payload_bytes": 2048,
            "request_bytes": 3072,
            "frames": 4,
            "sample_fps": 2,
            "round_label": "第 1/3 轮 · Codec",
            "relative_speed": 1.25,
            "reader": "pyav",
            "feature_shape": [2, 49, 64],
            "temporal_redundancy_acceleration": True,
            "codec_cnn_frames": 1,
            "codec_warp_frames": 3,
            "codec_selected_frame_types": ["I", "P", "P", "P"],
        }
    )
    assert "A person is walking." in rendered
    assert "端侧编码: 12.5 ms" in rendered
    assert "模拟时延: 3.0 ms" in rendered
    assert "纯计算 TTFT: 35.0 ms" in rendered
    assert "首响应时间: 38.0 ms" in rendered
    assert "Cloud process" not in rendered
    assert "第 1/3 轮" in rendered
    assert "总输入帧数: 4" in rendered
    assert "总负载大小: 3.00 KB" in rendered
    assert "相对速度: 1.25×" in rendered


def test_aggregate_result_sums_counts_and_averages_metrics():
    aggregates = {}
    terminal_demo.update_aggregate_result(
        aggregates,
        "baseline",
        {
            "label": "Baseline",
            "response": "first",
            "frames": 2,
            "request_bytes": 1024,
            "edge_encode_ms": 10,
            "cloud_ttft_ms": 1000,
            "first_response_ms": 1000,
            "relative_budget_ms": 4000,
            "relative_speed": 4,
        },
        round_index=1,
        total_rounds=2,
        start_time=0.0,
    )
    aggregate = terminal_demo.update_aggregate_result(
        aggregates,
        "baseline",
        {
            "label": "Baseline",
            "response": "second",
            "frames": 3,
            "request_bytes": 2048,
            "edge_encode_ms": 20,
            "cloud_ttft_ms": 1000,
            "first_response_ms": 1000,
            "relative_budget_ms": 2000,
            "relative_speed": 2,
        },
        round_index=2,
        total_rounds=2,
        start_time=2.0,
    )

    assert aggregate["completed_rounds"] == 2
    assert aggregate["frames"] == 5
    assert aggregate["request_bytes"] == 3072
    assert aggregate["edge_encode_ms"] == 15
    assert aggregate["relative_speed"] == 3
    rendered = terminal_demo.render_result(aggregate)
    assert "已完成轮次: 2/2" in rendered
    assert "第 1 轮（0s）: first" in rendered
    assert "第 2 轮（2s）: second" in rendered


def test_feature_reconstruction_metrics_render_in_header_and_round():
    aggregates = {}
    aggregate = terminal_demo.update_aggregate_result(
        aggregates,
        "so",
        {
            "label": "SO",
            "response": "done",
            "frames": 2,
            "feature_cosine_similarity": 0.9876543,
            "feature_mse": 0.0123456,
        },
        round_index=1,
        total_rounds=1,
        start_time=0.0,
    )
    rendered = terminal_demo.render_result(aggregate)
    assert "特征 Cossim: 0.987654" in rendered
    assert "特征 MSE: 0.012346" in rendered
    assert "第 1 轮（0s）: done" in rendered


def test_interrupted_round_counts_input_without_averaging_incomplete_speed():
    aggregates = {}
    aggregate = terminal_demo.update_aggregate_result(
        aggregates,
        "baseline",
        {
            "label": "Baseline",
            "response": "partial",
            "interrupted": True,
            "frames": 2,
            "request_bytes": 1024,
        },
        round_index=1,
        total_rounds=2,
        start_time=0.0,
    )

    assert aggregate["frames"] == 2
    assert aggregate["request_bytes"] == 1024
    assert aggregate.get("relative_speed") is None
    rendered = terminal_demo.render_result(aggregate)
    assert "已到点中断；部分回答：partial" in rendered


def test_comparison_renders_results_side_by_side():
    rendered = terminal_demo.render_comparison(
        [
            {"label": "Baseline", "response": "first", "ttft_without_network_ms": 10},
            {"label": "Ours", "response": "second", "ttft_without_network_ms": 5},
        ],
        terminal_width=100,
    )
    lines = rendered.splitlines()
    assert lines[0].count("┌") == 2
    assert any("Baseline" in line and "Ours" in line for line in lines)


def test_stop_cloud_only_acts_on_live_process():
    class FakeProcess:
        def __init__(self):
            self.terminated = False
            self.killed = False
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = 0

        def wait(self, timeout=None):
            if self.returncode is None:
                raise subprocess.TimeoutExpired("fake", timeout)
            return self.returncode

        def kill(self):
            self.killed = True
            self.returncode = -9

    process = FakeProcess()
    terminal_demo.stop_cloud(process)
    assert process.terminated
    assert not process.killed


def test_interactive_session_keeps_owned_cloud_running(tmp_path):
    args = _args(tmp_path, projects="baseline")

    class LiveProcess:
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

    process = LiveProcess()
    session = {"process": process, "log_path": None, "log_handle": None}
    health = _health(args)
    with (
        mock.patch.object(terminal_demo, "fetch_health", return_value=health),
        mock.patch.object(terminal_demo, "warm_existing_cloud", return_value=health),
        mock.patch.object(terminal_demo, "run_demo", return_value=0),
        mock.patch.object(terminal_demo, "start_cloud") as start,
    ):
        assert terminal_demo.execute(args, ["baseline"], cloud_session=session) == 0
    start.assert_not_called()
    assert process.terminated is False


def test_execute_uses_one_persistent_client_for_all_projects(tmp_path):
    args = _args(tmp_path, projects="baseline,so", rounds=3)
    health = _health(args)
    session = {"process": None, "log_path": None, "log_handle": None}
    with (
        mock.patch.object(terminal_demo, "fetch_health", return_value=health),
        mock.patch.object(terminal_demo, "warm_existing_cloud", return_value=health),
        mock.patch.object(terminal_demo, "run_demo", return_value=0) as run,
    ):
        assert terminal_demo.execute(
            args, ["baseline", "so"], cloud_session=session
        ) == 0

    run.assert_called_once()
    assert run.call_args.args[1] == ["baseline", "so"]
    assert run.call_args.kwargs["client_rounds"] == 3


def test_interrupt_mode_still_uses_one_persistent_client(tmp_path):
    args = _args(
        tmp_path,
        projects="baseline,so",
        rounds=3,
        interrupt_on_next_round=True,
    )
    health = _health(args)
    session = {"process": None, "log_path": None, "log_handle": None}
    with (
        mock.patch.object(terminal_demo, "fetch_health", return_value=health),
        mock.patch.object(terminal_demo, "warm_existing_cloud", return_value=health),
        mock.patch.object(terminal_demo, "run_demo", return_value=0) as run,
    ):
        assert terminal_demo.execute(
            args, ["baseline", "so"], cloud_session=session
        ) == 0

    run.assert_called_once()
    assert run.call_args.args[1] == ["baseline", "so"]
    assert run.call_args.kwargs["client_rounds"] == 3

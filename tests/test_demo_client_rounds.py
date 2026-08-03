import io
import json
from unittest import mock

from scripts import demo_client


def test_demo_client_reuses_edge_encoder_across_rounds(tmp_path):
    input_path = tmp_path / "input.jpg"
    input_path.touch()
    output = io.StringIO()
    argv = [
        "demo_client.py",
        "--input",
        str(input_path),
        "--projects",
        "so",
        "--edge_checkpoint",
        "edge.pth",
        "--rounds",
        "3",
        "--round_step_seconds",
        "0.001",
    ]
    rows = []

    def fake_run(*_args, **_kwargs):
        return {
            "label": "SO",
            "response": "ok",
            "frames": 1,
            "request_bytes": 10,
            "full_response_ms": 1,
        }

    with (
        mock.patch.object(demo_client.sys, "argv", argv),
        mock.patch.object(demo_client.sys, "stdout", output),
        mock.patch.object(
            demo_client,
            "_variant_specs",
            return_value=[("SO", True, False, False, False)],
        ),
        mock.patch.object(demo_client, "EdgeEncoder", return_value=object()) as encoder,
        mock.patch.object(demo_client, "_run_variant", side_effect=fake_run) as run,
    ):
        assert demo_client.main() == 0

    for line in output.getvalue().splitlines():
        if line.startswith("DEMO_RESULT_ITEM="):
            rows.append(json.loads(line.split("=", 1)[1]))
    assert encoder.call_count == 1
    assert run.call_count == 3
    assert [row["round"] for row in rows] == [1, 2, 3]
    assert [row["window_start_seconds"] for row in rows] == [0.0, 0.001, 0.002]


def test_demo_client_reuses_models_across_projects_and_rounds(tmp_path):
    input_path = tmp_path / "input.jpg"
    input_path.touch()
    output = io.StringIO()
    argv = [
        "demo_client.py",
        "--input",
        str(input_path),
        "--projects",
        "so,temporal",
        "--edge_checkpoint",
        "edge.pth",
        "--temporal_pair_checkpoint",
        "temporal.pth",
        "--rounds",
        "2",
        "--round_step_seconds",
        "0.001",
    ]
    fusion = mock.Mock()

    def fake_run(*args, **_kwargs):
        return {
            "label": args[4],
            "response": "ok",
            "frames": 1,
            "request_bytes": 10,
            "full_response_ms": 1,
        }

    with (
        mock.patch.object(demo_client.sys, "argv", argv),
        mock.patch.object(demo_client.sys, "stdout", output),
        mock.patch.object(
            demo_client,
            "_variant_specs",
            return_value=[
                ("SO", True, False, False, False),
                ("Temporal", True, False, True, False),
            ],
        ),
        mock.patch.object(demo_client, "EdgeEncoder", return_value=object()) as encoder,
        mock.patch.object(
            demo_client,
            "load_temporal_pair_fusion",
            return_value=(fusion, {"temporal_patch_size": 2}),
        ) as load_temporal,
        mock.patch.object(demo_client, "_run_variant", side_effect=fake_run) as run,
    ):
        assert demo_client.main() == 0

    rows = [
        json.loads(line.split("=", 1)[1])
        for line in output.getvalue().splitlines()
        if line.startswith("DEMO_RESULT_ITEM=")
    ]
    assert encoder.call_count == 1
    assert load_temporal.call_count == 1
    assert run.call_count == 4
    assert [(row["project"], row["round"]) for row in rows] == [
        ("so", 1),
        ("so", 2),
        ("temporal", 1),
        ("temporal", 2),
    ]

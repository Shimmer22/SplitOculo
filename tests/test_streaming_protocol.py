import io
import json
import threading
import time
from types import SimpleNamespace
from unittest import mock

from scripts import cloud_server, demo_client, terminal_demo


def test_cloud_ndjson_splits_text_into_characters():
    def task(emit, cancel_event):
        assert not cancel_event.is_set()
        emit("你好")
        return {"response": "你好", "cloud_ttft_ms": 12.3}

    with cloud_server.app.app_context():
        response = cloud_server._ndjson_response(task)
        events = [json.loads(line) for line in "".join(response.response).splitlines()]

    assert [event.get("text") for event in events if event["type"] == "delta"] == [
        "你",
        "好",
    ]
    assert events[-1]["type"] == "result"
    assert events[-1]["result"]["response"] == "你好"


def test_stream_close_signals_generation_cancellation():
    worker_finished = threading.Event()

    def task(emit, cancel_event):
        emit("x")
        cancel_event.wait(timeout=1.0)
        worker_finished.set()
        return {"response": "cancelled"}

    with cloud_server.app.app_context():
        response = cloud_server._ndjson_response(task)
        iterator = iter(response.response)
        assert '"text": "x"' in next(iterator)
        response.close()

    assert worker_finished.wait(timeout=1.0)


def test_cancel_stopping_criteria_tracks_event():
    event = threading.Event()
    criteria = cloud_server.CancelEventStoppingCriteria(event)
    assert criteria(None, None) is False
    event.set()
    assert criteria(None, None) is True


def test_feature_reconstruction_metrics_and_session_release():
    candidate = cloud_server.torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    reference = candidate.clone()
    metrics = cloud_server.CloudInferenceEngine.feature_reconstruction_metrics(
        candidate, reference
    )
    assert metrics["feature_cosine_similarity"] == 1.0
    assert metrics["feature_mse"] == 0.0

    payload = {"feature_session_id": "test-session", "feature_round": 1}
    references = {"per_frame": reference.cpu(), "temporal": reference.cpu()}
    assert cloud_server._store_feature_references(payload, references) is True
    with cloud_server.app.test_client() as client:
        response = client.post(
            "/feature_session/release",
            json={"feature_session_id": "test-session"},
        )
    assert response.status_code == 200
    assert response.get_json()["rounds_released"] == 1
    assert "test-session" not in cloud_server._feature_reference_sessions


def test_demo_client_forwards_delta_and_returns_result():
    class FakeResponse:
        ok = True
        encoding = None

        def iter_lines(self, decode_unicode=False):
            assert decode_unicode
            return iter(
                [
                    json.dumps({"type": "delta", "text": "你"}, ensure_ascii=False),
                    json.dumps({"type": "delta", "text": "好"}, ensure_ascii=False),
                    json.dumps(
                        {"type": "result", "result": {"response": "你好"}},
                        ensure_ascii=False,
                    ),
                ]
            )

    chunks = []
    args = SimpleNamespace(username=None, password=None, timeout=30)
    with mock.patch.object(demo_client.requests, "post", return_value=FakeResponse()):
        result = demo_client._post_cloud_stream(
            "http://127.0.0.1:8080/infer_stream",
            {"prompt": "test"},
            args,
            chunks.append,
        )

    assert chunks == ["你", "好"]
    assert result == {"response": "你好"}


def test_terminal_types_text_and_publishes_aggregated_card_immediately():
    class TtyBuffer(io.StringIO):
        def isatty(self):
            return True

    class FakeProcess:
        stdout = iter(
            [
                'DEMO_STREAM_START={"label":"基线"}\n',
                'DEMO_STREAM_DELTA={"label":"基线","text":"你"}\n',
                'DEMO_STREAM_DELTA={"label":"基线","text":"好"}\n',
                'DEMO_RESULT_ITEM={"label":"基线","response":"你好","edge_encode_ms":1,"bandwidth_delay_ms":2,"ttft_without_network_ms":3}\n',
            ]
        )

        def wait(self):
            return 0

    output = TtyBuffer()
    with (
        mock.patch.object(terminal_demo, "build_demo_command", return_value=["demo"]),
        mock.patch.object(terminal_demo.subprocess, "Popen", return_value=FakeProcess()),
        mock.patch.object(terminal_demo, "_clear_screen"),
        mock.patch.object(terminal_demo, "_clear_status"),
        mock.patch.object(terminal_demo.sys, "stdout", output),
    ):
        aggregates = {}
        assert terminal_demo.run_demo(
            SimpleNamespace(round_step_seconds=2.0),
            ["baseline"],
            aggregate_results=aggregates,
        ) == 0

    rendered = output.getvalue()
    assert "回答：你好" in rendered
    assert "第 1 轮（0s）: 你好" in rendered
    assert aggregates["baseline"]["completed_rounds"] == 1


def test_isolated_client_keeps_global_round_metadata():
    class FakeProcess:
        stdout = iter(
            [
                'DEMO_STREAM_START={"project":"baseline","label":"基线","round":1,"rounds":1,"window_start_seconds":10}\n',
                'DEMO_RESULT_ITEM={"project":"baseline","label":"基线","round":1,"rounds":1,"window_start_seconds":10,"response":"ok","frames":1,"request_bytes":10,"full_response_ms":100}\n',
            ]
        )

        def wait(self):
            return 0

    aggregates = {}
    with (
        mock.patch.object(terminal_demo, "build_demo_command", return_value=["demo"]),
        mock.patch.object(terminal_demo.subprocess, "Popen", return_value=FakeProcess()),
        mock.patch.object(terminal_demo, "_clear_status"),
    ):
        assert terminal_demo.run_demo(
            SimpleNamespace(round_step_seconds=5.0),
            ["baseline"],
            round_index=3,
            total_rounds=3,
            start_time=10.0,
            client_rounds=1,
            aggregate_results=aggregates,
        ) == 0

    row = aggregates["baseline"]
    assert row["completed_rounds"] == 1
    assert row["rounds"] == 3
    assert row["round_outputs"][0]["round"] == 3


def test_interrupt_timer_arms_after_stream_start_and_keeps_partial_text():
    class FakeProcess:
        returncode = None

        def __init__(self):
            def output():
                time.sleep(0.04)
                yield 'DEMO_STREAM_START={"project":"baseline","label":"基线"}\n'
                yield 'DEMO_STREAM_DELTA={"label":"基线","text":"部分"}\n'
                time.sleep(0.1)

            self.stdout = output()

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

        def wait(self, timeout=None):
            return self.returncode

    aggregates = {}
    with (
        mock.patch.object(terminal_demo, "build_demo_command", return_value=["demo"]),
        mock.patch.object(terminal_demo.subprocess, "Popen", return_value=FakeProcess()),
        mock.patch.object(terminal_demo, "_clear_status"),
    ):
        assert terminal_demo.run_demo(
            SimpleNamespace(round_step_seconds=5.0),
            ["baseline"],
            round_index=1,
            total_rounds=2,
            start_time=0.0,
            interrupt_after_seconds=0.02,
            aggregate_results=aggregates,
        ) == 124

    output = aggregates["baseline"]["round_outputs"][0]
    assert output["round"] == 1
    assert output["interrupted"] is True
    assert output["response"] == "部分"

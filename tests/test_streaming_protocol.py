import io
import json
from types import SimpleNamespace
from unittest import mock

from scripts import cloud_server, demo_client, terminal_demo


def test_cloud_ndjson_splits_text_into_characters():
    def task(emit):
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


def test_terminal_types_text_and_publishes_each_finished_card_immediately():
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
                'DEMO_STREAM_START={"label":"逐帧"}\n',
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
        assert terminal_demo.run_demo(SimpleNamespace(), ["baseline", "so"]) == 0

    rendered = output.getvalue()
    assert "回答：你好" in rendered
    next_start = rendered.index("正在生成：逐帧")
    assert rendered[:next_start].count("基线") >= 2

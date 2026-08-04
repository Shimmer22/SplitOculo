from unittest import mock

from scripts import cloud_server


class FakeWarmupEngine:
    def __init__(self):
        self.qwen_model = object()
        self.qwen_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        self.bottleneck = None
        self.hidden_size = 1280
        self.transmission_tokens = 49
        self.device = "cpu"
        self.native_calls = []
        self.split_calls = []

    def infer_qwen_frames_with_timing(self, frames, **kwargs):
        self.native_calls.append((frames, kwargs))
        return "", {}

    def infer_video_from_frame_features_with_timing(self, features, **kwargs):
        self.split_calls.append((features, kwargs))
        return "", {}


def test_compute_warmup_runs_native_and_split_once_then_uses_cache():
    engine = FakeWarmupEngine()
    payload = {
        "projects": ["baseline", "so"],
        "max_frames": 8,
        "video_pixel_budget": 224 * 224,
        "video_fps": 2.0,
    }
    with mock.patch.object(cloud_server, "engine", engine):
        client = cloud_server.app.test_client()
        first = client.post("/warmup", json=payload)
        second = client.post("/warmup", json=payload)

    assert first.status_code == 200
    assert first.get_json()["compute_warmed"] is True
    assert first.get_json()["paths"]["native_qwen"]["cached"] is False
    assert first.get_json()["paths"]["splitoculo"]["cached"] is False
    assert second.get_json()["paths"]["native_qwen"]["cached"] is True
    assert second.get_json()["paths"]["splitoculo"]["cached"] is True
    assert len(engine.native_calls) == 1
    assert len(engine.native_calls[0][0]) == 2
    assert len(engine.split_calls) == 1
    assert tuple(engine.split_calls[0][0].shape) == (2, 49, 1280)

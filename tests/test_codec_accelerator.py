"""Pure helper checks for decoder-motion-vector rasterization and warp."""

from types import SimpleNamespace
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.codec_accelerator import CodecWarpAccelerator, DecoderMotionVectorAccelerator
from models.codec_memory import LSFAFeatureMemory, MMNetFeatureMemory


MV_DTYPE = np.dtype({
    "names": [
        "source", "w", "h", "src_x", "src_y", "dst_x", "dst_y",
        "flags", "motion_x", "motion_y", "motion_scale",
    ],
    "formats": ["<i4", "u1", "u1", "<i2", "<i2", "<i2", "<i2", "<u8", "<i4", "<i4", "<u2"],
})


def test_decoder_mv_rasterization_uses_backward_flow():
    vectors = np.zeros(1, dtype=MV_DTYPE)
    vectors[0]["source"] = -1
    vectors[0]["w"] = 4
    vectors[0]["h"] = 4
    vectors[0]["dst_x"] = 2
    vectors[0]["dst_y"] = 2
    vectors[0]["motion_x"] = 8
    vectors[0]["motion_y"] = -4
    vectors[0]["motion_scale"] = 4
    flow, covered = DecoderMotionVectorAccelerator._dense_original_flow(vectors, 4, 4)
    assert covered.all()
    assert torch.allclose(flow[0], torch.full((4, 4), 2.0))
    assert torch.allclose(flow[1], torch.full((4, 4), -1.0))


def test_future_reference_vectors_are_not_used():
    vectors = np.zeros(1, dtype=MV_DTYPE)
    vectors[0]["source"] = 1
    vectors[0]["w"] = vectors[0]["h"] = 4
    vectors[0]["dst_x"] = vectors[0]["dst_y"] = 2
    vectors[0]["motion_x"] = 8
    vectors[0]["motion_scale"] = 4
    flow, covered = DecoderMotionVectorAccelerator._dense_original_flow(vectors, 4, 4)
    assert not covered.any()
    assert not flow.any()


def test_zero_flow_preserves_feature():
    accelerator = CodecWarpAccelerator.__new__(CodecWarpAccelerator)
    accelerator._grid_cache = {}
    feature = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    flow = torch.zeros(2, 8, 8)
    warped = accelerator._warp_feature(feature, flow)
    assert torch.allclose(warped, feature, atol=1e-5)


def test_direct_feature_grid_flow_uses_feature_cell_units():
    accelerator = DecoderMotionVectorAccelerator.__new__(DecoderMotionVectorAccelerator)
    accelerator.edge = SimpleNamespace(image_size=4)
    vectors = np.zeros(1, dtype=MV_DTYPE)
    vectors[0]["source"] = -1
    vectors[0]["w"] = vectors[0]["h"] = 4
    vectors[0]["dst_x"] = vectors[0]["dst_y"] = 2
    vectors[0]["motion_x"] = 8
    vectors[0]["motion_y"] = -4
    vectors[0]["motion_scale"] = 4
    flow, covered = accelerator._feature_grid_flow(vectors, 4, 4, 2, 2)
    assert covered.all()
    assert torch.allclose(flow[0], torch.full((2, 2), 1.0))
    assert torch.allclose(flow[1], torch.full((2, 2), -0.5))


def test_feature_grid_flow_matches_legacy_bilinear_pipeline():
    accelerator = DecoderMotionVectorAccelerator.__new__(DecoderMotionVectorAccelerator)
    accelerator.edge = SimpleNamespace(image_size=4)
    vectors = np.zeros(2, dtype=MV_DTYPE)
    vectors["source"] = -1
    vectors["w"] = 4
    vectors["h"] = 6
    vectors["dst_x"] = [2, 6]
    vectors["dst_y"] = 3
    vectors["motion_x"] = [8, -4]
    vectors["motion_y"] = [4, -8]
    vectors["motion_scale"] = 4

    dense, dense_covered = accelerator._dense_original_flow(vectors, 8, 6)
    cropped, _ = accelerator._flow_to_model_crop(dense, dense_covered, 8, 6)
    expected = torch.nn.functional.interpolate(
        cropped[None], size=(2, 2), mode="bilinear", align_corners=False
    )[0]
    expected[0] *= 2 / 4
    expected[1] *= 2 / 4
    actual, covered = accelerator._feature_grid_flow(vectors, 8, 6, 2, 2)
    assert covered.all()
    assert torch.allclose(actual, expected, atol=1e-6)


def test_keyframe_flow_composes_backward_translations():
    accelerator = DecoderMotionVectorAccelerator.__new__(
        DecoderMotionVectorAccelerator
    )
    accelerator._grid_cache = {}
    accelerator.cumulative_flow = torch.zeros(2, 4, 4)
    accelerator.cumulative_flow[0] = 1.0
    accelerator.cumulative_covered = torch.ones(4, 4, dtype=torch.bool)
    current_flow = torch.zeros(2, 4, 4)
    current_flow[1] = -0.5
    composed, covered = accelerator._keyframe_flow(
        current_flow, torch.ones(4, 4, dtype=torch.bool)
    )
    assert covered.all()
    assert torch.allclose(composed[0], torch.ones(4, 4), atol=1e-6)
    assert torch.allclose(composed[1], torch.full((4, 4), -0.5), atol=1e-6)


def test_resize_crop_flow_matches_model_input_shape():
    accelerator = DecoderMotionVectorAccelerator.__new__(DecoderMotionVectorAccelerator)
    accelerator.edge = SimpleNamespace(image_size=224)
    flow = torch.zeros(2, 240, 320)
    flow[0] = 10
    covered = torch.ones(240, 320, dtype=torch.bool)
    cropped_flow, cropped_covered = accelerator._flow_to_model_crop(
        flow, covered, original_width=320, original_height=240
    )
    assert cropped_flow.shape == (2, 224, 224)
    assert cropped_covered.shape == (224, 224)
    assert cropped_covered.all()
    assert 9.0 < float(cropped_flow[0].mean()) < 10.0


def test_mmnet_memory_is_identity_before_training():
    memory = MMNetFeatureMemory(feature_channels=4)
    warped = torch.randn(1, 4, 7, 7)
    residual = torch.randn(1, 3, 7, 7)
    motion = torch.randn(1, 2, 7, 7)
    covered = torch.ones(1, 1, 7, 7)
    corrected = memory(warped, residual, motion, covered)
    assert torch.allclose(corrected, warped, atol=1e-6)


def test_lsfa_memory_is_identity_before_training():
    memory = LSFAFeatureMemory(feature_channels=4)
    warped = torch.randn(1, 4, 7, 7)
    residual = torch.randn(1, 3, 7, 7)
    current_rgb = torch.rand(1, 3, 7, 7)
    motion = torch.randn(1, 2, 7, 7)
    covered = torch.ones(1, 1, 7, 7)
    corrected = memory(warped, residual, current_rgb, motion, covered)
    assert torch.allclose(corrected, warped, atol=1e-6)


def test_mmnet_path_and_periodic_refresh_guard():
    class DummyImage:
        size = (4, 4)

        def convert(self, _mode):
            return self

    class DummyStudent(nn.Module):
        def forward(self, image):
            return [image]

    edge = SimpleNamespace(
        device="cpu",
        image_size=4,
        transform=lambda _image: torch.zeros(3, 4, 4),
        student=DummyStudent(),
        projector=nn.Identity(),
        bottleneck=None,
    )
    accelerator = DecoderMotionVectorAccelerator(
        edge,
        flow_impl="feature_grid",
        max_p_chain=1,
    )
    accelerator.memory = MMNetFeatureMemory(feature_channels=3)
    image = DummyImage()
    i_record = {
        "source_index": 0,
        "pts": 0,
        "time_seconds": 0.0,
        "pict_type": "I",
        "selected": True,
        "image": image,
        "motion_vectors": None,
    }
    vectors = np.zeros(1, dtype=MV_DTYPE)
    vectors[0]["source"] = -1
    vectors[0]["w"] = vectors[0]["h"] = 4
    vectors[0]["dst_x"] = vectors[0]["dst_y"] = 2
    vectors[0]["motion_scale"] = 1
    p_record = dict(i_record)
    p_record.update(
        source_index=1,
        pts=1,
        time_seconds=1.0,
        pict_type="P",
        motion_vectors=vectors,
    )

    _, _, i_info = accelerator.encode_record(i_record)
    _, _, p_info = accelerator.encode_record(p_record)
    _, _, refresh_info = accelerator.encode_record(dict(p_record, source_index=2, pts=2))

    assert i_info["mode"] == "I"
    assert p_info["mode"] == "P_MMNET"
    assert p_info["memory_executed"]
    assert refresh_info["mode"] == "P_FULL_FALLBACK"
    assert refresh_info["fallback_reason"] == "max_p_chain:1"


def test_lsfa_path_uses_current_image_branch():
    class DummyImage:
        size = (4, 4)

        def convert(self, _mode):
            return self

    class DummyStudent(nn.Module):
        def forward(self, image):
            return [image]

    edge = SimpleNamespace(
        device="cpu",
        image_size=4,
        transform=lambda _image: torch.zeros(3, 4, 4),
        student=DummyStudent(),
        projector=nn.Identity(),
        bottleneck=None,
    )
    accelerator = DecoderMotionVectorAccelerator(
        edge,
        flow_impl="feature_grid",
        memory_arch="lsfa",
    )
    accelerator.memory = LSFAFeatureMemory(feature_channels=3)
    image = DummyImage()
    i_record = {
        "source_index": 0,
        "pts": 0,
        "time_seconds": 0.0,
        "pict_type": "I",
        "selected": True,
        "image": image,
        "motion_vectors": None,
    }
    vectors = np.zeros(1, dtype=MV_DTYPE)
    vectors[0]["source"] = -1
    vectors[0]["w"] = vectors[0]["h"] = 4
    vectors[0]["dst_x"] = vectors[0]["dst_y"] = 2
    vectors[0]["motion_scale"] = 1
    p_record = dict(i_record)
    p_record.update(
        source_index=1,
        pts=1,
        time_seconds=1.0,
        pict_type="P",
        motion_vectors=vectors,
    )

    _, _, p_info = accelerator.encode_record(i_record)
    _, _, p_info = accelerator.encode_record(p_record)
    assert p_info["mode"] == "P_LSFA"
    assert p_info["memory_arch"] == "lsfa"
    assert p_info["memory_executed"]


if __name__ == "__main__":
    test_decoder_mv_rasterization_uses_backward_flow()
    test_future_reference_vectors_are_not_used()
    test_zero_flow_preserves_feature()
    test_direct_feature_grid_flow_uses_feature_cell_units()
    test_feature_grid_flow_matches_legacy_bilinear_pipeline()
    test_keyframe_flow_composes_backward_translations()
    test_resize_crop_flow_matches_model_input_shape()
    test_mmnet_memory_is_identity_before_training()
    test_lsfa_memory_is_identity_before_training()
    test_mmnet_path_and_periodic_refresh_guard()
    test_lsfa_path_uses_current_image_branch()
    print("codec accelerator helper tests passed")

import torch

from models.temporal_pair import TemporalPairFusion


def test_temporal_pair_starts_as_mean():
    fusion = TemporalPairFusion(in_channels=16, hidden_channels=8)
    frame0 = torch.randn(2, 16, 7, 7)
    frame1 = torch.randn(2, 16, 7, 7)
    actual = fusion(frame0, frame1)
    expected = (frame0 + frame1) * 0.5
    torch.testing.assert_close(actual, expected)


def test_repeated_still_preserves_feature():
    fusion = TemporalPairFusion(in_channels=8, hidden_channels=8)
    image = torch.randn(1, 8, 4, 4)
    torch.testing.assert_close(fusion(image, image), image)

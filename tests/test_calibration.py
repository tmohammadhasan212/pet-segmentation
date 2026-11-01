import numpy as np
import pytest
import torch

from petseg.calibration import THRESHOLDS, threshold_scores
from petseg.metrics import region_metrics
from petseg.prediction import probabilities, restore_probability


def test_flip_averaging_aligns_predictions():
    class Echo(torch.nn.Module):
        def forward(self, x):
            return x[:, :1]

    x = torch.randn(2, 3, 32, 64)
    assert torch.allclose(probabilities(Echo(), x, True), x[:, :1].sigmoid())


@pytest.mark.parametrize("empty", [False, True])
def test_threshold_sweep_matches_masked_region_metrics(empty):
    p = np.array([[0.2, 0.3, 0.5], [0.6, 0.7, 0.8]], dtype=np.float32)
    y = np.zeros_like(p, bool) if empty else np.array([[1, 0, 1], [1, 0, 0]], bool)
    valid = np.array([[1, 0, 1], [1, 1, 1]], bool)
    ious, dice = threshold_scores(p, y, valid)
    for index, threshold in enumerate(THRESHOLDS):
        reference = region_metrics(
            torch.from_numpy(p),
            torch.from_numpy(y),
            torch.from_numpy(valid),
            float(threshold),
        )
        assert ious[index] == pytest.approx(reference["iou"])
        assert dice[index] == pytest.approx(reference["dice"])
    assert threshold_scores(p, y, np.zeros_like(valid)) is None


def test_inverse_projection_removes_ignored_padding():
    p = np.ones((64, 64), np.float32)
    p[16:48] = 0.25
    restored = restore_probability(p, (0, 16, 64, 48), (200, 100))
    assert restored.shape == (100, 200)
    assert np.allclose(restored, 0.25)

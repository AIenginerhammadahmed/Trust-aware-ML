"""
Unit tests for calibration metrics and temperature scaling.
"""

import unittest
import numpy as np
import torch

from src.calibration.metrics import (
    binary_expected_calibration_error,
    class_conditional_ece,
    brier_score_loss
)
from src.calibration.temperature import TemperatureScaler, fit_temperature_scaling


class TestCalibration(unittest.TestCase):

    def test_perfect_calibration_ece_is_zero(self):
        """When empirical accuracy matches predicted probability exactly, ECE should be near 0."""
        # 100 samples with prob 0.2 where exactly 20 are 1
        probs_1 = np.full(100, 0.2)
        y_1 = np.zeros(100)
        y_1[:20] = 1

        # 100 samples with prob 0.8 where exactly 80 are 1
        probs_2 = np.full(100, 0.8)
        y_2 = np.zeros(100)
        y_2[:80] = 1

        y_all = np.concatenate([y_1, y_2])
        p_all = np.concatenate([probs_1, probs_2])

        ece = binary_expected_calibration_error(y_all, p_all, n_bins=10)
        self.assertAlmostEqual(ece, 0.0, places=5)

    def test_brier_score_loss(self):
        y_true = np.array([1, 0, 1, 0])
        probs = np.array([1.0, 0.0, 1.0, 0.0])
        self.assertEqual(brier_score_loss(y_true, probs), 0.0)

        # Inverted predictions
        probs_bad = np.array([0.0, 1.0, 0.0, 1.0])
        self.assertEqual(brier_score_loss(y_true, probs_bad), 1.0)

    def test_temperature_scaler(self):
        scaler = TemperatureScaler(initial_temp=2.0)
        logits = torch.tensor([2.0, -4.0])
        scaled = scaler(logits)
        self.assertTrue(torch.allclose(scaled, torch.tensor([1.0, -2.0])))


if __name__ == "__main__":
    unittest.main()

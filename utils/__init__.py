"""
Utility shortcuts for metrics, visualization, and data loading.
"""

from src.calibration.metrics import binary_expected_calibration_error, class_conditional_ece, brier_score_loss
from src.uncertainty.mc_dropout import mc_dropout_predict, identify_failure_modes
from src.visualization.plots import plot_reliability_diagram, plot_risk_coverage_curves

__all__ = [
    "binary_expected_calibration_error",
    "class_conditional_ece",
    "brier_score_loss",
    "mc_dropout_predict",
    "identify_failure_modes",
    "plot_reliability_diagram",
    "plot_risk_coverage_curves"
]

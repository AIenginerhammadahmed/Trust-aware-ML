"""
Calibration evaluation metrics for binary classification under heavy class imbalance.
Implements correct Binary ECE, Class-Conditional ECE, Adaptive/Quantile ECE, and Brier Score.
"""

from typing import Tuple, Dict, Any
import numpy as np


def binary_expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform"
) -> float:
    """
    Computes Expected Calibration Error (ECE) for binary probabilistic predictions.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        probs: Predicted probability of class 1 in [0, 1].
        n_bins: Number of bins.
        strategy: 'uniform' (equal-width [0, 1/n, 2/n...]) or 'quantile' (equal-frequency).

    Returns:
        Weighted average difference between mean confidence and empirical accuracy across bins.
    """
    y_true = np.asarray(y_true).ravel()
    probs = np.asarray(probs).ravel()
    n_samples = len(y_true)

    if n_samples == 0:
        return 0.0

    if strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_assignments = np.digitize(probs, bins) - 1
        bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)
    elif strategy == "quantile":
        # Quantile binning ensures each bin has approx equal sample count
        quantiles = np.linspace(0.0, 100.0, n_bins + 1)
        bins = np.percentile(probs, quantiles)
        bins[0] = 0.0
        bins[-1] = 1.0
        bin_assignments = np.digitize(probs, bins) - 1
        bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    ece = 0.0
    for b in range(n_bins):
        mask = (bin_assignments == b)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_conf = np.mean(probs[mask])
            bin_acc = np.mean(y_true[mask])  # Empirical positive rate in bin
            ece += (bin_count / n_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def class_conditional_ece(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Computes Class-Conditional ECE for negative (legitimate) and positive (fraud) instances.
    Crucial for imbalanced domains where standard ECE is dominated by majority class 0.
    """
    y_true = np.asarray(y_true).ravel()
    probs = np.asarray(probs).ravel()

    neg_mask = (y_true == 0)
    pos_mask = (y_true == 1)

    ece_neg = binary_expected_calibration_error(y_true[neg_mask], probs[neg_mask], n_bins=n_bins) if neg_mask.sum() > 0 else 0.0
    ece_pos = binary_expected_calibration_error(y_true[pos_mask], probs[pos_mask], n_bins=n_bins) if pos_mask.sum() > 0 else 0.0
    ece_overall = binary_expected_calibration_error(y_true, probs, n_bins=n_bins)

    return {
        "ece_overall": float(ece_overall),
        "ece_non_fraud": float(ece_neg),
        "ece_fraud": float(ece_pos)
    }


def brier_score_loss(
    y_true: np.ndarray,
    probs: np.ndarray
) -> float:
    """Computes mean squared error between predicted probabilities and binary ground truth."""
    y_true = np.asarray(y_true).ravel()
    probs = np.asarray(probs).ravel()
    return float(np.mean((probs - y_true) ** 2))


def reliability_diagram_data(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10
) -> Dict[str, np.ndarray]:
    """
    Extracts bin midpoints, mean confidences, empirical accuracies, and sample counts
    for plotting reliability diagrams.
    """
    y_true = np.asarray(y_true).ravel()
    probs = np.asarray(probs).ravel()

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confs = []
    bin_accs = []
    bin_counts = []
    bin_edges = []

    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        mask = (probs >= low) & (probs <= high if i == n_bins - 1 else probs < high)
        count = int(np.sum(mask))
        bin_counts.append(count)
        bin_edges.append((low + high) / 2.0)
        if count > 0:
            bin_confs.append(float(np.mean(probs[mask])))
            bin_accs.append(float(np.mean(y_true[mask])))
        else:
            bin_confs.append((low + high) / 2.0)
            bin_accs.append(0.0)

    return {
        "bin_centers": np.array(bin_edges),
        "mean_conf": np.array(bin_confs),
        "empirical_acc": np.array(bin_accs),
        "counts": np.array(bin_counts),
        "bins": bins
    }

"""
Bayesian approximation via Monte Carlo Dropout (Gal & Ghahramani, ICML 2016).
Provides epistemic uncertainty quantification and failure mode profiling for high-stakes fraud detection.
"""

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn


def mc_dropout_predict(
    model: nn.Module,
    X_tensor: torch.Tensor,
    n_samples: int = 50,
    device: Optional[torch.device] = None,
    batch_size: int = 1024
) -> Dict[str, np.ndarray]:
    """
    Perform Monte Carlo Dropout sampling over input data.
    Maintains active dropout masks during inference to sample from the approximate
    posterior distribution over weights.

    Args:
        model: PyTorch model with Dropout layers.
        X_tensor: Input features as float tensor.
        n_samples: Number of stochastic forward passes T (default 50).
        device: PyTorch device.
        batch_size: Batch size for batched forward passes.

    Returns:
        Dictionary containing:
            - mc_preds: Raw samples (T, N)
            - mean_probs: Mean predicted probability (N,)
            - std_epistemic: Epistemic standard deviation (N,)
            - entropy_total: Total predictive Shannon entropy (N,)
            - mutual_information: Epistemic uncertainty via mutual information (N,)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.train()  # CRITICAL: keep dropout active

    n_total = len(X_tensor)
    all_sample_preds = []

    # Run n_samples forward passes
    with torch.no_grad():
        for s in range(n_samples):
            sample_probs = []
            for i in range(0, n_total, batch_size):
                batch_x = X_tensor[i:i + batch_size].to(device)
                logits = model(batch_x)
                probs = torch.sigmoid(logits).cpu().numpy().ravel()
                sample_probs.extend(probs)
            all_sample_preds.append(sample_probs)

    mc_preds = np.array(all_sample_preds)  # Shape: (T, N)

    # Predictive mean and standard deviation
    mean_probs = np.mean(mc_preds, axis=0)
    std_epistemic = np.std(mc_preds, axis=0)

    # Information-theoretic decomposition
    eps = 1e-7
    # 1. Total predictive entropy H(Y|X):
    p_clamped = np.clip(mean_probs, eps, 1.0 - eps)
    total_entropy = -(p_clamped * np.log2(p_clamped) + (1.0 - p_clamped) * np.log2(1.0 - p_clamped))
    total_entropy = np.nan_to_num(total_entropy, nan=0.0)

    # 2. Expected data entropy E_w[H(Y|X, w)] (Aleatoric uncertainty):
    sample_clamped = np.clip(mc_preds, eps, 1.0 - eps)
    sample_entropies = -(sample_clamped * np.log2(sample_clamped) + (1.0 - sample_clamped) * np.log2(1.0 - sample_clamped))
    sample_entropies = np.nan_to_num(sample_entropies, nan=0.0)
    expected_data_entropy = np.mean(sample_entropies, axis=0)

    # 3. Mutual Information I(Y, W | X) = H(Y|X) - E_w[H(Y|X, w)] (Pure Epistemic uncertainty):
    mutual_info = np.maximum(0.0, total_entropy - expected_data_entropy)

    return {
        "mc_preds": mc_preds,
        "mean_probs": mean_probs,
        "std_epistemic": std_epistemic,
        "entropy_total": total_entropy,
        "mutual_information": mutual_info
    }


def compute_uncertainty_metrics(
    mc_results: Dict[str, np.ndarray],
    y_true: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Compute summary statistics for uncertainty across correct vs incorrect predictions."""
    y_true = np.asarray(y_true).ravel()
    preds = (mc_results["mean_probs"] >= threshold).astype(int)
    correct_mask = (preds == y_true)
    incorrect_mask = ~correct_mask

    std = mc_results["std_epistemic"]
    mi = mc_results["mutual_information"]

    return {
        "mean_std_overall": float(np.mean(std)),
        "mean_std_correct": float(np.mean(std[correct_mask])) if np.any(correct_mask) else 0.0,
        "mean_std_incorrect": float(np.mean(std[incorrect_mask])) if np.any(incorrect_mask) else 0.0,
        "mean_mi_correct": float(np.mean(mi[correct_mask])) if np.any(correct_mask) else 0.0,
        "mean_mi_incorrect": float(np.mean(mi[incorrect_mask])) if np.any(incorrect_mask) else 0.0,
    }


def identify_failure_modes(
    y_true: np.ndarray,
    probs: np.ndarray,
    uncertainty: np.ndarray,
    conf_threshold: float = 0.90,
    uncertainty_threshold: float = 0.10
) -> Dict[str, Any]:
    """
    Profile high-risk failure modes:
    - Silent Killers: High Confidence (>90%), Wrong Prediction, LOW Uncertainty (<= tau).
      These are the most catastrophic failure mode in automated decision engines.
    - Recoverable Errors: High Confidence (>90%), Wrong Prediction, HIGH Uncertainty (> tau).
      These can be saved by uncertainty-guided abstention.
    """
    y_true = np.asarray(y_true).ravel()
    probs = np.asarray(probs).ravel()
    uncertainty = np.asarray(uncertainty).ravel()

    confidence = np.maximum(probs, 1.0 - probs)
    preds = (probs >= 0.5).astype(int)
    is_wrong = (preds != y_true)
    is_high_conf = (confidence >= conf_threshold)

    high_conf_wrong = is_high_conf & is_wrong
    silent_killers = high_conf_wrong & (uncertainty <= uncertainty_threshold)
    recoverable_errors = high_conf_wrong & (uncertainty > uncertainty_threshold)

    return {
        "total_samples": len(y_true),
        "total_errors": int(np.sum(is_wrong)),
        "high_conf_wrong_count": int(np.sum(high_conf_wrong)),
        "silent_killer_count": int(np.sum(silent_killers)),
        "recoverable_error_count": int(np.sum(recoverable_errors)),
        "silent_killer_indices": np.where(silent_killers)[0],
        "recoverable_indices": np.where(recoverable_errors)[0]
    }

"""
Post-hoc calibration methods: Temperature Scaling and Platt Scaling.
Guo et al. (ICML 2017) formulation adapted for binary sigmoid classification.
"""

from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn


class TemperatureScaler(nn.Module):
    """
    Temperature Scaling for binary classification (Guo et al., 2017).
    Calibrates logits by a single scalar parameter T > 0:
        p_calibrated = sigmoid(logit / T)
    Preserves model accuracy and ROC-AUC while softening/sharpening overconfident probabilities.
    """
    def __init__(self, initial_temp: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Clamp temperature to avoid division by zero or negative scaling
        clamped_temp = torch.clamp(self.temperature, min=1e-3, max=100.0)
        return logits / clamped_temp


def fit_temperature_scaling(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    lr: float = 0.01,
    max_iter: int = 50,
    device: Optional[torch.device] = None
) -> Tuple[TemperatureScaler, float]:
    """
    Learn optimal temperature T on an independent validation set using L-BFGS optimizer.

    Args:
        val_logits: Validation set unscaled logits.
        val_labels: Validation set ground truth binary labels (0 or 1).
        lr: Learning rate for L-BFGS.
        max_iter: Maximum optimization iterations.

    Returns:
        Fitted TemperatureScaler and the optimal scalar temperature value.
    """
    if device is None:
        device = torch.device("cpu")

    scaler = TemperatureScaler().to(device)
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter)
    criterion = nn.BCEWithLogitsLoss()

    logits_t = torch.tensor(val_logits, dtype=torch.float32, device=device).reshape(-1, 1)
    labels_t = torch.tensor(val_labels, dtype=torch.float32, device=device).reshape(-1, 1)

    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler(logits_t), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    opt_temp = float(scaler.temperature.detach().cpu().item())
    return scaler, opt_temp

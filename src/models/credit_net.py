"""
PyTorch neural network architecture for credit card fraud detection.
Equipped with Dropout layers for both regularization and Monte Carlo epistemic uncertainty estimation.
"""

from typing import Optional
import torch
import torch.nn as nn


class CreditNet(nn.Module):
    """
    Multi-layer Perceptron with Dropout for Credit Card Fraud Detection.
    Follows Gal & Ghahramani (2016) formulation where Dropout layers remain
    active during inference to approximate deep Gaussian processes.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple = (64, 32),
        dropout_rate: float = 0.3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw logit output."""
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic sigmoid probabilities (standard eval mode)."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        if was_training:
            self.train()
        return probs

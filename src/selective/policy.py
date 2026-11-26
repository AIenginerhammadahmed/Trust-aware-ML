"""
Dual-gated trust policy for selective classification (Geifman & El-Yaniv, NeurIPS 2017).
Combines probability-based confidence (aleatoric risk) and MC Dropout variance (epistemic uncertainty).
"""

from typing import Dict, Any, Optional
import numpy as np


class DualGatedTrustPolicy:
    """
    Automated decision orchestrator:
    - Approves or Declines transactions automatically ONLY when trust is high.
    - Abstained / deferred transactions are routed to Human Tier-2 Fraud Investigators.

    Rejection conditions:
    1. Epistemic Rejection: std_epistemic > tau_sigma (model does not recognize the pattern)
    2. Aleatoric Rejection: confidence < tau_conf (transaction lies dangerously close to decision boundary)
    """
    def __init__(
        self,
        tau_sigma: float = 0.10,
        tau_conf: float = 0.80,
        decision_threshold: float = 0.50
    ):
        self.tau_sigma = tau_sigma
        self.tau_conf = tau_conf
        self.decision_threshold = decision_threshold

    def evaluate(
        self,
        probs: np.ndarray,
        std_epistemic: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Apply dual-gated rules on predicted probabilities and epistemic uncertainties.

        Returns:
            Dictionary containing:
                - decisions: Array of strings ('AUTO_FRAUD', 'AUTO_LEGITIMATE', 'HUMAN_AUDIT')
                - accepted_mask: Boolean mask of samples where model automated the decision
                - abstained_mask: Boolean mask of samples routed to human review
                - predictions: Integer predictions for accepted samples (-1 for abstained)
        """
        probs = np.asarray(probs).ravel()
        std_epistemic = np.asarray(std_epistemic).ravel()

        confidence = np.maximum(probs, 1.0 - probs)

        epistemic_abstain = (std_epistemic > self.tau_sigma)
        aleatoric_abstain = (confidence < self.tau_conf)

        abstained_mask = epistemic_abstain | aleatoric_abstain
        accepted_mask = ~abstained_mask

        predictions = np.full(len(probs), fill_value=-1, dtype=int)
        predictions[accepted_mask] = (probs[accepted_mask] >= self.decision_threshold).astype(int)

        decisions = np.empty(len(probs), dtype=object)
        decisions[abstained_mask] = "HUMAN_AUDIT"
        decisions[accepted_mask & (predictions == 1)] = "AUTO_FRAUD"
        decisions[accepted_mask & (predictions == 0)] = "AUTO_LEGITIMATE"

        return {
            "decisions": decisions,
            "accepted_mask": accepted_mask,
            "abstained_mask": abstained_mask,
            "predictions": predictions,
            "epistemic_abstain_mask": epistemic_abstain,
            "aleatoric_abstain_mask": aleatoric_abstain,
            "coverage": float(np.mean(accepted_mask))
        }

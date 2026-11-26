"""
Risk-Coverage dynamics and Cost-Sensitive Financial Loss modeling.
Geifman & El-Yaniv (2017) selective classification curves tailored for imbalanced fraud detection.
"""

from typing import Tuple, Dict, Any, List
import numpy as np


class FinancialCostMatrix:
    """
    Financial cost model for credit card transaction auditing.
    - cost_fn: Undetected fraudulent transaction losses (chargebacks, loss of goods).
    - cost_fp: False alarm cost (customer friction, customer support call).
    - cost_audit: Tier-2 human manual audit cost.
    """
    def __init__(
        self,
        cost_fn: float = 250.0,
        cost_fp: float = 15.0,
        cost_audit: float = 5.0
    ):
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp
        self.cost_audit = cost_audit

    def compute_total_cost(
        self,
        y_true: np.ndarray,
        predictions: np.ndarray,
        abstained_mask: np.ndarray
    ) -> Dict[str, float]:
        """Compute aggregate cost in dollars."""
        y_true = np.asarray(y_true).ravel()
        predictions = np.asarray(predictions).ravel()
        abstained_mask = np.asarray(abstained_mask).ravel()

        accepted_mask = ~abstained_mask

        # For accepted decisions:
        accepted_preds = predictions[accepted_mask]
        accepted_true = y_true[accepted_mask]

        fn_count = np.sum((accepted_preds == 0) & (accepted_true == 1))
        fp_count = np.sum((accepted_preds == 1) & (accepted_true == 0))
        audit_count = np.sum(abstained_mask)

        cost_from_fn = fn_count * self.cost_fn
        cost_from_fp = fp_count * self.cost_fp
        cost_from_audit = audit_count * self.cost_audit
        total_cost = cost_from_fn + cost_from_fp + cost_from_audit

        return {
            "total_cost": float(total_cost),
            "cost_from_fn": float(cost_from_fn),
            "cost_from_fp": float(cost_from_fp),
            "cost_from_audit": float(cost_from_audit),
            "fn_count": int(fn_count),
            "fp_count": int(fp_count),
            "audit_count": int(audit_count),
            "avg_cost_per_tx": float(total_cost / max(1, len(y_true)))
        }


def compute_risk_coverage_curve(
    y_true: np.ndarray,
    probs: np.ndarray,
    uncertainty: np.ndarray,
    n_points: int = 100,
    decision_threshold: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Compute selective classification Risk vs Coverage curve.
    Uses monotonic thresholds on uncertainty (from strictest to most lenient).

    Avoids undefined metric zero-division warnings by handling sparse positive subsets safely.
    """
    y_true = np.asarray(y_true).ravel()
    probs = np.asarray(probs).ravel()
    uncertainty = np.asarray(uncertainty).ravel()

    # Thresholds from lowest uncertainty (most conservative) to maximum
    u_min, u_max = float(np.min(uncertainty)), float(np.max(uncertainty))
    thresholds = np.linspace(u_min, u_max, n_points)

    coverages = []
    risks = []
    recalls = []
    precisions = []

    total_positives = max(1, int(np.sum(y_true == 1)))

    for t in thresholds:
        keep = (uncertainty <= t)
        cov = float(np.mean(keep))
        coverages.append(cov)

        if np.sum(keep) == 0:
            risks.append(0.0)
            recalls.append(0.0)
            precisions.append(1.0)
            continue

        y_sub = y_true[keep]
        p_sub = (probs[keep] >= decision_threshold).astype(int)

        # Risk = 1 - Accuracy on accepted subset
        accuracy = float(np.mean(y_sub == p_sub))
        risk = 1.0 - accuracy
        risks.append(risk)

        tp = np.sum((p_sub == 1) & (y_sub == 1))
        fn = np.sum((p_sub == 0) & (y_sub == 1))
        fp = np.sum((p_sub == 1) & (y_sub == 0))

        # Fraud Recall among retained subset
        sub_positives = tp + fn
        recall = float(tp / sub_positives) if sub_positives > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0

        recalls.append(recall)
        precisions.append(precision)

    return {
        "thresholds": thresholds,
        "coverage": np.array(coverages),
        "risk": np.array(risks),
        "recall": np.array(recalls),
        "precision": np.array(precisions)
    }


def compute_cost_coverage_curve(
    y_true: np.ndarray,
    probs: np.ndarray,
    uncertainty: np.ndarray,
    cost_matrix: FinancialCostMatrix,
    n_points: int = 100,
    decision_threshold: float = 0.5
) -> Dict[str, np.ndarray]:
    """Compute financial dollar cost as a function of coverage."""
    y_true = np.asarray(y_true).ravel()
    probs = np.asarray(probs).ravel()
    uncertainty = np.asarray(uncertainty).ravel()

    u_min, u_max = float(np.min(uncertainty)), float(np.max(uncertainty))
    thresholds = np.linspace(u_min, u_max, n_points)

    coverages = []
    total_costs = []
    avg_costs = []

    for t in thresholds:
        abstained = (uncertainty > t)
        cov = float(np.mean(~abstained))
        coverages.append(cov)

        preds = (probs >= decision_threshold).astype(int)
        res = cost_matrix.compute_total_cost(y_true, preds, abstained)
        total_costs.append(res["total_cost"])
        avg_costs.append(res["avg_cost_per_tx"])

    return {
        "thresholds": thresholds,
        "coverage": np.array(coverages),
        "total_costs": np.array(total_costs),
        "avg_costs": np.array(avg_costs)
    }

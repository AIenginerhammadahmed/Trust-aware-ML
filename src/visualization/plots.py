"""
Publication-quality plotting functions for calibration, uncertainty, and selective prediction.
Outputs publication-ready figures for reports and papers.
"""

from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_reliability_diagram(
    rel_uncal: Dict[str, np.ndarray],
    rel_cal: Optional[Dict[str, np.ndarray]] = None,
    ece_uncal: Optional[float] = None,
    ece_cal: Optional[float] = None,
    title: str = "Reliability Diagram (Calibration Comparison)",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate reliability diagram comparing uncalibrated vs calibrated model.
    Includes lower subplot showing sample counts per confidence bin.
    """
    fig, (ax_main, ax_hist) = plt.subplots(
        2, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # Diagonal reference: Perfect calibration
    ax_main.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfect Calibration")

    # Uncalibrated curve
    label_uncal = f"Before Calibration (ECE = {ece_uncal:.4f})" if ece_uncal is not None else "Before Calibration"
    ax_main.plot(
        rel_uncal["mean_conf"],
        rel_uncal["empirical_acc"],
        marker="o",
        linewidth=2,
        color="#d9534f",
        label=label_uncal
    )

    # Calibrated curve
    if rel_cal is not None:
        label_cal = f"After Temp Scaling (ECE = {ece_cal:.4f})" if ece_cal is not None else "After Calibration"
        ax_main.plot(
            rel_cal["mean_conf"],
            rel_cal["empirical_acc"],
            marker="s",
            linewidth=2,
            color="#0275d8",
            label=label_cal
        )

    ax_main.set_ylabel("Empirical Accuracy", fontsize=12)
    ax_main.set_title(title, fontsize=13, fontweight="bold")
    ax_main.legend(loc="upper left", frameon=True)
    ax_main.grid(True, linestyle=":", alpha=0.6)
    ax_main.set_ylim(-0.02, 1.02)

    # Bin counts histogram
    bin_centers = rel_uncal["bin_centers"]
    width = 0.08
    ax_hist.bar(bin_centers, rel_uncal["counts"], width=width, color="#5bc0de", alpha=0.7, edgecolor="black")
    ax_hist.set_xlabel("Predicted Probability / Confidence", fontsize=12)
    ax_hist.set_ylabel("Count (log)", fontsize=11)
    ax_hist.set_yscale("log")
    ax_hist.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_risk_coverage_curves(
    risk_cov_data: Dict[str, np.ndarray],
    title: str = "Selective Classification: Risk & Recall vs Coverage",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot Risk-Coverage and Recall-Coverage curves (Geifman & El-Yaniv, NeurIPS 2017).
    """
    fig, ax1 = plt.subplots(figsize=(7, 5))

    cov = risk_cov_data["coverage"]
    risk = risk_cov_data["risk"]
    recall = risk_cov_data["recall"]

    color_risk = "#d9534f"
    ax1.set_xlabel("Coverage (Proportion of Accepted Transactions)", fontsize=12)
    ax1.set_ylabel("Selective Risk (1 - Accuracy)", color=color_risk, fontsize=12)
    line1 = ax1.plot(cov, risk, color=color_risk, linewidth=2.2, label="Selective Risk")
    ax1.tick_params(axis="y", labelcolor=color_risk)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Secondary y-axis for Fraud Recall
    ax2 = ax1.twinx()
    color_recall = "#0275d8"
    ax2.set_ylabel("Fraud Recall on Retained Data", color=color_recall, fontsize=12)
    line2 = ax2.plot(cov, recall, color=color_recall, linewidth=2.0, linestyle="--", label="Fraud Recall")
    ax2.tick_params(axis="y", labelcolor=color_recall)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", frameon=True)

    plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_uncertainty_distributions(
    std_correct: np.ndarray,
    std_incorrect: np.ndarray,
    title: str = "Epistemic Uncertainty Distribution: Correct vs Misclassified",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distribution of epistemic uncertainty std for correct vs erroneous predictions.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    bins = np.linspace(0, max(np.max(std_correct), np.max(std_incorrect)), 40)
    ax.hist(std_correct, bins=bins, alpha=0.6, color="#5cb85c", label="Correct Predictions", density=True)
    ax.hist(std_incorrect, bins=bins, alpha=0.6, color="#d9534f", label="Errors (Misclassified)", density=True)

    ax.set_xlabel("Epistemic Uncertainty ($\sigma_{MC}$)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_cost_coverage_curve(
    cost_data: Dict[str, np.ndarray],
    baseline_cost: float,
    title: str = "Financial Cost Optimization via Selective Abstention",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot total financial loss in USD vs model coverage."""
    fig, ax = plt.subplots(figsize=(7, 5))

    cov = cost_data["coverage"]
    total_cost = cost_data["total_costs"]

    ax.plot(cov, total_cost, color="#2e6da4", linewidth=2.5, label="Trust-Aware Selective Policy")
    ax.axhline(baseline_cost, color="#d9534f", linestyle="--", linewidth=1.8, label=f"100% Automated Baseline (${baseline_cost:,.0f})")

    # Identify optimal minimum cost point
    min_idx = np.argmin(total_cost)
    opt_cov = cov[min_idx]
    opt_cost = total_cost[min_idx]
    ax.scatter([opt_cov], [opt_cost], color="gold", edgecolor="black", s=120, zorder=5, label=f"Optimal Min Cost: ${opt_cost:,.0f} @ {opt_cov*100:.1f}% Cov")

    ax.set_xlabel("Coverage (Proportion of Auto-Processed Transactions)", fontsize=12)
    ax.set_ylabel("Total Financial Loss ($)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig

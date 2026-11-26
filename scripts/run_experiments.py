"""
End-to-end experiment runner for Trust-Aware Credit Card Fraud Detection.
Executes training, calibration, MC Dropout uncertainty quantification, and selective triage.
"""

import os
import sys
import argparse
import json

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.data import load_fraud_data, prepare_datasets, create_dataloaders
from src.models.credit_net import CreditNet
from src.models.trainer import train_model, evaluate_predictions
from src.calibration.metrics import (
    binary_expected_calibration_error,
    class_conditional_ece,
    brier_score_loss,
    reliability_diagram_data
)
from src.calibration.temperature import fit_temperature_scaling
from src.uncertainty.mc_dropout import (
    mc_dropout_predict,
    compute_uncertainty_metrics,
    identify_failure_modes
)
from src.selective.policy import DualGatedTrustPolicy
from src.selective.risk_coverage import (
    FinancialCostMatrix,
    compute_risk_coverage_curve,
    compute_cost_coverage_curve
)
from src.visualization.plots import (
    plot_reliability_diagram,
    plot_risk_coverage_curves,
    plot_uncertainty_distributions,
    plot_cost_coverage_curve
)


def run_experiment(args):
    print("=" * 70)
    print("🚀 TRUST-AWARE ML SYSTEM FOR CREDIT CARD FRAUD DETECTION")
    print("=" * 70)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data
    print(f"\n[1/6] Loading dataset from: {args.data}")
    df = load_fraud_data(args.data)
    if args.subset and len(df) > args.subset:
        print(f"      Subsetting dataset to first {args.subset} samples (chronological order)")
        df = df.iloc[:args.subset].copy()

    total_samples = len(df)
    fraud_count = int(df["Class"].sum())
    fraud_rate = (fraud_count / total_samples) * 100.0
    print(f"      Total records: {total_samples:,} | Fraud cases: {fraud_count:,} ({fraud_rate:.3f}%)")

    # 2. Split & Standardize
    print(f"\n[2/6] Preparing splits using strategy='{args.split}'...")
    data = prepare_datasets(
        df,
        split_strategy=args.split,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.seed
    )
    print(f"      Train: {len(data['y_train']):,} | Val: {len(data['y_val']):,} | Test: {len(data['y_test']):,}")

    loaders = create_dataloaders(data, batch_size=args.batch_size)

    # Compute positive class weight to counteract imbalance
    train_neg = np.sum(data["y_train"] == 0)
    train_pos = max(1, np.sum(data["y_train"] == 1))
    pos_weight = float(train_neg / train_pos)
    print(f"      Computed positive class weight: {pos_weight:.2f}")

    # 3. Train Model
    print(f"\n[3/6] Training CreditNet (Dropout=0.3, Adam lr={args.lr})...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CreditNet(input_dim=data["X_train"].shape[1], dropout_rate=0.3)

    train_res = train_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        pos_weight=pos_weight,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        verbose=True
    )

    # Extract Validation and Test Logits deterministically
    model.eval()
    X_val_t = torch.tensor(data["X_val"], dtype=torch.float32, device=device)
    X_test_t = torch.tensor(data["X_test"], dtype=torch.float32, device=device)

    with torch.no_grad():
        val_logits = model(X_val_t).cpu().numpy().ravel()
        test_logits = model(X_test_t).cpu().numpy().ravel()

    test_probs_det = torch.sigmoid(torch.tensor(test_logits)).numpy()
    y_test = data["y_test"]

    # 4. Post-Hoc Calibration via Validation Set
    print("\n[4/6] Optimizing Temperature Scaling on Validation Split (Guo et al. 2017)...")
    temp_scaler, optimal_T = fit_temperature_scaling(
        val_logits=val_logits,
        val_labels=data["y_val"]
    )
    print(f"      Learned Optimal Temperature T: {optimal_T:.4f}")

    with torch.no_grad():
        test_logits_scaled = temp_scaler(torch.tensor(test_logits)).numpy().ravel()
        test_probs_calibrated = torch.sigmoid(torch.tensor(test_logits_scaled)).numpy()

    # Compute Calibration Metrics
    ece_det_uncal = binary_expected_calibration_error(y_test, test_probs_det, n_bins=10)
    ece_det_cal = binary_expected_calibration_error(y_test, test_probs_calibrated, n_bins=10)
    cc_ece_uncal = class_conditional_ece(y_test, test_probs_det, n_bins=10)
    cc_ece_cal = class_conditional_ece(y_test, test_probs_calibrated, n_bins=10)

    brier_uncal = brier_score_loss(y_test, test_probs_det)
    brier_cal = brier_score_loss(y_test, test_probs_calibrated)

    eval_det = evaluate_predictions(y_test, test_probs_det)
    eval_cal = evaluate_predictions(y_test, test_probs_calibrated)

    print("\n" + "-" * 60)
    print(" 📊 CALIBRATION BENCHMARK RESULTS")
    print("-" * 60)
    print(f" Metric                     | Before Calibration | After Temp Scaling")
    print(f" ROC-AUC                    | {eval_det['roc_auc']:.4f}             | {eval_cal['roc_auc']:.4f}")
    print(f" PR-AUC (Avg Precision)     | {eval_det['pr_auc']:.4f}             | {eval_cal['pr_auc']:.4f}")
    print(f" Binary ECE (Overall)       | {ece_det_uncal:.4f}             | {ece_det_cal:.4f}")
    print(f" ECE (Fraud Class)          | {cc_ece_uncal['ece_fraud']:.4f}             | {cc_ece_cal['ece_fraud']:.4f}")
    print(f" ECE (Non-Fraud Class)      | {cc_ece_uncal['ece_non_fraud']:.4f}             | {cc_ece_cal['ece_non_fraud']:.4f}")
    print(f" Brier Score (Loss)         | {brier_uncal:.4f}             | {brier_cal:.4f}")
    print("-" * 60)

    # 5. Bayesian Uncertainty via MC Dropout
    print(f"\n[5/6] Performing Monte Carlo Dropout ({args.mc_samples} stochastic forward passes)...")
    mc_results = mc_dropout_predict(
        model=model,
        X_tensor=X_test_t,
        n_samples=args.mc_samples,
        device=device
    )

    unc_metrics = compute_uncertainty_metrics(mc_results, y_test)
    failure_profile = identify_failure_modes(
        y_true=y_test,
        probs=mc_results["mean_probs"],
        uncertainty=mc_results["std_epistemic"],
        conf_threshold=0.90,
        uncertainty_threshold=0.10
    )

    print("\n" + "-" * 60)
    print(" 🔬 UNCERTAINTY & FAILURE MODE ANALYSIS")
    print("-" * 60)
    print(f" Mean Epistemic Std (Overall):       {unc_metrics['mean_std_overall']:.4f}")
    print(f" Mean Epistemic Std (Correct):       {unc_metrics['mean_std_correct']:.4f}")
    print(f" Mean Epistemic Std (Misclassified): {unc_metrics['mean_std_incorrect']:.4f}")
    print(f" Total Misclassified Samples:        {failure_profile['total_errors']}")
    print(f" High-Confidence (>90%) Wrong:       {failure_profile['high_conf_wrong_count']}")
    print(f" Recoverable Errors (High Unc):      {failure_profile['recoverable_error_count']}")
    print(f" Silent Killers (High Conf, Low Unc):{failure_profile['silent_killer_count']}")
    print("-" * 60)

    # 6. Cost-Sensitive Selective Classification
    print("\n[6/6] Computing Selective Classification & Financial Loss Utility...")
    cost_matrix = FinancialCostMatrix(cost_fn=250.0, cost_fp=15.0, cost_audit=5.0)

    # Full automation baseline cost
    det_preds = (test_probs_calibrated >= 0.5).astype(int)
    zero_abstain = np.zeros(len(y_test), dtype=bool)
    baseline_fin = cost_matrix.compute_total_cost(y_test, det_preds, zero_abstain)

    risk_cov = compute_risk_coverage_curve(
        y_true=y_test,
        probs=test_probs_calibrated,
        uncertainty=mc_results["std_epistemic"]
    )

    cost_cov = compute_cost_coverage_curve(
        y_true=y_test,
        probs=test_probs_calibrated,
        uncertainty=mc_results["std_epistemic"],
        cost_matrix=cost_matrix
    )

    min_cost_idx = np.argmin(cost_cov["total_costs"])
    opt_cov = cost_cov["coverage"][min_cost_idx]
    opt_cost = cost_cov["total_costs"][min_cost_idx]
    cost_savings = baseline_fin["total_cost"] - opt_cost

    print("\n" + "-" * 60)
    print(" 💼 FINANCIAL HUMAN-IN-THE-LOOP AUDIT IMPACT")
    print("-" * 60)
    print(f" 100% Automated Baseline Total Loss: ${baseline_fin['total_cost']:,.2f}")
    print(f" Optimal Trust-Aware Operating Loss: ${opt_cost:,.2f} (at {opt_cov*100:.1f}% Coverage)")
    print(f" Net Financial Savings:              ${cost_savings:,.2f} ({(cost_savings/baseline_fin['total_cost'])*100:.1f}% reduction)")
    print("-" * 60)

    # 7. Generate & Save Figures
    print("\n[Saving] Exporting publication figures to output directory...")
    rel_uncal = reliability_diagram_data(y_test, test_probs_det, n_bins=10)
    rel_cal = reliability_diagram_data(y_test, test_probs_calibrated, n_bins=10)

    # 1. Reliability diagram
    fig_rel = plot_reliability_diagram(
        rel_uncal=rel_uncal,
        rel_cal=rel_cal,
        ece_uncal=ece_det_uncal,
        ece_cal=ece_det_cal,
        save_path=os.path.join(args.output_dir, "fig_reliability_diagram.png")
    )

    # 2. Risk & Recall vs Coverage
    fig_rc = plot_risk_coverage_curves(
        risk_cov_data=risk_cov,
        save_path=os.path.join(args.output_dir, "fig_risk_coverage.png")
    )

    # 3. Uncertainty distributions
    preds_mc_class = (mc_results["mean_probs"] >= 0.5).astype(int)
    correct_m = (preds_mc_class == y_test)
    fig_unc = plot_uncertainty_distributions(
        std_correct=mc_results["std_epistemic"][correct_m],
        std_incorrect=mc_results["std_epistemic"][~correct_m],
        save_path=os.path.join(args.output_dir, "fig_uncertainty_distributions.png")
    )

    # 4. Cost vs Coverage
    fig_cost = plot_cost_coverage_curve(
        cost_data=cost_cov,
        baseline_cost=baseline_fin["total_cost"],
        save_path=os.path.join(args.output_dir, "fig_cost_optimization.png")
    )

    # Save metrics JSON
    summary_results = {
        "dataset": {
            "total_samples": total_samples,
            "fraud_count": fraud_count,
            "test_samples": len(y_test)
        },
        "optimal_temperature": optimal_T,
        "calibration": {
            "ece_uncalibrated": ece_det_uncal,
            "ece_calibrated": ece_det_cal,
            "brier_uncalibrated": brier_uncal,
            "brier_calibrated": brier_cal,
            "roc_auc": eval_cal["roc_auc"],
            "pr_auc": eval_cal["pr_auc"]
        },
        "uncertainty": {
            "mean_epistemic_std": unc_metrics["mean_std_overall"],
            "std_correct": unc_metrics["mean_std_correct"],
            "std_incorrect": unc_metrics["mean_std_incorrect"],
            "high_conf_wrong": failure_profile["high_conf_wrong_count"],
            "silent_killers": failure_profile["silent_killer_count"]
        },
        "financial_optimization": {
            "baseline_loss_usd": baseline_fin["total_cost"],
            "optimal_loss_usd": opt_cost,
            "dollar_savings_usd": cost_savings,
            "optimal_coverage": opt_cov
        }
    }

    results_json_path = os.path.join(args.output_dir, "metrics_summary.json")
    with open(results_json_path, "w") as f:
        json.dump(summary_results, f, indent=4)

    print(f"\n✅ All results and figures saved successfully to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trust-Aware ML Experiment Runner")
    parser.add_argument("--data", type=str, default="data/creditcard.csv", help="Path to credit card CSV")
    parser.add_argument("--subset", type=int, default=25838, help="Number of rows to use (default: 25838 matching prior baseline, or 0 for full)")
    parser.add_argument("--split", type=str, default="stratified", choices=["stratified", "temporal"], help="Split strategy")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--mc-samples", type=int, default=50, help="Number of Monte Carlo forward passes")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="experiments/results", help="Directory to save plots & metrics")

    args = parser.parse_args()
    if args.subset == 0:
        args.subset = None
    run_experiment(args)

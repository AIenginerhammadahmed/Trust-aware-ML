"""
Comprehensive Benchmark Runner for the Uncertainty-Gated Cascade on the 2026 Dataset.
Compares:
1. Standard Ungated Tabular Baseline
2. Calibrated Tabular (Confidence-Gated only)
3. Traditional Selective Classification (Abstention without resolution)
4. Proposed Uncertainty-Gated Cascade (System-1 + System-2 LLM Auditor)
Evaluates on the Q4 (Nov-Dec) Concept Drift test partition.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Ensure repo root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_2026 import (
    load_and_preprocess_2026_data,
    prepare_temporal_datasets,
    create_dataloaders
)
from src.models.credit_net import CreditNet
from src.models.trainer import train_model, evaluate_predictions
from src.calibration.temperature import fit_temperature_scaling
from src.calibration.metrics import binary_expected_calibration_error, brier_score_loss
from src.uncertainty.mc_dropout import mc_dropout_predict
from src.selective.cascade import UncertaintyGatedCascade
from src.system2.auditor import System2Auditor
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score


def run_benchmark():
    output_dir = "experiments/results_2026"
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("🚀 UNCERTAINTY-GATED FRAUD DETECTION CASCADE: FULL BENCHMARK")
    print(f"   Execution Device: {device}")
    print("=" * 80)

    # 1. Load Data & Prepare Temporal Split
    print("\n[1/5] Loading 2026 Dataset and Preparing Temporal Split...")
    df = load_and_preprocess_2026_data("data/credit_card_fraud_detection_2026.csv")
    data = prepare_temporal_datasets(df)

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    test_df = data["test_df"]

    print(f"      Train Set (Jan-Aug): {len(X_train):,} samples | Fraud: {np.mean(y_train)*100:.2f}%")
    print(f"      Val Set   (Sep-Oct): {len(X_val):,} samples | Fraud: {np.mean(y_val)*100:.2f}%")
    print(f"      Test Set  (Nov-Dec): {len(X_test):,} samples | Fraud: {np.mean(y_test)*100:.2f}% (Holiday Drift)")

    # 2. Train System-1 Tabular Model (CreditNet with Dropout)
    print("\n[2/5] Training System-1 Neural Model with MC Dropout...")
    loaders = create_dataloaders(data, batch_size=512)
    pos_weight = float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))

    input_dim = X_train.shape[1]
    model = CreditNet(input_dim=input_dim, hidden_dims=(128, 64, 32), dropout_rate=0.25)
    
    # Train for 10 epochs
    train_history = train_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        pos_weight=pos_weight * 0.5,  # smooth weight to balance precision/recall
        epochs=10,
        lr=2e-3,
        device=device,
        verbose=False
    )
    print("      System-1 Model training complete.")

    # 3. Fit Temperature Scaling on Validation Set
    print("\n[3/5] Calibrating Probabilities via Temperature Scaling (Guo et al. 2017)...")
    model.eval()
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)

    with torch.no_grad():
        val_logits = model(X_val_t).cpu().numpy().ravel()
        test_logits = model(X_test_t).cpu().numpy().ravel()

    raw_val_probs = torch.sigmoid(torch.tensor(val_logits)).numpy()
    raw_test_probs = torch.sigmoid(torch.tensor(test_logits)).numpy()
    raw_test_preds = (raw_test_probs >= 0.5).astype(int)
    
    ece_before = binary_expected_calibration_error(y_val, raw_val_probs)
    scaler, opt_temp = fit_temperature_scaling(val_logits, y_val, device=device)
    
    val_calibrated_probs = torch.sigmoid(torch.tensor(val_logits) / opt_temp).numpy()
    ece_after = binary_expected_calibration_error(y_val, val_calibrated_probs)
    print(f"      Optimal Temperature T: {opt_temp:.4f}")
    print(f"      Val ECE Before Scaling: {ece_before:.4f} -> After Scaling: {ece_after:.4f} ({(ece_before-ece_after)/ece_before*100:.1f}% reduction)")

    # 4. Tune Dual Gate Thresholds on Validation Set
    print("\n[4/5] Tuning Dual Gate Epistemic and Aleatoric Thresholds...")
    val_mc = mc_dropout_predict(model, torch.from_numpy(X_val).float(), n_samples=25, device=device)
    val_epistemic_std = val_mc["std_epistemic"]
    
    # Set tau_sigma at 88th percentile of in-distribution epistemic uncertainty
    tau_sigma = float(np.percentile(val_epistemic_std, 88))
    tau_conf = 0.85
    print(f"      Calibrated Epistemic Threshold (tau_sigma): {tau_sigma:.4f}")
    print(f"      Calibrated Aleatoric Threshold (tau_conf):  {tau_conf:.4f}")

    # 5. Run Full Benchmark on Test Set (Q4 Holiday Concept Drift)
    print("\n[5/5] Running End-to-End Evaluation on Test Set (16,707 samples)...")

    # --- Run Uncertainty-Gated Cascade ---
    auditor = System2Auditor(model_name="heuristic-auditor")
    cascade = UncertaintyGatedCascade(
        sys1_model=model,
        temperature=opt_temp,
        tau_conf=tau_conf,
        tau_sigma=tau_sigma,
        sys2_auditor=auditor
    )

    cascade_results = cascade.process_stream(
        X=X_test,
        df_records=test_df,
        mc_samples=25,
        device=device
    )

    # Compute Metrics for Each Method:
    # -------------------------------------------------------------
    # Method 1: Ungated Baseline (Standard Forced-Choice Model)
    # -------------------------------------------------------------
    auc_1 = roc_auc_score(y_test, raw_test_probs)
    pr_auc_1 = average_precision_score(y_test, raw_test_probs)
    f1_1 = f1_score(y_test, raw_test_preds)
    ece_1 = binary_expected_calibration_error(y_test, raw_test_probs)
    # Silent errors: False negatives with confidence >= 0.80
    conf_1 = np.maximum(raw_test_probs, 1.0 - raw_test_probs)
    silent_errors_1 = int(np.sum((y_test == 1) & (raw_test_preds == 0) & (conf_1 >= 0.80)))

    # -------------------------------------------------------------
    # Method 2: Calibrated Tabular (Confidence Gate Only)
    # -------------------------------------------------------------
    cal_test_probs = torch.sigmoid(torch.tensor(test_logits) / opt_temp).numpy()
    cal_test_preds = (cal_test_probs >= 0.5).astype(int)
    auc_2 = roc_auc_score(y_test, cal_test_probs)
    pr_auc_2 = average_precision_score(y_test, cal_test_probs)
    f1_2 = f1_score(y_test, cal_test_preds)
    ece_2 = binary_expected_calibration_error(y_test, cal_test_probs)
    conf_2 = np.maximum(cal_test_probs, 1.0 - cal_test_probs)
    silent_errors_2 = int(np.sum((y_test == 1) & (cal_test_preds == 0) & (conf_2 >= 0.80)))

    # -------------------------------------------------------------
    # Method 3: Traditional Selective Classification (Abstention)
    # Evaluated on the accepted partition (Coverage)
    # -------------------------------------------------------------
    auto_mask = cascade_results["automated_mask"]
    coverage_3 = cascade_results["coverage"]
    y_test_acc = y_test[auto_mask]
    probs_acc = cascade_results["sys1_probs"][auto_mask]
    preds_acc = (probs_acc >= 0.5).astype(int)
    
    auc_3 = roc_auc_score(y_test_acc, probs_acc)
    pr_auc_3 = average_precision_score(y_test_acc, probs_acc)
    f1_3 = f1_score(y_test_acc, preds_acc)
    ece_3 = binary_expected_calibration_error(y_test_acc, probs_acc)
    conf_3 = np.maximum(probs_acc, 1.0 - probs_acc)
    silent_errors_3 = int(np.sum((y_test_acc == 1) & (preds_acc == 0) & (conf_3 >= 0.80)))

    # -------------------------------------------------------------
    # Method 4: Proposed Uncertainty-Gated Cascade (Full System)
    # Resolves all samples across both Tier-1 and Tier-2
    # -------------------------------------------------------------
    final_probs = cascade_results["final_probs"]
    final_preds = cascade_results["final_preds"]
    
    auc_4 = roc_auc_score(y_test, final_probs)
    pr_auc_4 = average_precision_score(y_test, final_probs)
    f1_4 = f1_score(y_test, final_preds)
    ece_4 = binary_expected_calibration_error(y_test, final_probs)
    conf_4 = np.maximum(final_probs, 1.0 - final_probs)
    silent_errors_4 = int(np.sum((y_test == 1) & (final_preds == 0) & (conf_4 >= 0.80)))

    # Print Summary Results Table
    print("\n" + "=" * 95)
    print("🎯 BENCHMARK RESULTS: COMPARISON ACROSS PARADIGMS (TEST SET: NOV-DEC)")
    print("=" * 95)
    header = f"{'Method':<32} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'F1':<6} | {'ECE':<7} | {'Silent FNs':<10} | {'Coverage':<8}"
    print(header)
    print("-" * 95)
    print(f"{'1. Standard Tabular (Forced)':<32} | {auc_1:.4f}   | {pr_auc_1:.4f}   | {f1_1:.4f} | {ece_1:.4f}  | {silent_errors_1:<10} | 100.0%")
    print(f"{'2. Calibrated Tabular (Temp)':<32} | {auc_2:.4f}   | {pr_auc_2:.4f}   | {f1_2:.4f} | {ece_2:.4f}  | {silent_errors_2:<10} | 100.0%")
    print(f"{'3. Selective Abstention (Dead-End)':<32} | {auc_3:.4f}   | {pr_auc_3:.4f}   | {f1_3:.4f} | {ece_3:.4f}  | {silent_errors_3:<10} | {coverage_3*100:.1f}%")
    print(f"{'4. Proposed Cascade (Sys1+Sys2)':<32} | {auc_4:.4f}   | {pr_auc_4:.4f}   | {f1_4:.4f} | {ece_4:.4f}  | {silent_errors_4:<10} | 100.0%")
    print("=" * 95)

    print("\n⚡ SYSTEM LATENCY & THROUGHPUT METRICS:")
    print(f"   - System-1 Processing Time (16,707 samples): {cascade_results['sys1_time']:.2f}s ({cascade_results['sys1_time']/len(X_test)*1000:.3f} ms/tx)")
    print(f"   - Automated by System-1 Tier: {cascade_results['n_automated']:,} ({cascade_results['coverage']*100:.1f}%)")
    print(f"   - Escalated to System-2 LLM Tier: {cascade_results['n_escalated']:,} ({(1-cascade_results['coverage'])*100:.1f}%)")
    print(f"   - Overall Stream Throughput: {cascade_results['throughput_tps']:.1f} transactions/second")

    # Save Results JSON
    results_dict = {
        "dataset": "credit_card_fraud_detection_2026",
        "split": "temporal_q4_drift",
        "n_test": len(X_test),
        "temperature": float(opt_temp),
        "tau_sigma": float(tau_sigma),
        "tau_conf": float(tau_conf),
        "metrics": {
            "standard_tabular": {
                "roc_auc": float(auc_1), "pr_auc": float(pr_auc_1), "f1": float(f1_1),
                "ece": float(ece_1), "silent_false_negatives": silent_errors_1, "coverage": 1.0
            },
            "calibrated_tabular": {
                "roc_auc": float(auc_2), "pr_auc": float(pr_auc_2), "f1": float(f1_2),
                "ece": float(ece_2), "silent_false_negatives": silent_errors_2, "coverage": 1.0
            },
            "selective_abstention": {
                "roc_auc": float(auc_3), "pr_auc": float(pr_auc_3), "f1": float(f1_3),
                "ece": float(ece_3), "silent_false_negatives": silent_errors_3, "coverage": float(coverage_3)
            },
            "proposed_cascade": {
                "roc_auc": float(auc_4), "pr_auc": float(pr_auc_4), "f1": float(f1_4),
                "ece": float(ece_4), "silent_false_negatives": silent_errors_4, "coverage": 1.0,
                "n_automated": cascade_results["n_automated"],
                "n_escalated": cascade_results["n_escalated"],
                "throughput_tps": float(cascade_results["throughput_tps"])
            }
        }
    }
    with open(os.path.join(output_dir, "benchmark_summary.json"), "w") as f:
        json.dump(results_dict, f, indent=4)

    # 6. Generate Publication-Quality Comparison Figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Subplot 1: PR-AUC and ROC-AUC
    methods = ["Standard", "Calibrated", "Selective\n(Accepted)", "Proposed\nCascade"]
    pr_aucs = [pr_auc_1, pr_auc_2, pr_auc_3, pr_auc_4]
    roc_aucs = [auc_1, auc_2, auc_3, auc_4]
    
    x = np.arange(len(methods))
    width = 0.35
    axes[0].bar(x - width/2, roc_aucs, width, label="ROC-AUC", color="#3b82f6")
    axes[0].bar(x + width/2, pr_aucs, width, label="PR-AUC", color="#10b981")
    axes[0].set_ylabel("Score", fontsize=11, fontweight="bold")
    axes[0].set_title("Discrimination Performance (Q4 Test)", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].set_ylim([0.0, 1.05])
    axes[0].legend(loc="lower right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot 2: Expected Calibration Error (ECE)
    eces = [ece_1, ece_2, ece_3, ece_4]
    colors_ece = ["#ef4444", "#f59e0b", "#8b5cf6", "#059669"]
    bars = axes[1].bar(methods, eces, color=colors_ece, width=0.5)
    axes[1].set_ylabel("Expected Calibration Error (ECE)", fontsize=11, fontweight="bold")
    axes[1].set_title("Calibration Reliability (Lower is Better)", fontsize=12, fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.002, f"{yval:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Subplot 3: Silent False Negatives (Overconfident Misses)
    silent_counts = [silent_errors_1, silent_errors_2, silent_errors_3, silent_errors_4]
    colors_silent = ["#dc2626", "#ea580c", "#7c3aed", "#16a34a"]
    bars2 = axes[2].bar(methods, silent_counts, color=colors_silent, width=0.5)
    axes[2].set_ylabel("Count of Silent High-Conf False Negatives", fontsize=11, fontweight="bold")
    axes[2].set_title("Safety: Overconfident Misses (Lower is Better)", fontsize=12, fontweight="bold")
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        yval = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{int(yval)}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "cascade_performance_comparison.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"\n📊 Benchmark comparison chart saved to: {chart_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()

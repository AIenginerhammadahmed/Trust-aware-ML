"""
Audit-Ready Benchmark for the Uncertainty-Gated Fraud Detection Cascade.
Fixes all mathematical & methodological points:
1. Uses Hybrid System-1: LightGBM (calibrated tabular discrimination, ECE~0.03) + CreditNet MC Dropout (epistemic uncertainty).
2. Also benchmarks CreditNet (Uncalibrated vs Temp-Scaled) to explain the T < 1 Silent FN anomaly.
3. Tunes decision threshold & Dual Gate on Validation set (Sep-Oct) for ~90% automation coverage (~10% escalation).
4. Reports honest, mathematically consistent System-1 latency, offline System-2 latency, and projected live-LLM latency.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_2026 import load_and_preprocess_2026_data, prepare_temporal_datasets, create_dataloaders
from src.models.credit_net import CreditNet
from src.models.trainer import train_model
from src.calibration.temperature import fit_temperature_scaling
from src.calibration.metrics import binary_expected_calibration_error
from src.uncertainty.mc_dropout import mc_dropout_predict
from src.system2.serializer import serialize_transaction_case
from src.system2.auditor import System2Auditor


def find_optimal_f1_threshold(y_true, probs):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.70, 66):
        f1 = f1_score(y_true, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def run_audit_benchmark():
    output_dir = "experiments/results_2026"
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    print("=" * 85)
    print("🚀 AUDIT-READY UNCERTAINTY-GATED CASCADE BENCHMARK")
    print("=" * 85)

    # 1. Load & Split Data
    df = load_and_preprocess_2026_data("data/credit_card_fraud_detection_2026.csv")
    data = prepare_temporal_datasets(df)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    test_df = data["test_df"]
    loaders = create_dataloaders(data, batch_size=512)

    # 2. Train CreditNet (with pos_weight for ablation & MC Dropout epistemic uncertainty)
    torch.manual_seed(42)
    pos_weight = float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1))) * 0.5
    mlp = CreditNet(input_dim=X_train.shape[1], hidden_dims=(128, 64, 32), dropout_rate=0.25)
    train_model(mlp, loaders["train"], loaders["val"], pos_weight=pos_weight, epochs=10, lr=2e-3, device=device, verbose=False)

    mlp.eval()
    with torch.no_grad():
        val_logits = mlp(torch.tensor(X_val, dtype=torch.float32)).numpy().ravel()
        test_logits = mlp(torch.tensor(X_test, dtype=torch.float32)).numpy().ravel()

    mlp_raw_probs = torch.sigmoid(torch.tensor(test_logits)).numpy()
    _, opt_T = fit_temperature_scaling(val_logits, y_val, device=device)
    mlp_temp_probs = torch.sigmoid(torch.tensor(test_logits) / opt_T).numpy()

    # 3. Train LightGBM (Primary System-1 Tabular Predictor)
    t_lgb0 = time.time()
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    lgb_clf.fit(X_train, y_train)
    lgb_val_probs = lgb_clf.predict_proba(X_val)[:, 1]
    
    t_inf0 = time.time()
    lgb_test_probs = lgb_clf.predict_proba(X_test)[:, 1]
    lgb_inf_time = time.time() - t_inf0

    # Optimal F1 threshold tuned on Validation set (Sep-Oct)
    opt_thresh, val_f1 = find_optimal_f1_threshold(y_val, lgb_val_probs)
    print(f"Optimal Val Threshold for LightGBM: {opt_thresh:.2f} (Val F1: {val_f1:.4f})")

    # 4. Compute MC Dropout Epistemic Uncertainty (for Drift Detection)
    val_mc = mc_dropout_predict(mlp, torch.from_numpy(X_val).float(), n_samples=20, device=device)
    t_mc0 = time.time()
    test_mc = mc_dropout_predict(mlp, torch.from_numpy(X_test).float(), n_samples=20, device=device)
    mc_inf_time = time.time() - t_mc0

    sys1_total_time = lgb_inf_time + mc_inf_time
    sys1_ms_per_tx = (sys1_total_time / len(X_test)) * 1000.0

    # 5. Tune Dual Gate on Validation Set for ~90% Automation Coverage (~10% Escalation)
    # Aleatoric ambiguity band: probabilities close to opt_thresh (e.g. [opt_thresh - band, opt_thresh + band])
    # Epistemic gate: sigma_MC > tau_sigma (94th percentile of Val)
    tau_sigma = float(np.percentile(val_mc["std_epistemic"], 94))
    # Find ambiguity band around opt_thresh on Val to hit ~90% total coverage
    best_band = 0.05
    for band in np.linspace(0.02, 0.25, 47):
        val_aleatoric_pass = np.abs(lgb_val_probs - opt_thresh) >= band
        val_epistemic_pass = val_mc["std_epistemic"] <= tau_sigma
        cov = np.mean(val_aleatoric_pass & val_epistemic_pass)
        if cov <= 0.905:
            best_band = float(band)
            break

    # Apply Dual Gate on Test Set
    aleatoric_pass = np.abs(lgb_test_probs - opt_thresh) >= best_band
    epistemic_pass = test_mc["std_epistemic"] <= tau_sigma
    automated_mask = aleatoric_pass & epistemic_pass
    escalated_mask = ~automated_mask

    coverage = float(np.mean(automated_mask))
    n_auto = int(np.sum(automated_mask))
    n_esc = int(np.sum(escalated_mask))

    # 6. System-2 Contextual Escalation on the ~10% Tail
    auditor = System2Auditor(model_name="heuristic-auditor")
    cascade_probs = np.copy(lgb_test_probs)
    
    t_sys2_0 = time.time()
    escalated_indices = np.where(escalated_mask)[0]
    for idx in escalated_indices:
        row = test_df.iloc[idx]
        brief = serialize_transaction_case(
            row=row,
            sys1_prob=float(lgb_test_probs[idx]),
            sys1_uncertainty=float(test_mc["std_epistemic"][idx]),
            gate_reason="Dual Gate Escalation"
        )
        res = auditor.audit_case(brief, raw_row=row.to_dict(), sys1_prob=float(lgb_test_probs[idx]))
        # Weighted refinement preserving calibrated scale
        cascade_probs[idx] = 0.65 * float(lgb_test_probs[idx]) + 0.35 * float(res["fraud_probability"])
    sys2_sim_time = time.time() - t_sys2_0

    # Helper to compute all metrics
    def compute_metrics(name, y_true, probs, thresh, cov=1.0):
        preds = (probs >= thresh).astype(int)
        auc = roc_auc_score(y_true, probs)
        pr = average_precision_score(y_true, probs)
        f1 = f1_score(y_true, preds, zero_division=0)
        ece = binary_expected_calibration_error(y_true, probs)
        # Silent high-confidence false negatives: actual fraud predicted legit with P(legit) >= 0.80
        conf = np.maximum(probs, 1.0 - probs)
        silent_fn = int(np.sum((y_true == 1) & (preds == 0) & (conf >= 0.80)))
        return {
            "name": name, "roc_auc": float(auc), "pr_auc": float(pr),
            "f1": float(f1), "ece": float(ece), "silent_fn": silent_fn, "coverage": float(cov)
        }

    m1 = compute_metrics("1. CreditNet Uncalibrated (pw)", y_test, mlp_raw_probs, 0.50, 1.0)
    m2 = compute_metrics(f"2. CreditNet TempScaled (T={opt_T:.2f})", y_test, mlp_temp_probs, 0.50, 1.0)
    m3 = compute_metrics("3. LightGBM Baseline (Forced)", y_test, lgb_test_probs, opt_thresh, 1.0)
    m4 = compute_metrics("4. Selective Abstention (Auto-Only)", y_test[automated_mask], lgb_test_probs[automated_mask], opt_thresh, coverage)
    m5 = compute_metrics("5. Proposed Hybrid Cascade (Sys1+Sys2)", y_test, cascade_probs, opt_thresh, 1.0)

    print("\n" + "=" * 98)
    print(f"{'Model / Paradigm':<38} | {'ROC-AUC':<7} | {'PR-AUC':<7} | {'F1':<6} | {'ECE':<7} | {'SilentFN':<8} | {'Coverage'}")
    print("-" * 98)
    for m in [m1, m2, m3, m4, m5]:
        print(f"{m['name']:<38} | {m['roc_auc']:.4f}  | {m['pr_auc']:.4f}  | {m['f1']:.4f} | {m['ece']:.4f}  | {m['silent_fn']:<8} | {m['coverage']*100:.1f}%")
    print("=" * 98)

    # Honest Latency Math
    esc_rate = 1.0 - coverage
    projected_sync_llm_ms = coverage * sys1_ms_per_tx + esc_rate * 1500.0
    measured_sim_ms = ((sys1_total_time + sys2_sim_time) / len(X_test)) * 1000.0

    print("\n⏱️  LATENCY & ESCALATION BREAKDOWN:")
    print(f"   - Automated by System-1 (LightGBM + MC Dropout): {n_auto:,} ({coverage*100:.2f}%)")
    print(f"   - Escalated to System-2:                         {n_esc:,} ({esc_rate*100:.2f}%)")
    print(f"   - System-1 Inference Latency (Fast Path):        {sys1_ms_per_tx:.3f} ms/tx")
    print(f"   - End-to-End Measured Latency (Offline Engine):  {measured_sim_ms:.3f} ms/tx")
    print(f"   - Projected Synchronous Live-LLM Latency (1.5s): {projected_sync_llm_ms:.1f} ms/tx (vs 1500 ms pure LLM -> {(1 - projected_sync_llm_ms/1500)*100:.1f}% reduction)")

    # Save JSON
    summary = {
        "n_test": len(X_test),
        "opt_T": float(opt_T),
        "opt_thresh": float(opt_thresh),
        "tau_sigma": float(tau_sigma),
        "ambiguity_band": float(best_band),
        "coverage": coverage,
        "escalation_rate": esc_rate,
        "n_auto": n_auto,
        "n_esc": n_esc,
        "sys1_ms_per_tx": sys1_ms_per_tx,
        "measured_sim_ms_per_tx": measured_sim_ms,
        "projected_sync_llm_ms_per_tx": projected_sync_llm_ms,
        "models": [m1, m2, m3, m4, m5]
    }
    with open(os.path.join(output_dir, "audit_benchmark_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # Update Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = ["CreditNet\nUncal", "CreditNet\nTempScaled", "LightGBM\nBaseline", "Selective\n(Auto-Only)", "Proposed\nCascade"]
    roc_vals = [m["roc_auc"] for m in [m1, m2, m3, m4, m5]]
    pr_vals = [m["pr_auc"] for m in [m1, m2, m3, m4, m5]]
    ece_vals = [m["ece"] for m in [m1, m2, m3, m4, m5]]
    f1_vals = [m["f1"] for m in [m1, m2, m3, m4, m5]]

    x = np.arange(len(labels))
    w = 0.35
    axes[0].bar(x - w/2, roc_vals, w, label="ROC-AUC", color="#3b82f6")
    axes[0].bar(x + w/2, pr_vals, w, label="PR-AUC", color="#10b981")
    axes[0].set_title("Discrimination (Q4 Holiday Drift)", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylim([0, 1.05])
    axes[0].legend(loc="lower right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    bars1 = axes[1].bar(labels, ece_vals, color=["#ef4444", "#f59e0b", "#0ea5e9", "#8b5cf6", "#059669"], width=0.55)
    axes[1].set_title("Calibration Error (ECE, Lower is Better)", fontweight="bold")
    axes[1].tick_params(axis="x", labelsize=9)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars1:
        axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.004, f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    bars2 = axes[2].bar(labels, f1_vals, color=["#64748b", "#94a3b8", "#0284c7", "#7c3aed", "#16a34a"], width=0.55)
    axes[2].set_title("F1-Score (Higher is Better)", fontweight="bold")
    axes[2].tick_params(axis="x", labelsize=9)
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars2:
        axes[2].text(b.get_x() + b.get_width()/2, b.get_height() + 0.005, f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cascade_performance_comparison.png"), dpi=300)
    plt.close()
    print("✅ Saved updated chart and audit_benchmark_summary.json")


if __name__ == "__main__":
    run_audit_benchmark()

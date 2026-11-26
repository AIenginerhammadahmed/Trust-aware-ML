"""
Run Live Gemini 3.5 Flash Lite Auditor Evaluation on Escalated Transactions.
Evaluates precision, recall, and qualitative reasoning of the live LLM
against the heuristic proxy.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from src.data_2026 import load_and_preprocess_2026_data, prepare_temporal_datasets
from src.system2.serializer import serialize_transaction_case
from src.system2.auditor import System2Auditor


def main():
    print("=" * 60)
    print("LIVE GEMINI 3.5 FLASH LITE AUDITOR BENCHMARK")
    print("=" * 60)

    # 1. Load data
    df = load_and_preprocess_2026_data()
    d = prepare_temporal_datasets(df)
    Xtr, ytr = d["X_train"], d["y_train"]
    Xte, yte = d["X_test"], d["y_test"]
    test_df = d["test_df"].copy().reset_index(drop=True)

    # 2. Quick LightGBM inference
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1).fit(Xtr, ytr)
    pt = clf.predict_proba(Xte)[:, 1]

    # Load thresholds from saved audit config
    saved_path = "experiments/results_2026/audit_benchmark_summary.json"
    if os.path.exists(saved_path):
        saved = json.load(open(saved_path))
        th = saved.get("opt_thresh", 0.08)
        band = saved.get("ambiguity_band", 0.03)
        tau = saved.get("tau_sigma", 0.0873)
    else:
        th, band, tau = 0.08, 0.03, 0.0873

    # Aleatoric condition: ambiguous probability around threshold
    ambiguous = np.abs(pt - th) < band

    # For fast evaluation, select ambiguous test cases where LightGBM was uncertain
    # or let's use the exact escalated mask if MC dropout std is cached, else aleatoric band
    esc_indices = np.where(ambiguous)[0]
    print(f"Total transactions in ambiguous band [0.05, 0.11]: {len(esc_indices)}")

    # Separate into frauds and legitimate transactions
    frauds_in_esc = [i for i in esc_indices if yte[i] == 1]
    legits_in_esc = [i for i in esc_indices if yte[i] == 0]
    print(f"Ambiguous slice contains {len(frauds_in_esc)} true frauds and {len(legits_in_esc)} legitimate transactions.")

    # Sample 15 true frauds and 25 legitimate ambiguous transactions for a 40-case live audit
    np.random.seed(42)
    sample_f = list(np.random.choice(frauds_in_esc, size=min(15, len(frauds_in_esc)), replace=False))
    sample_l = list(np.random.choice(legits_in_esc, size=min(25, len(legits_in_esc)), replace=False))
    eval_indices = sample_f + sample_l
    np.random.shuffle(eval_indices)

    print(f"\nRunning Live Gemini 3.5 Flash Lite Audit on {len(eval_indices)} test cases...")
    print("Evaluating semantic reasoning, precision, and false-alarm suppression...\n")

    auditor = System2Auditor(model_name="gemini-3.5-flash-lite")
    results = []
    latencies = []

    for step, idx in enumerate(eval_indices):
        row = test_df.iloc[idx]
        sys1_prob = float(pt[idx])
        y_true = int(yte[idx])

        brief = serialize_transaction_case(
            row=row,
            sys1_prob=sys1_prob,
            sys1_uncertainty=0.089,
            gate_reason=f"Ambiguous Aleatoric Margin (|p - {th}| < {band})"
        )

        t0 = time.time()
        audit = auditor.audit_case(
            case_text=brief,
            raw_row=row.to_dict(),
            sys1_prob=sys1_prob
        )
        dt = time.time() - t0
        latencies.append(dt)

        y_pred = 1 if audit["verdict"] == "FRAUD_CONFIRMED" else 0
        match = "✓" if y_true == y_pred else "✗"

        print(f"[{step+1:02d}/{len(eval_indices)}] {match} True: {y_true} | Pred: {y_pred} ({audit['verdict']}) | Prob: {audit['fraud_probability']:.2f} | Latency: {dt:.2f}s")
        print(f"      Rationale: {audit['rationale'][:110]}...")

        results.append({
            "idx": int(idx),
            "txn_id": str(row.get("transaction_id", f"TXN_{idx}")),
            "amount": float(row.get("transaction_amount", 0.0)),
            "category": str(row.get("merchant_category", "Unknown")),
            "sys1_prob": sys1_prob,
            "y_true": y_true,
            "y_pred": y_pred,
            "llm_prob": audit["fraud_probability"],
            "risk_tier": audit["risk_tier"],
            "verdict": audit["verdict"],
            "rationale": audit["rationale"],
            "latency_s": dt
        })

        time.sleep(0.4)

    res_df = pd.DataFrame(results)
    y_true_all = res_df["y_true"].values
    y_pred_all = res_df["y_pred"].values

    acc = accuracy_score(y_true_all, y_pred_all)
    prec = precision_score(y_true_all, y_pred_all, zero_division=0)
    rec = recall_score(y_true_all, y_pred_all, zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, zero_division=0)
    avg_latency = np.mean(latencies)

    print("\n" + "=" * 60)
    print("LIVE GEMINI 3.5 FLASH LITE AUDIT EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Transactions Evaluated: {len(res_df)}")
    print(f"Accuracy:                     {acc:.4f} ({acc*100:.1f}%)")
    print(f"Precision:                    {prec:.4f} ({prec*100:.1f}%)")
    print(f"Recall:                       {rec:.4f} ({rec*100:.1f}%)")
    print(f"F1 Score:                     {f1:.4f}")
    print(f"Average Latency:              {avg_latency:.3f} s / audit")
    print("=" * 60)

    # Save to disk
    os.makedirs("experiments/results_2026", exist_ok=True)
    res_df.to_csv("experiments/results_2026/live_gemini_audit_sample_results.csv", index=False)
    
    summary = {
        "n_evaluated": len(res_df),
        "n_fraud": int(y_true_all.sum()),
        "n_legit": int((1 - y_true_all).sum()),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "avg_latency_s": float(avg_latency)
    }
    with open("experiments/results_2026/live_gemini_audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved results to experiments/results_2026/live_gemini_audit_sample_results.csv")


if __name__ == "__main__":
    main()

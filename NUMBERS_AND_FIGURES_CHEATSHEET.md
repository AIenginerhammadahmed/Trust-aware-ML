# 📊 Verified Numbers, Tables & Figures Cheat Sheet
*(Raw empirical facts, formulas, and file paths only — zero prose for safe academic writing)*

---

## 1. Dataset & Temporal Split (`data/credit_card_fraud_detection_2026.csv`)

- **Total Records**: `100,000` transactions (`2,055` frauds; `2.06%` overall prevalence)
- **Feature Dimensions**: `35` input features after preprocessing (`20` numerical, `9` categorical, `6` binary)
- **Missing Value Imputation**: `41,709` missing `browser` entries (physical POS/ATM channels) $\rightarrow$ `'None/In-Store'`

| Split | Date Range | Total Rows | Fraud Count | Fraud Rate | Purpose |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Train** | Jan 1 – Aug 31, 2026 | `66,388` | `1,341` | `2.02%` | Model fitting (`LightGBM`, `CreditNet`) |
| **Validation / Calibration** | Sep 1 – Oct 31, 2026 | `16,905` | `362` | `2.14%` | Thresholding ($\theta^*$), Temperature ($T$), $\tau_{\sigma}$ |
| **Test (Drift Evaluation)** | Nov 1 – Dec 31, 2026 | `16,707` | `352` | `2.11%` | Final held-out evaluation |

---

## 2. System-1 & Dual-Gate Hyperparameters (`experiments/results_2026/audit_benchmark_summary.json`)

- **`CreditNet` Architecture**: `35 -> 128 -> 64 -> 32 -> 1` (PyTorch MLP, Dropout $p = 0.25$, $10$ epochs, `lr = 2e-3`)
- **MC Dropout Passes**: $M = 20$ stochastic forward passes
- **Temperature Scaling ($T$)**: $T = 0.7705$ (`0.77045226`)
- **Operating Threshold ($\theta^*$)**: `0.08`
- **Aleatoric Ambiguity Band ($\Delta$)**: `±0.03` (Escalation interval: `[0.05, 0.11]`)
- **Epistemic Uncertainty Threshold ($\tau_{\sigma}$)**: `0.0873` (`0.08730762`, calibrated at the `94th` percentile of validation $\sigma_{\text{MC}}$)
- **Dual-Gate Routing Formula**:
  $$\text{Automate}(x) = (|p_{\text{cal}}(x) - 0.08| \ge 0.03) \;\land\; (\sigma_{\text{MC}}(x) \le 0.0873)$$

### Triage Split on Test Set (`N = 16,707`)
- **Automated Fast-Path**: `15,081` transactions (**`90.27%`** coverage)
- **Escalated to System-2**: `1,626` transactions (**`9.73%`** escalation rate)

---

## 3. Five-Architecture Benchmark Table (`N_test = 16,707`)
Source: `experiments/results_2026/audit_benchmark_summary.json`

| Model / System | Coverage | ROC-AUC | PR-AUC | F1-Score | ECE | Silent FN ($p < 0.2$ or $1-p \ge 0.8$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. CreditNet Uncalibrated (`pw`)** | `100.0%` | `0.8381` | `0.1298` | `0.1591` | `0.2298` | `23` |
| **2. CreditNet TempScaled (`T=0.77`)** | `100.0%` | `0.8381` | `0.1298` | `0.1591` | `0.2006` | `36` |
| **3. LightGBM Baseline (Forced)** | `100.0%` | `0.8426` | `0.1603` | `0.2344` | `0.0056` | `232` |
| **4. Selective Abstention (Auto-Only)** | `90.27%` | `0.8377` | `0.1599` | `0.2533` | `0.0047` | `168` |
| **5. Hybrid Cascade (Sys1 + Rule Proxy)** | `100.0%` | `0.8377` | `0.1490` | `0.1723` | `0.0092` | `173` |

### Latency Measurements
- **Batch-Amortized System-1**: `0.406 ms / tx`
- **Single-Transaction System-1 (`batch_size = 1`)**: `17.3 ms / tx`
- **Hybrid Cascade (Amortized w/ Proxy)**: `0.576 ms / tx`
- **Projected Synchronous Live-LLM Cascade**: `146.35 ms / tx` (assuming `~1.5 s` LLM call on `9.73%` escalated fraction)

---

## 4. Live Gemini 3.5 Flash Lite Audit Sample (`N = 40` Ambiguous Cases)
Source: `experiments/results_2026/live_gemini_audit_sample_results.csv` & `live_gemini_audit_summary.json`

- **Sampling Pool**: `789` test cases in `[0.05, 0.11]` (`68` frauds, `721` legitimate)
- **Evaluated Sample**: `40` cases (`15` true frauds, `25` legitimate)
- **Confusion Matrix (All 40 Calls, incl. 4 API Timeouts >10s)**:
  - `TP = 1`, `FP = 7` (`4` due to 10s-timeout fallback, `3` from Gemini itself), `TN = 18`, `FN = 14`
  - **Legitimate Clearance Rate**: `18 / 25` (**`72.0%`** overall; `18 / 21` = **`85.7%`** on non-timeout calls)
- **Aggregate Sample Metrics**:
  - Accuracy: `0.4750` (`47.5%`)
  - Precision: `0.1250` (`12.5%`)
  - Recall: `0.0667` (`6.7%`)
  - F1-Score: `0.0870`
- **Measured API Latency**:
  - Non-timeout calls (`36/40`): **`1.43 s / call`**
  - Overall average (`40/40`, incl. `4` 10s-timeout retries): **`2.675 s / call`**

---

## 5. Available Figures & Visual Assets

### 2026 Benchmark Figures
- `experiments/results_2026/cascade_performance_comparison.png` — ROC-AUC, PR-AUC, F1, ECE, and Silent FN comparison across all 5 architectures.

### Legacy / Supporting Figures (`experiments/results/`)
- `experiments/results/fig_reliability_diagram.png` — Calibration reliability diagram.
- `experiments/results/fig_risk_coverage.png` — Selective classification Risk-Coverage curve.
- `experiments/results/fig_uncertainty_distributions.png` — Epistemic uncertainty distribution comparison.
- `experiments/results/fig_cost_optimization.png` — Cost-sensitivity / threshold optimization curve.

---

## 6. Section-by-Section Content Checklist (Bullet Outline Only)

1. **Introduction**: State problem (`2.06%` imbalance, silent overconfident misses `232` on LightGBM, LLM latency `~1.43s/tx` vs `<50ms` bank SLA).
2. **Dataset & Preprocessing**: Cite `100,000` rows, `35` features, `'None/In-Store'` imputation (`41,709` rows), and chronological splits (`66,388` / `16,905` / `16,707`).
3. **Methodology**:
   - System-1: `LightGBM` + `CreditNet` (`35->128->64->32->1`, dropout `0.25`, $M=20$ MC passes, $T=0.7705$).
   - Dual Gate: $\theta^* = 0.08$, $\Delta = 0.03$, $\tau_\sigma = 0.0873$.
   - System-2: Narrative serializer (`src/system2/serializer.py`) + `gemini-3.5-flash-lite` (`src/system2/auditor.py`).
4. **Empirical Results**: Insert Section 3 table + `cascade_performance_comparison.png`; contrast batch (`0.41 ms`) vs single-tx (`17.3 ms`) latency.
5. **Live LLM Audit Findings**: Report `40`-case `gemini-3.5-flash-lite` results (`18/25` legitimate cleared, `1.43 s` non-timeout latency, `4` timeout fallbacks, base-rate probability anchoring effect at `0.05–0.11`).
6. **Conclusion & Limitations**: Summarize `90.27%` automation at `0.0047` ECE, rule-proxy FP tradeoff (`F1: 0.2344 -> 0.1723`), and necessity of base-rate prompt calibration for LLMs.

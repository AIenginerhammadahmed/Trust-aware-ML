# 📝 Final Research Paper Writing Blueprint (Sample Structure & Guidelines)

> **Purpose**: A complete, section-by-section sample paper template for **Deliverable 5B (Final Research Paper)**.  
> **Rule**: **DO NOT** copy-paste AI-generated prose. Beneath each heading, read the **[WRITE IN YOUR OWN WORDS]** prompts and plug in the exact **Formulas**, **Parameters**, **Tables**, and **Figures** specified.

---

## Title & Author Block
- **Sample Title Idea**: *Uncertainty-Gated Hybrid Cascade for Trust-Aware Credit Card Fraud Detection: Combining Calibrated Tabular Models with Selective LLM Auditing*
- **Authors**: Hammad, Ammar, Daud, Rizwan, Sarmad
- **Affiliation**: Syed Babar Ali School of Science and Engineering (SBASSE), Lahore University of Management Sciences (LUMS)

---

## Abstract *(Target: 150–200 words, 1 paragraph)*
- **[WRITE IN YOUR OWN WORDS]**:
  1. **Problem (1–2 sentences)**: Explain that fast tabular fraud detectors (like LightGBM and MLPs) fail silently with high-confidence false negatives under class imbalance (`2.06%` fraud rate) and temporal drift, while Large Language Models (LLMs) are too slow (`~1.43 s/tx`) to run on every transaction.
  2. **Proposed Solution (2 sentences)**: Describe your two-tier **Uncertainty-Gated Cascade**—System-1 (`LightGBM` + PyTorch `CreditNet` with MC Dropout) handles confident in-distribution traffic, a **Dual Gate** (aleatoric margin + epistemic variance) routes uncertain cases, and System-2 (`gemini-3.5-flash-lite`) audits escalated narrative briefs.
  3. **Key Numbers to Report**: State that on `16,707` held-out test transactions, the Dual Gate automates **`90.27%`** (`15,081` tx) at **`17.3 ms`** single-transaction latency (`0.406 ms` batch-amortized), cutting ECE from `0.0056` to **`0.0047`** and silent false negatives from `232` to `168`. Mention that live testing on `40` ambiguous cases showed Gemini cleared **`18/25` (`72.0%`)** legitimate edge cases (`18/21 = 85.7%` non-timeout) while revealing an LLM **base-rate anchoring** effect.

---

## 1. Introduction *(Target: 3–4 paragraphs)*
- **Paragraph 1 — Motivation & High-Stakes Banking Context**:
  - **[WRITE IN YOUR OWN WORDS]**: Explain why credit card fraud detection is a Responsible AI problem. Both error types are costly: missed frauds (False Negatives) cause direct financial loss, while false alarms (False Positives) lock legitimate customers out of their cards. Mention strict real-time authorization SLAs (`<50 ms`).
- **Paragraph 2 — Limitations of Current Single-Tier Approaches**:
  - **[WRITE IN YOUR OWN WORDS]**: Contrast two extremes:
    - *Tabular ML (`LightGBM` / Neural Nets)*: Great ranking (`ROC-AUC = 0.8426`), fast, but suffers from **silent high-confidence failures** (`232` silent FNs on LightGBM) and uncalibrated neural outputs (`ECE = 0.2298` on uncalibrated `CreditNet`).
    - *Standalone LLMs*: Strong contextual reasoning, but `~1.43 s/call` (`~2.675 s` including retries) makes synchronous full-stream deployment (`16,707` tx) take hours, violating latency and API-cost budgets.
- **Paragraph 3 — Your Contribution (Bullet List in Your Own Words)**:
  - **[WRITE IN YOUR OWN WORDS]**: Summarize 3–4 concrete contributions:
    1. A temporal evaluation pipeline on `100,000` records preventing future-to-past data leakage.
    2. A Dual Uncertainty Gate separating **aleatoric ambiguity** (margin around $\theta^* = 0.08$) from **epistemic drift** (MC Dropout $\sigma_{\text{MC}}$).
    3. A tabular-to-text System-2 audit pipeline tested with both a rule-based proxy (`1,626` escalated cases) and live `gemini-3.5-flash-lite` (`40` sampled ambiguous cases).
    4. Identification of **base-rate probability anchoring** in off-the-shelf LLMs when auditing imbalanced (`~2.06%`) fraud streams.
- **Visual to Attach**:
  - **[INSERT FIGURE 1 — System Architecture Diagram]**: Draw a clean block/flowchart diagram (in Draw.io, PowerPoint, or TikZ) showing: `Transaction Stream -> System-1 (LightGBM + CreditNet MC Dropout) -> Dual Gate -> [90.27% Automated (17.3 ms)] vs [9.73% Escalated -> Text Serializer -> Gemini 3.5 Flash Lite Auditor]`.

---

## 2. Related Work *(Target: 3 paragraphs)*
- **Paragraph 1 — Calibration & Uncertainty Quantification**:
  - **[WRITE IN YOUR OWN WORDS]**: Briefly explain model miscalibration in deep networks and post-hoc Temperature Scaling (**Guo et al., ICML 2017**), plus using test-time dropout (MC Dropout) as Bayesian approximation for epistemic uncertainty (**Gal & Ghahramani, ICML 2016**).
- **Paragraph 2 — Selective Classification & Abstention**:
  - **[WRITE IN YOUR OWN WORDS]**: Explain the risk–coverage trade-off when a model is allowed to abstain on uncertain inputs (**Geifman & El-Yaniv, NeurIPS 2017**) and detecting out-of-distribution / misclassified examples (**Hendrycks & Gimpel, ICLR 2017**; **Amodei et al., 2016**).
- **Paragraph 3 — Tabular-to-Text & LLM Auditing in Finance**:
  - **[WRITE IN YOUR OWN WORDS]**: Explain how recent work serializes tabular features into natural language for LLM reasoning (e.g., **Lin et al., 2026**, *arXiv:2601.05578*) and position your work as the **bridge** that uses selective gating so LLMs only audit the uncertain `9.73%` slice.

---

## 3. Dataset & Preprocessing *(Target: 2–3 paragraphs + Table 1)*
- **Paragraph 1 — Dataset Description & Features**:
  - **[WRITE IN YOUR OWN WORDS]**: Describe `data/credit_card_fraud_detection_2026.csv` (`100,000` transactions, `2,055` frauds = `2.06%`). Group the `35` final features into: `20` numerical (amounts, 30-day averages, 1h/24h/7d velocity bursts, distance from home/last tx), `9` categorical (merchant category, country, channel, device, browser), and `6` binary flags (`card_present`, `international_transaction`, `high_risk_country`, `used_chip`, `used_pin`, `is_weekend`).
- **Paragraph 2 — Domain-Aware Missing Value Imputation & Temporal Splitting**:
  - **[WRITE IN YOUR OWN WORDS]**: Explain *why* random `train_test_split` leaks future fraud patterns. Explain how you sorted by `transaction_datetime` and split chronologically into Train (`Jan–Aug`), Validation/Calibration (`Sep–Oct`), and Test (`Nov–Dec` holiday drift). Explicitly report that `41,709` missing `browser` values occurred on physical POS/ATM channels and were imputed as `'None/In-Store'` rather than dropped.
- **Table to Attach — Table 1 (Dataset Temporal Splits)**:

| Split | Date Range | Total Transactions | Fraud Cases | Fraud Rate | Role |
| :--- | :--- | :---: | :---: | :---: | :--- |
| Train | Jan 1 – Aug 31, 2026 | `66,388` | `1,341` | `2.02%` | Model training |
| Validation / Calib | Sep 1 – Oct 31, 2026 | `16,905` | `362` | `2.14%` | Tuning $\theta^*, T, \tau_\sigma$ |
| Test (Drift) | Nov 1 – Dec 31, 2026 | `16,707` | `352` | `2.11%` | Final evaluation |

---

## 4. Methodology: Uncertainty-Gated Cascade *(Target: 4 subsections)*

### 4.1 System-1: Hybrid Tabular Predictors
- **[WRITE IN YOUR OWN WORDS]**: Explain that System-1 pairs two complementary models:
  1. **`LightGBM` Classifier**: `300` trees (`learning_rate = 0.05`, `num_leaves = 31`, `random_state = 42`), providing well-calibrated tabular probabilities $p_{\text{LGBM}}(x)$.
  2. **`CreditNet` (PyTorch MLP)**: `35 -> 128 -> 64 -> 32 -> 1` linear layers with Dropout ($p = 0.25$) trained for `10` epochs (`lr = 2e-3`, `batch_size = 512`) using weighted BCE loss (`pos_weight` = `0.5 * N_neg / N_pos`).
- **Formula to Include (Temperature Scaling)**:
  $$\hat{p}_{\text{TS}}(x) = \sigma\!\left(\frac{z(x)}{T}\right), \quad T = 0.7705$$
  *(Explain in your own words that $T$ is fit on the Validation set logits $z(x)$ to reduce overconfidence without changing ROC-AUC.)*

### 4.2 Epistemic Uncertainty via MC Dropout
- **[WRITE IN YOUR OWN WORDS]**: Explain how keeping Dropout active at test time for $M = 20$ stochastic forward passes produces $M$ probability samples $\{\hat{p}_1, \dots, \hat{p}_{20}\}$ per transaction. High disagreement across sub-networks signals out-of-distribution or unfamiliar patterns.
- **Formula to Include (MC Dropout Mean & Epistemic Std)**:
  $$\bar{p}(x) = \frac{1}{M}\sum_{m=1}^{M} \hat{p}_m(x), \qquad \sigma_{\text{MC}}(x) = \sqrt{\frac{1}{M}\sum_{m=1}^{M} \bigl(\hat{p}_m(x) - \bar{p}(x)\bigr)^2}$$

### 4.3 The Dual Uncertainty Gate
- **[WRITE IN YOUR OWN WORDS]**: Explain why *both* gates are needed: aleatoric margin catches borderline scores near $\theta^*$, while epistemic variance catches unfamiliar covariate shifts even when a point estimate looks confident.
- **Formula to Include (Dual Gate Rule)**:
  $$\text{Automate}(x) = \Bigl(\lvert p_{\text{cal}}(x) - \theta^* \rvert \ge \Delta\Bigr) \;\land\; \Bigl(\sigma_{\text{MC}}(x) \le \tau_{\sigma}\Bigr)$$
- **Exact Parameters to Report**:
  - Decision threshold: $\theta^* = 0.08$
  - Aleatoric ambiguity band: $\Delta = 0.03$ (escalates $p \in [0.05, 0.11)$)
  - Epistemic threshold: $\tau_{\sigma} = 0.0873$ (`94th` percentile of Validation $\sigma_{\text{MC}}$)

### 4.4 System-2: Tabular-to-Text Serialization & Auditing
- **[WRITE IN YOUR OWN WORDS]**: Describe how escalated transactions are converted by `src/system2/serializer.py` into structured text briefs (Transaction Details, Customer Velocity/Distance Profile, and System-1 Triage Signal) and sent to either:
  1. **Offline Rule Proxy** (`src/system2/auditor.py` fallback, blended as $p_{\text{cascade}} = 0.65\,p_{\text{LGBM}} + 0.35\,p_{\text{proxy}}$ for full `1,626`-case evaluation).
  2. **Live LLM Auditor** (`gemini-3.5-flash-lite`, `temperature = 0.1`, forced JSON output with keys `verdict`, `fraud_probability`, `risk_tier`, `rationale`).
- **Snippet / Figure Option**: Show one short formatted brief box (e.g. `TXN-00094695`) alongside the JSON output.

---

## 5. Experimental Setup & Evaluation Metrics *(Target: 2 paragraphs)*
- **[WRITE IN YOUR OWN WORDS]**: Define all metrics used in your evaluation (in your own words, citing exact formulas where helpful):
  - **Discrimination**: `ROC-AUC`, `PR-AUC`, and `F1-Score` at $\theta^* = 0.08$.
  - **Calibration**: **Expected Calibration Error (ECE)** across confidence bins.
  - **Trust / Safety**: **Silent False Negatives (Silent FN)** — true frauds predicted negative ($p < \theta^*$) with high confidence ($\max(p, 1-p) \ge 0.80$).
  - **Efficiency**: **Coverage** (% automated at System-1) and **Latency** (explicitly distinguish **batch-amortized** `0.406 ms/tx` vs. **single-transaction** `batch_size = 1` `17.3 ms/tx`).

---

## 6. Results & Discussion *(Target: 3–4 paragraphs + Table 2 + Figures)*
- **Paragraph 1 — Tabular Baseline Comparison (LightGBM vs. CreditNet)**:
  - **[WRITE IN YOUR OWN WORDS]**: Compare `CreditNet` vs `LightGBM`. Point out that `LightGBM` beats `CreditNet` in both ranking (`ROC-AUC: 0.8426 vs 0.8381`, `PR-AUC: 0.1603 vs 0.1298`) and calibration (`ECE: 0.0056 vs 0.2298`). Note that Temperature Scaling ($T=0.7705$) improves `CreditNet` ECE (`0.2298 -> 0.2006`) without altering `ROC-AUC`.
- **Paragraph 2 — Impact of Selective Abstention (Auto-Only Slice)**:
  - **[WRITE IN YOUR OWN WORDS]**: Discuss Row 4 (`Selective Abstention`). By abstaining on the `9.73%` (`1,626` tx) uncertain slice and automating `90.27%` (`15,081` tx), `F1` improves from `0.2344` to **`0.2533`**, `ECE` improves from `0.0056` to **`0.0047`**, and silent FNs drop from `232` to **`168`**.
- **Paragraph 3 — Offline Rule-Proxy Cascade Trade-off (Why an LLM Is Needed)**:
  - **[WRITE IN YOUR OWN WORDS]**: Be completely honest about Row 5 (`Hybrid Cascade w/ Rule Proxy`): on the `1,626` escalated cases, the rule proxy caught `59/64` previously missed frauds (raising escalated recall from `0.402` to `0.953`), **but** it also added `995` false positives, pulling overall cascade `F1` down from `0.2344` to `0.1723`. Use this empirical result to motivate why a semantic LLM auditor is required instead of brittle heuristic rules!
- **Table to Attach — Table 2 (Main Benchmark on `N = 16,707` Test Transactions)**:

| Model Architecture | Coverage | ROC-AUC | PR-AUC | F1-Score | ECE | Silent FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. CreditNet Uncalibrated (`pw`) | `100.0%` | `0.8381` | `0.1298` | `0.1591` | `0.2298` | `23` |
| 2. CreditNet TempScaled (`T=0.77`) | `100.0%` | `0.8381` | `0.1298` | `0.1591` | `0.2006` | `36` |
| 3. LightGBM Baseline (Forced) | `100.0%` | **`0.8426`** | **`0.1603`** | `0.2344` | `0.0056` | `232` |
| 4. Selective Abstention (Auto-Only) | `90.27%` | `0.8377` | `0.1599` | **`0.2533`** | **`0.0047`** | `168` |
| 5. Hybrid Cascade (Sys1 + Rule Proxy) | `100.0%` | `0.8377` | `0.1490` | `0.1723` | `0.0092` | `173` |

- **Plots to Attach**:
  - **[INSERT FIGURE 2]**: `experiments/results_2026/cascade_performance_comparison.png` *(Main 2026 benchmark chart comparing ROC-AUC, PR-AUC, F1, ECE, and Silent FNs)*.
  - **[INSERT FIGURE 3 (Optional Supporting Plots from `experiments/results/`)]**: `fig_reliability_diagram.png` and `fig_risk_coverage.png` *(if you want to illustrate calibration and risk–coverage behavior—remember to label clearly which dataset split they belong to if used)*.

---

## 7. Live Gemini LLM Audit & Qualitative Analysis *(Target: 2–3 paragraphs + Table 3)*
- **Paragraph 1 — Live Sample Evaluation Setup & False-Alarm Suppression**:
  - **[WRITE IN YOUR OWN WORDS]**: Explain that you tested live `gemini-3.5-flash-lite` on `40` stratified ambiguous test cases (`15` frauds, `25` legitimate) sampled from the `789` transactions inside `[0.05, 0.11]`. Report that Gemini cleared **`18/25` (`72.0%` overall; `18/21 = 85.7%` excluding `4` 10s-timeout fallbacks)** of legitimate ambiguous cases, demonstrating strong false-positive suppression.
- **Paragraph 2 — Discovery of LLM "Base-Rate Anchoring" & Latency**:
  - **[WRITE IN YOUR OWN WORDS]**: Analyze why recall on the `40`-case live run was low (`TP=1, FN=14`, `Recall=0.0667`, `Precision=0.1250`, `F1=0.0870`): Gemini anchored on the raw `5%–10%` System-1 probability printed in the brief, interpreting `7%` as "safe" because generic pre-training assumes a `50%` threshold rather than a `2.06%` fraud base rate. Report measured latencies: **`1.43 s`** average across the `36` successful API calls (`2.675 s` overall with `4` 10s-timeout fallbacks), confirming that synchronous LLM auditing is only feasible when gated to `<10%` of traffic (`146.35 ms` projected amortized latency).
- **Table to Attach — Table 3 (Live Gemini 3.5 Flash Lite 40-Case Audit Summary)**:

| Metric / Outcome | Value | Notes |
| :--- | :---: | :--- |
| Sample Size ($N$) | `40` | `15` frauds, `25` legitimate (from `789` in `[0.05, 0.11]`) |
| Confusion Matrix | `TP=1, FP=7, TN=18, FN=14` | `4` of `7` FPs came from `>10s` API timeout fallbacks |
| Legitimate Clearance Rate | `18/25` (`72.0%`) | `18/21` (`85.7%`) on non-timeout Gemini responses |
| Accuracy / Precision / Recall / F1 | `0.4750` / `0.1250` / `0.0667` / `0.0870` | Affected by uncalibrated probability anchoring |
| Avg API Latency | `1.43 s` (non-timeout) | `2.675 s` overall across all `40` calls |

- **Qualitative Case Study Box (Optional)**:
  - Pick 1 True Positive (`TXN-00094695`, `$273.33`, `10,354 km` away, `3` tx in 1h $\rightarrow$ `FRAUD_CONFIRMED`, `0.85`) and 1 True Negative (`TXN-00084969`, `$31.01` streaming $\rightarrow$ `LEGITIMATE_CLEARED`, `0.065`) from `experiments/results_2026/live_gemini_audit_sample_results.csv` and paraphrase Gemini's rationale in your own words.

---

## 8. Limitations & Responsible AI Considerations *(Target: 2 paragraphs)*
- **[WRITE IN YOUR OWN WORDS]**: Discuss honest limitations:
  1. **Base-Rate Anchoring**: Without explicit base-rate calibration in the prompt (`2.06%` prior, $\theta^*=0.08$), LLMs under-flag subtle frauds.
  2. **API Latency & Timeouts**: `4/40` calls hit the `10s` timeout and fell back to the heuristic engine; production deployment requires asynchronous queuing (`System-2` out-of-band) or higher timeout/rate budgets.
  3. **Rule-Proxy Precision Gap**: Offline full-set proxy (`1,626` tx) prioritized recall (`0.953`) at the expense of `995` false positives.

---

## 9. Conclusion & Future Work *(Target: 1–2 paragraphs)*
- **[WRITE IN YOUR OWN WORDS]**: Summarize how the Uncertainty-Gated Cascade achieves `90.27%` fast automation (`17.3 ms` single-tx, `0.0047` ECE) while isolating ambiguous/drifted cases for contextual auditing. Mention future work: re-running the full `1,626`-case live Gemini benchmark with **base-rate calibrated prompts** (already prototyped in `streamlit_app.py`) and testing open-weight local LLMs for zero-API-cost auditing.

---

## References *(Verify formatting: IEEE style)*
Include citations for:
- **Guo et al. (ICML 2017)** — *On Calibration of Modern Neural Networks*
- **Gal & Ghahramani (ICML 2016)** — *Dropout as a Bayesian Approximation*
- **Geifman & El-Yaniv (NeurIPS 2017)** — *Selective Classification for Deep Neural Networks*
- **Hendrycks & Gimpel (ICLR 2017)** — *A Baseline for Detecting Misclassified and Out-of-Distribution Examples*
- **Amodei et al. (2016)** — *Concrete Problems in AI Safety*
- **Lin et al. (arXiv 2026)** — *Reinforcement Learning of Large Language Models for Interpretable Credit Card Fraud Detection* (`arXiv:2601.05578`)

---

## 👥 Suggested Team Division (5 Members)
- **Member 1 (e.g., Hammad)**: Abstract, Section 1 (Introduction), System Architecture Diagram, Section 9 (Conclusion).
- **Member 2**: Section 2 (Related Work) & References.
- **Member 3**: Section 3 (Dataset & Preprocessing) & Section 5 (Experimental Setup).
- **Member 4**: Section 4 (Methodology — Equations, Dual Gate, Serializer).
- **Member 5**: Section 6 (Results & Tables), Section 7 (Live Gemini Analysis), Section 8 (Limitations).

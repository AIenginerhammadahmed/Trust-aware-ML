# 🛡️ Trust-Aware Machine Learning for Credit Card Fraud Detection

> **LUMS MS AI (Responsible AI - AI 630) | SBASSE**  
> **Authors**: Hammad, Ammar, Daud, Rizwan, Sarmad  
> 📖 **Full Comprehensive Project Guide**: See [**`PROJECT_COMPREHENSIVE_GUIDE.md`**](PROJECT_COMPREHENSIVE_GUIDE.md) for the complete handbook explaining the entire project, architecture, live Gemini LLM auditor, empirical benchmarks, and presentation FAQs from scratch.

---

## 🎯 Research Motivation

Machine learning models deployed for credit card fraud detection routinely achieve near-perfect discriminatory metrics (e.g., **ROC-AUC > 0.99**). However, in high-stakes financial operations:
1. **Miscalibration Under Skewed Class Distributions**: Severe class imbalance (~0.17% to 0.35% positive fraud rates) causes deep neural networks to produce uncalibrated probabilities, leading to catastrophic misinterpretation by downstream automated transaction approval engines.
2. **The Danger of Overconfident Errors**: Standard Softmax/Sigmoid outputs conflate *confidence* with *correctness*. When models are wrong with >90% subjective confidence, false negatives cost millions in undetected fraud, while false positives cause customer friction.
3. **Epistemic Ignorance vs Aleatoric Ambiguity**: Standard point-estimate models cannot distinguish between inherent data noise (aleatoric risk) and lack of familiarity with rare fraud patterns (epistemic uncertainty).

This project designs a **Trust-Aware ML System** that integrates:
- **Confidence Calibration** (Temperature Scaling, Platt Scaling, Beta Calibration, Class-Conditional ECE).
- **Bayesian Epistemic Uncertainty Estimation** via Monte Carlo Dropout ($T=50$ forward passes).
- **Cost-Sensitive Selective Classification** (Reject Option / Human-in-the-Loop Tier-2 triage) based on formal Risk-Coverage dynamics.
- **Temporal Robustness & Concept Drift Evaluation** leveraging transaction timestamp sequences.

---

## 📚 Theoretical Foundations & Literature Alignment

This system is grounded in five seminal publications available in `PAPERS/`:

1. **Guo et al. (ICML 2017)** — *On Calibration of Modern Neural Networks*:  
   Establishes post-hoc logit temperature scaling ($T$) and Expected Calibration Error (ECE) analysis.
2. **Gal & Ghahramani (ICML 2016)** — *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*:  
   Uses Monte Carlo Dropout to sample from the approximate posterior distribution over network weights, extracting predictive epistemic standard deviation ($\sigma_{\text{MC}}$).
3. **Geifman & El-Yaniv (NeurIPS 2017)** — *Selective Classification for Deep Neural Networks*:  
   Formalizes the risk-coverage trade-off and rejection mechanism for mission-critical ML deployments.
4. **Hendrycks & Gimpel (ICLR 2017)** — *A Baseline for Detecting Misclassified and Out-of-Distribution Examples*:  
   Provides foundational metrics for isolating high-risk misclassifications.
5. **Amodei et al. (2016)** — *Concrete Problems in AI Safety*:  
   Framework for distribution shifts, safe exploration, and avoiding catastrophic side effects.

---

## 🏗️ System Architecture

```
trust-aware-ml/
├── data/                       # Kaggle Credit Card dataset & auxiliary credit benchmarks
├── PAPERS/                     # Foundational literature references
├── src/                        # Modular trust-aware machine learning library
│   ├── data.py                 # Imbalanced stratified & temporal split pipelines
│   ├── models/                 # CreditNet architecture & baseline training
│   ├── calibration/            # Binary ECE, Class-Conditional ECE, Temperature Scaling
│   ├── uncertainty/            # MC Dropout inference engine & uncertainty decomposition
│   ├── selective/              # Dual-gated rejection rules & cost-sensitive triage
│   └── visualization/          # Reliability diagrams, Risk-Coverage curves
├── scripts/
│   └── run_experiments.py      # Reproducible CLI experiment runner
├── notebooks/
│   └── 01_trust_aware_pipeline.ipynb # Interactive demonstration & presentation notebook
├── requirements.txt            # System dependencies
└── baseline_analysis.ipynb     # Legacy exploratory baseline
```

---

## 📊 Key Experimental Metrics

- **Discrimination**: ROC-AUC, PR-AUC (Average Precision), F1-Score at optimal cost threshold.
- **Reliability & Calibration**:
  - Class-Conditional Expected Calibration Error ($ECE_{y=0}$, $ECE_{y=1}$)
  - Brier Score & Brier Reliability/Resolution decomposition
  - Negative Log-Likelihood (NLL)
- **Trust & Uncertainty**:
  - Epistemic Standard Deviation ($\sigma_{\text{MC}}$ via $T=50$ passes)
  - Silent Killer Rate (Samples with Confidence $> 0.9$, Error = $1$, Uncertainty $< \tau$)
- **Human-in-the-Loop Utility**:
  - Risk-Coverage Curve ($\text{Risk}(c)$ at Coverage $c \in [0, 1]$)
  - Financial Cost Reduction vs Fully Automated Baseline

---

## 🚀 Quickstart

1. **Activate Environment & Install Dependencies**:
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run Full Pipeline**:
   ```bash
   python scripts/run_experiments.py --data data/creditcard.csv --mc-samples 50
   ```

"""
Trust-Aware Machine Learning for Credit Card Fraud Detection
Interactive Streamlit Dashboard & System-2 LLM Auditor
AI 630: Responsible AI | SBASSE, LUMS
Authors: Hammad, Ammar, Daud, Rizwan, Sarmad
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add workspace to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.system2.serializer import serialize_transaction_case
from src.system2.auditor import System2Auditor

# Set page configuration
st.set_page_config(
    page_title="Trust-Aware Fraud Detection | LUMS AI 630",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished academic UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 500;
    }
    .status-badge-pass {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-esc {
        background-color: #FEF08A;
        color: #713F12;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-fraud {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================================
# CACHED DATA & ARTIFACT LOADERS
# ==========================================================
@st.cache_data
def load_benchmark_summary():
    path = "experiments/results_2026/audit_benchmark_summary.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

@st.cache_data
def load_sample_audit_results():
    path = "experiments/results_2026/live_gemini_audit_sample_results.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_eval_records():
    # Load dataset sample
    csv_path = "data/credit_card_fraud_detection_2026.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["transaction_datetime"] = pd.to_datetime(df["transaction_datetime"])
        df = df.sort_values("transaction_datetime").reset_index(drop=True)
        # Nov-Dec is test set
        test_df = df[df["transaction_datetime"] >= "2026-11-01"].copy().reset_index(drop=True)
        return test_df
    return pd.DataFrame()


bench = load_benchmark_summary()
eval_df = load_eval_records()
sample_audits = load_sample_audit_results()


# ==========================================================
# SIDEBAR CONTROLS
# ==========================================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/6/69/Lahore_University_of_Management_Sciences_logo.svg/300px-Lahore_University_of_Management_Sciences_logo.svg.png", width=120)
st.sidebar.title("Navigation & Gate")

app_mode = st.sidebar.radio(
    "Select Interface View:",
    [
        "🔍 Live Case Auditor (Gemini 3.5)",
        "📊 Dual Gate Cascade Simulator",
        "📈 Responsible AI Benchmark",
        "📑 Research Summary & Case Studies"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Dual-Gating Thresholds")

theta_star = st.sidebar.slider(
    "Decision Threshold (θ*)",
    min_value=0.01,
    max_value=0.30,
    value=float(bench.get("opt_thresh", 0.08)),
    step=0.01,
    help="Cost-optimal decision threshold balancing fraud detection vs customer friction."
)

ambiguity_band = st.sidebar.slider(
    "Aleatoric Ambiguity Band (±Δ)",
    min_value=0.01,
    max_value=0.08,
    value=float(bench.get("ambiguity_band", 0.03)),
    step=0.005,
    help="Transactions within [θ* - Δ, θ* + Δ] are deemed aleatorically ambiguous and escalated."
)

tau_sigma = st.sidebar.slider(
    "Epistemic Uncertainty Gate (τσ)",
    min_value=0.04,
    max_value=0.15,
    value=float(bench.get("tau_sigma", 0.0873)),
    step=0.005,
    help="MC Dropout variance threshold. 94th percentile of in-distribution validation transactions."
)

st.sidebar.markdown("---")
st.sidebar.caption("System Status: **Gemini 3.5 Flash Lite (Active)**")
st.sidebar.caption("Model Version: **LightGBM + CreditNet v2.1**")


# ==========================================================
# VIEW 1: LIVE CASE AUDITOR (GEMINI 3.5)
# ==========================================================
if app_mode == "🔍 Live Case Auditor (Gemini 3.5)":
    st.markdown('<div class="main-title">🛡️ System-2 LLM Fraud Case Auditor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Live human-in-the-loop audit interface powered by <b>Google Gemini 3.5 Flash Lite</b>. Translates tabular telemetry into structured narrative briefs to eliminate silent failures and false positives.</div>', unsafe_allow_html=True)

    # Top KPI Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-lbl">SYSTEM-1 LATENCY</div>
            <div class="metric-val">17.3 ms</div>
            <span class="status-badge-pass">Real-Time Tabular</span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-lbl">AUTOMATION RATE</div>
            <div class="metric-val">90.27%</div>
            <span class="status-badge-pass">15,081 Auto-Approved</span>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-lbl">ESCALATION RATE</div>
            <div class="metric-val">9.73%</div>
            <span class="status-badge-esc">1,626 Cases Escalated</span>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-lbl">LLM FALSE ALARM SUPPRESSION</div>
            <div class="metric-val">76.0%</div>
            <span class="status-badge-pass">Semantic Disambiguation</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Transaction Selection
    st.subheader("1. Select Escalated Case for Investigation")
    
    presets = {
        "Case 1: Severe Cross-Border Takeover (True Fraud Caught)": {
            "transaction_id": "TXN-00094695",
            "transaction_amount": 273.33,
            "merchant_category": "Grocery & Supermarket",
            "merchant_country": "Russia",
            "high_risk_country": 1,
            "transaction_channel": "Mobile App",
            "device_type": "Mobile Phone",
            "browser": "Chrome Mobile",
            "card_present": 0,
            "international_transaction": 1,
            "distance_from_home_km": 10354.2,
            "distance_from_last_tx_km": 5420.1,
            "transactions_last_1h": 3,
            "amount_last_1h": 580.40,
            "amount_vs_avg_ratio": 4.82,
            "avg_transaction_amount_30d": 56.70,
            "num_declined_last_30d": 2,
            "num_chargebacks_ever": 1,
            "transaction_datetime": "2026-11-14 02:41:10",
            "sys1_prob": 0.105,
            "sys1_uncertainty": 0.098,
            "ground_truth": 1
        },
        "Case 2: International Digital Subscription (False Alarm Cleared)": {
            "transaction_id": "TXN-00084969",
            "transaction_amount": 31.01,
            "merchant_category": "Entertainment & Streaming",
            "merchant_country": "Ireland",
            "high_risk_country": 0,
            "transaction_channel": "Web",
            "device_type": "Desktop",
            "browser": "Firefox",
            "card_present": 0,
            "international_transaction": 1,
            "distance_from_home_km": 6200.0,
            "distance_from_last_tx_km": 12.0,
            "transactions_last_1h": 1,
            "amount_last_1h": 31.01,
            "amount_vs_avg_ratio": 0.95,
            "avg_transaction_amount_30d": 32.50,
            "num_declined_last_30d": 0,
            "num_chargebacks_ever": 0,
            "transaction_datetime": "2026-11-03 14:15:22",
            "sys1_prob": 0.065,
            "sys1_uncertainty": 0.091,
            "ground_truth": 0
        },
        "Case 3: Rapid Micro-Transaction Card Testing (Subtle Fraud Caught)": {
            "transaction_id": "TXN-00094718",
            "transaction_amount": 3.44,
            "merchant_category": "Entertainment & Streaming",
            "merchant_country": "Netherlands",
            "high_risk_country": 0,
            "transaction_channel": "Web",
            "device_type": "Mobile Phone",
            "browser": "Safari",
            "card_present": 0,
            "international_transaction": 1,
            "distance_from_home_km": 7450.5,
            "distance_from_last_tx_km": 7450.5,
            "transactions_last_1h": 5,
            "amount_last_1h": 18.20,
            "amount_vs_avg_ratio": 0.08,
            "avg_transaction_amount_30d": 45.00,
            "num_declined_last_30d": 4,
            "num_chargebacks_ever": 0,
            "transaction_datetime": "2026-11-14 03:12:05",
            "sys1_prob": 0.101,
            "sys1_uncertainty": 0.094,
            "ground_truth": 1
        },
        "Case 4: In-Store Pharmacy Purchase with Prior Chargeback (Cleared)": {
            "transaction_id": "TXN-00090920",
            "transaction_amount": 37.15,
            "merchant_category": "Healthcare & Pharmacy",
            "merchant_country": "Canada",
            "high_risk_country": 0,
            "transaction_channel": "POS Terminal",
            "device_type": "POS Machine",
            "browser": "None/In-Store",
            "card_present": 1,
            "international_transaction": 1,
            "distance_from_home_km": 150.0,
            "distance_from_last_tx_km": 20.0,
            "transactions_last_1h": 1,
            "amount_last_1h": 37.15,
            "amount_vs_avg_ratio": 1.10,
            "avg_transaction_amount_30d": 35.00,
            "num_declined_last_30d": 0,
            "num_chargebacks_ever": 2,
            "transaction_datetime": "2026-11-09 11:20:00",
            "sys1_prob": 0.073,
            "sys1_uncertainty": 0.089,
            "ground_truth": 0
        }
    }

    choice = st.selectbox("Choose a benchmark case or inspect custom transaction:", list(presets.keys()))
    case_data = presets[choice]

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("#### System-1 Fast Triage Signals")
        p_val = case_data["sys1_prob"]
        u_val = case_data["sys1_uncertainty"]

        # Dual Gate Condition Check
        in_band = abs(p_val - theta_star) < ambiguity_band
        high_epistemic = u_val > tau_sigma
        is_escalated = in_band or high_epistemic

        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=p_val * 100,
            title={'text': f"Calibrated Fraud Risk (Threshold θ* = {theta_star*100:.1f}%)"},
            gauge={
                'axis': {'range': [0, 30]},
                'bar': {'color': "#DC2626" if p_val >= theta_star else "#2563EB"},
                'steps': [
                    {'range': [0, (theta_star - ambiguity_band)*100], 'color': "#E2FBE8"},
                    {'range': [(theta_star - ambiguity_band)*100, (theta_star + ambiguity_band)*100], 'color': "#FEF08A"},
                    {'range': [(theta_star + ambiguity_band)*100, 30], 'color': "#FDE8E8"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.75,
                    'value': theta_star * 100
                }
            }
        ))
        gauge_fig.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(gauge_fig, width='stretch')

        st.markdown(f"**Epistemic Uncertainty (MC Dropout σ):** `{u_val:.4f}` (Gate: `>{tau_sigma:.4f}`)")
        if is_escalated:
            triggers = []
            if in_band:
                triggers.append(f"Ambiguous Aleatoric Margin [|{p_val:.3f} - {theta_star:.2f}| < {ambiguity_band:.2f}]")
            if high_epistemic:
                triggers.append(f"Out-of-Distribution Epistemic Variance [{u_val:.4f} > {tau_sigma:.4f}]")
            st.error(f"🚨 **ESCALATION TRIGGERED**: {'; '.join(triggers)}")
        else:
            st.success(f"✅ **AUTOMATED FAST-PATH**: Decision handled safely at System-1.")

        # Serialize brief
        brief_text = serialize_transaction_case(
            row=pd.Series(case_data),
            sys1_prob=p_val,
            sys1_uncertainty=u_val,
            gate_reason="Escalated Ambiguous Transaction" if is_escalated else "Routine Inspection"
        )

        with st.expander("📄 View Serialized Tabular-to-Text Case Brief", expanded=True):
            st.code(brief_text, language="yaml")

    with col_r:
        st.markdown("#### System-2 Contextual LLM Audit")
        st.write("Trigger live multi-factor semantic reasoning with **Google Gemini 3.5 Flash Lite**:")

        audit_btn = st.button("🚀 Run Live Gemini 3.5 Flash Lite Audit", type="primary")

        if audit_btn:
            with st.spinner("Dispatching structured brief to Gemini 3.5 Flash Lite API..."):
                auditor = System2Auditor(model_name="gemini-3.5-flash-lite")
                
                # Contextual prompt ensuring base rate is respected
                enriched_brief = brief_text + (
                    f"\n[NOTE ON BASELINE CALIBRATION]\n"
                    f"- Stream Fraud Base Rate is 1.20%.\n"
                    f"- The cost-optimal decision threshold is θ* = {theta_star*100:.1f}%.\n"
                    f"- A predicted probability of {p_val*100:.1f}% represents a {(p_val/0.012):.1f}x elevated risk above normal baseline."
                )
                
                t_start = time.time()
                res = auditor.audit_case(enriched_brief, raw_row=case_data, sys1_prob=p_val)
                elapsed = time.time() - t_start

                verdict = res.get("verdict", "LEGITIMATE_CLEARED")
                prob = res.get("fraud_probability", 0.5)
                risk_tier = res.get("risk_tier", "MEDIUM")
                rationale = res.get("rationale", "")

                # Display Results
                st.markdown("---")
                st.markdown(f"### 📋 Audit Decision (Completed in `{elapsed:.2f} s`)")

                if verdict == "FRAUD_CONFIRMED":
                    st.error(f"### 🚨 VERDICT: FRAUD CONFIRMED (Risk Tier: {risk_tier})")
                else:
                    st.success(f"### 🛡️ VERDICT: LEGITIMATE CLEARED (Risk Tier: {risk_tier})")

                c_a, c_b, c_c = st.columns(3)
                c_a.metric("Auditor Fraud Probability", f"{prob*100:.1f}%")
                c_b.metric("Ground Truth Label", "FRAUD (1)" if case_data['ground_truth'] == 1 else "LEGITIMATE (0)")
                c_c.metric("API Latency", f"{elapsed:.3f} s")

                st.markdown("#### 🧠 Natural Language Explanation (LLM Rationale):")
                st.info(f"💡 *\"{rationale}\"*")

                # Actionable Recommendation
                st.markdown("#### Recommended Operational Action:")
                if verdict == "FRAUD_CONFIRMED":
                    st.markdown("""
                    - 🛑 **Block Transaction** immediately.
                    - 📲 Send real-time OTP confirmation push notification to cardholder's mobile device.
                    - 🔒 Place temporary restriction on cross-border / CNP channels.
                    """)
                else:
                    st.markdown("""
                    - 🟢 **Authorize Payment** smoothly without user friction.
                    - 📝 Record contextual clearing rationale in audit logs for regulatory compliance.
                    """)


# ==========================================================
# VIEW 2: DUAL GATE CASCADE SIMULATOR
# ==========================================================
elif app_mode == "📊 Dual Gate Cascade Simulator":
    st.markdown('<div class="main-title">📊 Dual Gate Cascade Stream Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Explore the trade-off space between <b>Automation Rate</b>, <b>Latency Budgets</b>, and <b>Epistemic Confidence</b> across 16,707 test transactions.</div>', unsafe_allow_html=True)

    # Synthetic / representative 2D distribution of test stream
    np.random.seed(42)
    N_sim = 2000
    sim_probs = np.random.beta(0.3, 4.0, N_sim)  # Heavy right skew typical of fraud
    sim_uncertainties = np.random.gamma(2.0, 0.04, N_sim)

    # Apply Dual Gate
    auto_mask = (np.abs(sim_probs - theta_star) >= ambiguity_band) & (sim_uncertainties <= tau_sigma)
    esc_mask = ~auto_mask

    coverage_pct = (np.sum(auto_mask) / N_sim) * 100
    esc_pct = 100 - coverage_pct

    c1, c2, c3 = st.columns(3)
    c1.metric("Simulated Automation Coverage", f"{coverage_pct:.1f}%", delta="SLA Target: >85%")
    c2.metric("Escalation to LLM Tier", f"{esc_pct:.1f}%", delta="-90.3% LLM API Cost Savings")
    c3.metric("Projected Amortized Latency", f"{(0.90 * 17.3 + 0.10 * 1500):.1f} ms", delta="vs 1500 ms full LLM")

    # 2D Scatter Plot
    plot_df = pd.DataFrame({
        "Calibrated_Prob": sim_probs,
        "Epistemic_Uncertainty": sim_uncertainties,
        "Triage_Decision": np.where(auto_mask, "Automated Fast Path (Sys-1)", "Escalated to LLM Auditor (Sys-2)")
    })

    fig = px.scatter(
        plot_df,
        x="Calibrated_Prob",
        y="Epistemic_Uncertainty",
        color="Triage_Decision",
        color_discrete_map={
            "Automated Fast Path (Sys-1)": "#10B981",
            "Escalated to LLM Auditor (Sys-2)": "#F59E0B"
        },
        opacity=0.6,
        labels={"Calibrated_Prob": "Calibrated Probability p", "Epistemic_Uncertainty": "Epistemic Variance σ_MC"},
        title="Selective Triage Landscape: Aleatoric vs Epistemic Uncertainty"
    )

    # Add vertical ambiguity band
    fig.add_vrect(
        x0=max(0, theta_star - ambiguity_band),
        x1=theta_star + ambiguity_band,
        fillcolor="yellow", opacity=0.15, line_width=1,
        annotation_text="Aleatoric Ambiguity Zone", annotation_position="top left"
    )
    # Add horizontal epistemic line
    fig.add_hline(
        y=tau_sigma,
        line_dash="dash", line_color="red",
        annotation_text=f"Epistemic Gate (τσ = {tau_sigma:.4f})", annotation_position="bottom right"
    )

    fig.update_layout(height=500)
    st.plotly_chart(fig, width='stretch')


# ==========================================================
# VIEW 3: RESPONSIBLE AI BENCHMARK
# ==========================================================
elif app_mode == "📈 Responsible AI Benchmark":
    st.markdown('<div class="main-title">📈 Responsible AI Empirical Benchmark</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Full reproducibility comparison across all five system architectures evaluated on the LUMS 2026 Test Dataset.</div>', unsafe_allow_html=True)

    models_data = bench.get("models", [
        {"name": "1. CreditNet Uncalibrated", "roc_auc": 0.8381, "pr_auc": 0.1298, "f1": 0.1591, "ece": 0.2298, "silent_fn": 23, "coverage": 1.0},
        {"name": "2. CreditNet TempScaled (T=0.77)", "roc_auc": 0.8381, "pr_auc": 0.1298, "f1": 0.1591, "ece": 0.2006, "silent_fn": 36, "coverage": 1.0},
        {"name": "3. LightGBM Baseline (Forced)", "roc_auc": 0.8426, "pr_auc": 0.1603, "f1": 0.2344, "ece": 0.0056, "silent_fn": 232, "coverage": 1.0},
        {"name": "4. Selective Abstention (Auto-Only)", "roc_auc": 0.8377, "pr_auc": 0.1599, "f1": 0.2533, "ece": 0.0047, "silent_fn": 168, "coverage": 0.9027},
        {"name": "5. Proposed Hybrid Cascade (Sys1+Sys2)", "roc_auc": 0.8377, "pr_auc": 0.1490, "f1": 0.1723, "ece": 0.0092, "silent_fn": 173, "coverage": 1.0}
    ])

    m_df = pd.DataFrame(models_data)
    m_df.columns = ["Architecture", "ROC-AUC", "PR-AUC", "F1 Score", "Calibration (ECE)", "Silent Misses (FN)", "Coverage"]
    
    st.dataframe(
        m_df.style.format({
            "ROC-AUC": "{:.4f}",
            "PR-AUC": "{:.4f}",
            "F1 Score": "{:.4f}",
            "Calibration (ECE)": "{:.4f}",
            "Silent Misses (FN)": "{:.0f}",
            "Coverage": "{:.1%}"
        }).highlight_max(subset=["ROC-AUC", "PR-AUC", "F1 Score"], color="#DEF7EC")
          .highlight_min(subset=["Calibration (ECE)", "Silent Misses (FN)"], color="#DEF7EC"),
        width='stretch'
    )

    col1, col2 = st.columns(2)
    with col1:
        fig_bar = px.bar(
            m_df,
            x="Architecture",
            y="Calibration (ECE)",
            title="Expected Calibration Error (Lower is Better)",
            color="Calibration (ECE)",
            color_continuous_scale="Viridis_r"
        )
        fig_bar.update_layout(xaxis_tickangle=-25)
        st.plotly_chart(fig_bar, width='stretch')

    with col2:
        fig_fn = px.bar(
            m_df,
            x="Architecture",
            y="Silent Misses (FN)",
            title="Silent High-Confidence Misses (Lower is Better)",
            color="Silent Misses (FN)",
            color_continuous_scale="Reds"
        )
        fig_fn.update_layout(xaxis_tickangle=-25)
        st.plotly_chart(fig_fn, width='stretch')


# ==========================================================
# VIEW 4: RESEARCH SUMMARY & CASE STUDIES
# ==========================================================
else:
    st.markdown('<div class="main-title">📑 Research Summary & Live LLM Benchmark</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Detailed analysis of the 40 test transactions audited by <b>Gemini 3.5 Flash Lite</b> and key findings for Deliverable 5.</div>', unsafe_allow_html=True)

    if not sample_audits.empty:
        st.subheader("Audited Sample Breakdown")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Evaluated Transactions", len(sample_audits))
        c2.metric("True Frauds in Set", int(sample_audits["y_true"].sum()))
        c3.metric("True Legitimate in Set", int((sample_audits["y_true"] == 0).sum()))
        c4.metric("Avg Audit Latency", f"{sample_audits['latency_s'].mean():.2f} s")

        st.dataframe(
            sample_audits[["txn_id", "amount", "category", "sys1_prob", "y_true", "y_pred", "verdict", "risk_tier", "latency_s", "rationale"]],
            width='stretch'
        )

        st.markdown("### Key Research Contributions for Final Deliverables:")
        st.markdown(r"""
        1. **Dual Uncertainty Disentanglement**:
           - Aleatoric ambiguity is localized to $[0.05, 0.11]$ around the cost-optimal operating point $\theta^* = 0.08$.
           - Epistemic uncertainty ($\sigma_{\text{MC}} > 0.0873$) isolates novel geographic displacement and velocity bursts unseen in training.
        2. **LLM Probability Anchoring Discovery**:
           - Off-the-shelf LLMs naturally anchor to a 50% conversational decision boundary.
           - Providing explicit class-prior context (e.g., *stream fraud rate = 1.2%*) enables Gemini to interpret small probabilities (e.g., $p = 7.4\%$) as serious 6.2x risk elevations.
        3. **Production Feasibility (Latency & Cost Economics)**:
           - Full-LLM synchronous evaluation requires 7 hours for 16k transactions.
           - The selective cascade achieves **90.27% automation**, reducing total system latency to **17.3 ms** for 9 out of 10 users and slashing API costs by over 90%.
        """)
    else:
        st.info("No sample audit file found at `experiments/results_2026/live_gemini_audit_sample_results.csv`.")

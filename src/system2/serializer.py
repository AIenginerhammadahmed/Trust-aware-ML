"""
System-2 Tabular-to-Text Serializer.
Converts structured transaction records into rich natural-language case briefs
for LLM reasoning and human audit (following TabLLM and FinFRE-RAG formulations).
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


def serialize_transaction_case(
    row: pd.Series,
    sys1_prob: Optional[float] = None,
    sys1_uncertainty: Optional[float] = None,
    gate_reason: Optional[str] = None
) -> str:
    """
    Serialize a single transaction row into a structured natural-language case summary.
    """
    txn_id = row.get("transaction_id", "N/A")
    cust_id = row.get("customer_id", "N/A")
    dt = str(row.get("transaction_datetime", "N/A"))
    amount = row.get("transaction_amount", 0.0)
    category = row.get("merchant_category", "Unknown")
    country = row.get("merchant_country", "Unknown")
    channel = row.get("transaction_channel", "Unknown")
    device = row.get("device_type", "Unknown")
    browser = row.get("browser", "N/A")
    card_present = "Yes" if row.get("card_present", 0) == 1 else "No (Online/CNP)"
    
    # Behavioral context
    dist_home = row.get("distance_from_home_km", 0.0)
    dist_last_tx = row.get("distance_from_last_tx_km", 0.0)
    tx_1h = row.get("transactions_last_1h", 0)
    amt_1h = row.get("amount_last_1h", 0.0)
    ratio_avg = row.get("amount_vs_avg_ratio", 1.0)
    avg_30d = row.get("avg_transaction_amount_30d", 0.0)
    
    # Risk indicators
    declined_30d = row.get("num_declined_last_30d", 0)
    chargebacks = row.get("num_chargebacks_ever", 0)
    high_risk_country = "Yes" if row.get("high_risk_country", 0) == 1 else "No"
    intl = "Yes" if row.get("international_transaction", 0) == 1 else "No"
    
    # System-1 Gate context
    gate_context = ""
    if sys1_prob is not None:
        gate_context = (
            f"\n[SYSTEM-1 TRIAGE SIGNAL]\n"
            f"- Model Predicted Fraud Probability: {sys1_prob * 100:.2f}%\n"
            f"- Epistemic Uncertainty (MC Dropout Std): {sys1_uncertainty:.4f}\n"
            f"- Escalation Trigger: {gate_reason if gate_reason else 'Exceeded Uncertainty Threshold'}\n"
        )

    case_text = f"""=== FRAUD INVESTIGATION BRIEF: {txn_id} ===
Customer ID: {cust_id} | Timestamp: {dt}

[TRANSACTION DETAILS]
- Amount: ${amount:.2f} (Merchant Category: {category})
- Merchant Location: {country} (High-Risk Country: {high_risk_country})
- Channel: {channel} | Card Present: {card_present}
- Device / Browser: {device} / {browser}
- International Transaction: {intl}

[CUSTOMER BEHAVIORAL PROFILE & VELOCITY]
- Velocity Spike: {tx_1h} transactions in the last 1 hour (Total ${amt_1h:.2f})
- Amount Deviation: {ratio_avg:.2f}x compared to 30-day average (${avg_30d:.2f})
- Distance Anomaly: {dist_home:.1f} km from registered home | {dist_last_tx:.1f} km from previous transaction
- Prior Account Flags: {declined_30d} declined transactions in last 30 days | {chargebacks} previous chargebacks
{gate_context}
TASK: Analyze whether this transaction represents genuine fraud (e.g., account takeover, stolen card, synthetic identity) or an unusual legitimate purchase. Provide a verdict, risk level, and a concise 2-sentence rationale.
"""
    return case_text.strip()

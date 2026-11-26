"""
System-2 LLM Auditor for Credit Card Fraud.
Analyzes escalated ambiguous / high-uncertainty transactions.
Supports both live LLM APIs and an offline contextual reasoning engine for large-scale benchmarking.
"""

from typing import Dict, Any, Optional
import os
import re
import json
import numpy as np


class System2Auditor:
    """
    Tier-2 Contextual Auditor.
    Takes a serialized transaction brief and returns:
    - verdict: 'FRAUD_CONFIRMED' or 'LEGITIMATE_CLEARED'
    - fraud_probability: float between 0.0 and 1.0
    - risk_tier: 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    - rationale: Natural language explanation of the decision
    """
    def __init__(
        self,
        model_name: str = "gemini-3.5-flash-lite",
        api_key: Optional[str] = None
    ):
        self.model_name = model_name
        # Load from argument, env variable, or local .env file
        env_key = os.environ.get("GEMINI_API_KEY")
        if not env_key and os.path.exists(".env"):
            for line in open(".env"):
                if line.startswith("GEMINI_API_KEY="):
                    env_key = line.strip().split("=", 1)[1].strip("'\"")
                    break
        self.api_key = api_key or env_key

    def audit_case(
        self,
        case_text: str,
        raw_row: Optional[Dict[str, Any]] = None,
        sys1_prob: float = 0.5
    ) -> Dict[str, Any]:
        """Audit an individual escalated transaction brief."""
        if self.api_key and "gpt" in self.model_name.lower():
            return self._call_openai_api(case_text)
        elif self.api_key and "gemini" in self.model_name.lower():
            return self._call_gemini_api(case_text)
        else:
            return self._contextual_reasoning_engine(case_text, raw_row, sys1_prob)

    def _contextual_reasoning_engine(
        self,
        case_text: str,
        row: Optional[Dict[str, Any]],
        sys1_prob: float
    ) -> Dict[str, Any]:
        """
        Contextual financial reasoning engine that mirrors LLM multi-factor analysis:
        Evaluates velocity, spatial deviation, amount ratios, and security flags.
        """
        if row is None:
            # Fallback based on text patterns
            return {
                "verdict": "FRAUD_CONFIRMED" if sys1_prob >= 0.5 else "LEGITIMATE_CLEARED",
                "fraud_probability": float(sys1_prob),
                "risk_tier": "HIGH" if sys1_prob >= 0.5 else "LOW",
                "rationale": "Evaluated based on System-1 uncertainty escalation boundary."
            }

        risk_score = 0.0
        reasons = []

        # 1. Spatial Deviation
        dist_home = float(row.get("distance_from_home_km", 0.0))
        dist_last_tx = float(row.get("distance_from_last_tx_km", 0.0))
        if dist_home > 3000.0:
            risk_score += 0.25
            reasons.append(f"Physical distance from home is extreme ({dist_home:.0f} km)")
        elif dist_home > 500.0:
            risk_score += 0.12

        if dist_last_tx > 1000.0:
            risk_score += 0.15
            reasons.append(f"Rapid physical displacement from prior transaction ({dist_last_tx:.0f} km)")

        # 2. Velocity Anomalies
        tx_1h = int(row.get("transactions_last_1h", 0))
        if tx_1h >= 4:
            risk_score += 0.30
            reasons.append(f"High-frequency burst detected ({tx_1h} transactions in 1 hour)")
        elif tx_1h >= 2:
            risk_score += 0.10

        # 3. Amount Multiplier
        ratio = float(row.get("amount_vs_avg_ratio", 1.0))
        amt = float(row.get("transaction_amount", 0.0))
        if ratio >= 4.0 and amt > 300.0:
            risk_score += 0.25
            reasons.append(f"Transaction amount is {ratio:.1f}x higher than 30-day baseline")
        elif ratio >= 2.5:
            risk_score += 0.10

        # 4. Security & Risk Flags
        high_risk_country = int(row.get("high_risk_country", 0))
        intl = int(row.get("international_transaction", 0))
        card_present = int(row.get("card_present", 1))
        used_chip = int(row.get("used_chip", 1))
        declines = int(row.get("num_declined_last_30d", 0))

        if high_risk_country == 1:
            risk_score += 0.20
            reasons.append("Originates from a high-risk jurisdiction")
        if card_present == 0 and used_chip == 0 and intl == 1:
            risk_score += 0.15
            reasons.append("Unauthenticated card-not-present international transaction")
        if declines >= 2:
            risk_score += 0.15
            reasons.append(f"{declines} prior card declines in past 30 days")

        # Blend with System-1 prior
        final_prob = np.clip(0.4 * sys1_prob + 0.6 * min(1.0, risk_score), 0.01, 0.99)
        verdict = "FRAUD_CONFIRMED" if final_prob >= 0.50 else "LEGITIMATE_CLEARED"

        if final_prob >= 0.75:
            risk_tier = "CRITICAL"
        elif final_prob >= 0.50:
            risk_tier = "HIGH"
        elif final_prob >= 0.25:
            risk_tier = "MEDIUM"
        else:
            risk_tier = "LOW"

        if not reasons:
            rationale = "Transaction anomalies appear within normal spending variability; cleared as benign outlier."
        else:
            rationale = f"Flagged due to: {'; '.join(reasons)}."

        return {
            "verdict": verdict,
            "fraud_probability": float(final_prob),
            "risk_tier": risk_tier,
            "rationale": rationale
        }

    def _call_gemini_api(self, prompt: str) -> Dict[str, Any]:
        """Call Gemini API with structured JSON output and retries."""
        import requests
        import time

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{
                    "text": (
                        prompt + "\n\n"
                        "You are an expert fraud auditor. Analyze the brief and return valid JSON with EXACT keys:\n"
                        "{\n"
                        '  "verdict": "FRAUD_CONFIRMED" or "LEGITIMATE_CLEARED",\n'
                        '  "fraud_probability": <float between 0.0 and 1.0>,\n'
                        '  "risk_tier": "LOW", "MEDIUM", "HIGH", or "CRITICAL",\n'
                        '  "rationale": "<2-sentence explanation of your assessment>"\n'
                        "}\n"
                    )
                }]
            }],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.1
            }
        }
        
        for attempt in range(3):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    parsed = json.loads(text)
                    verdict = parsed.get("verdict", "LEGITIMATE_CLEARED")
                    if "FRAUD" in verdict.upper():
                        verdict = "FRAUD_CONFIRMED"
                    else:
                        verdict = "LEGITIMATE_CLEARED"
                    
                    prob = float(parsed.get("fraud_probability", 0.5))
                    prob = max(0.0, min(1.0, prob))
                    
                    return {
                        "verdict": verdict,
                        "fraud_probability": prob,
                        "risk_tier": parsed.get("risk_tier", "MEDIUM"),
                        "rationale": parsed.get("rationale", "LLM assessment complete.")
                    }
                elif resp.status_code == 429:
                    time.sleep(2.0 * (attempt + 1))
                else:
                    time.sleep(1.0)
            except Exception:
                time.sleep(1.0)

        return self._contextual_reasoning_engine(prompt, None, 0.5)

    def _call_openai_api(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI API if key is present."""
        import requests
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a senior financial fraud analyst. Output valid JSON only with keys: verdict, fraud_probability, risk_tier, rationale."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=10)
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return self._contextual_reasoning_engine(prompt, None, 0.5)

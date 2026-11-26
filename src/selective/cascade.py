"""
Uncertainty-Gated Cascade Orchestrator (System-1 to System-2).
Routes high-confidence in-distribution transactions through fast tabular inference,
while escalating ambiguous/high-epistemic-uncertainty transactions to System-2.
"""

from typing import Dict, Any, List, Optional
import time
import numpy as np
import pandas as pd
import torch

from src.system2.serializer import serialize_transaction_case
from src.system2.auditor import System2Auditor


class UncertaintyGatedCascade:
    """
    Two-Tier Cascade:
    - Tier-1: Fast Tabular Classifier with Temperature Scaling and MC Dropout
    - Dual Gate: Aleatoric Confidence + Epistemic Certainty
    - Tier-2: Contextual LLM Auditor
    """
    def __init__(
        self,
        sys1_model: Any,
        temperature: float = 1.0,
        tau_conf: float = 0.85,
        tau_sigma: float = 0.08,
        sys2_auditor: Optional[System2Auditor] = None
    ):
        self.sys1_model = sys1_model
        self.temperature = max(1e-4, temperature)
        self.tau_conf = tau_conf
        self.tau_sigma = tau_sigma
        self.sys2_auditor = sys2_auditor or System2Auditor()

    def process_stream(
        self,
        X: np.ndarray,
        df_records: pd.DataFrame,
        mc_samples: int = 30,
        device: Optional[torch.device] = None,
        max_sys2_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run the full cascade across an incoming batch of transactions.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        N = len(X)
        t0 = time.time()

        # ==========================================
        # STEP 1: System-1 Fast Inference & Uncertainty
        # ==========================================
        self.sys1_model.to(device)
        self.sys1_model.train()  # Keep dropout active for MC Dropout

        X_tensor = torch.from_numpy(X).float()
        batch_size = 1024

        all_sample_probs = []
        with torch.no_grad():
            for _ in range(mc_samples):
                sample_probs = []
                for i in range(0, N, batch_size):
                    batch_x = X_tensor[i:i + batch_size].to(device)
                    logits = self.sys1_model(batch_x)
                    # Apply temperature scaling to logits
                    scaled_logits = logits / self.temperature
                    probs = torch.sigmoid(scaled_logits).cpu().numpy().ravel()
                    sample_probs.extend(probs)
                all_sample_probs.append(sample_probs)

        all_sample_probs = np.array(all_sample_probs)  # Shape: (mc_samples, N)
        sys1_calibrated_probs = np.mean(all_sample_probs, axis=0)
        sys1_epistemic_std = np.std(all_sample_probs, axis=0)

        sys1_time = time.time() - t0

        # ==========================================
        # STEP 2: Dual Gate Evaluation
        # ==========================================
        confidence = np.maximum(sys1_calibrated_probs, 1.0 - sys1_calibrated_probs)

        # Gate conditions
        aleatoric_pass = (confidence >= self.tau_conf)
        epistemic_pass = (sys1_epistemic_std <= self.tau_sigma)

        automated_mask = aleatoric_pass & epistemic_pass
        escalated_mask = ~automated_mask

        n_automated = int(np.sum(automated_mask))
        n_escalated = int(np.sum(escalated_mask))
        coverage = n_automated / N

        # ==========================================
        # STEP 3: System-1 Automated Decisions
        # ==========================================
        final_probs = np.copy(sys1_calibrated_probs)
        final_preds = (sys1_calibrated_probs >= 0.50).astype(int)
        decision_tiers = np.array(["SYSTEM_1_AUTOMATED"] * N, dtype=object)
        rationales = np.array(["Cleared automatically by System-1 Dual Gate."] * N, dtype=object)

        # ==========================================
        # STEP 4: System-2 Escalation
        # ==========================================
        escalated_indices = np.where(escalated_mask)[0]
        if max_sys2_samples is not None and len(escalated_indices) > max_sys2_samples:
            # Subsample for evaluation efficiency if requested
            eval_escalated_indices = escalated_indices[:max_sys2_samples]
        else:
            eval_escalated_indices = escalated_indices

        t2_start = time.time()
        for idx in eval_escalated_indices:
            row = df_records.iloc[idx]
            
            # Identify specific gate trigger
            triggers = []
            if not aleatoric_pass[idx]:
                triggers.append(f"Low Aleatoric Confidence ({confidence[idx]:.2f} < {self.tau_conf:.2f})")
            if not epistemic_pass[idx]:
                triggers.append(f"High Epistemic Variance ({sys1_epistemic_std[idx]:.3f} > {self.tau_sigma:.3f})")
            trigger_str = " & ".join(triggers)

            brief = serialize_transaction_case(
                row=row,
                sys1_prob=sys1_calibrated_probs[idx],
                sys1_uncertainty=sys1_epistemic_std[idx],
                gate_reason=trigger_str
            )

            audit_result = self.sys2_auditor.audit_case(
                case_text=brief,
                raw_row=row.to_dict(),
                sys1_prob=sys1_calibrated_probs[idx]
            )

            final_probs[idx] = audit_result["fraud_probability"]
            final_preds[idx] = 1 if audit_result["verdict"] == "FRAUD_CONFIRMED" else 0
            decision_tiers[idx] = "SYSTEM_2_AUDITED"
            rationales[idx] = audit_result["rationale"]

        sys2_time = time.time() - t2_start

        return {
            "final_probs": final_probs,
            "final_preds": final_preds,
            "sys1_probs": sys1_calibrated_probs,
            "sys1_epistemic_std": sys1_epistemic_std,
            "confidence": confidence,
            "automated_mask": automated_mask,
            "escalated_mask": escalated_mask,
            "decision_tiers": decision_tiers,
            "rationales": rationales,
            "n_automated": n_automated,
            "n_escalated": n_escalated,
            "coverage": coverage,
            "sys1_time": sys1_time,
            "sys2_time": sys2_time,
            "total_time": sys1_time + sys2_time,
            "throughput_tps": N / (sys1_time + sys2_time)
        }

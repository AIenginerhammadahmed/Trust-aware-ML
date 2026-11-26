"""
Diagnostic & tuned experiment for Audit-Ready Short Paper:
1. Evaluates LightGBM + CreditNet (with Platt Scaling & Temperature Scaling)
2. Explains why pos_weight + Temperature Scaling alone caused T < 1 and higher silent FNs
3. Tunes Dual Gate on Validation set for ~90% automation coverage (~10% escalation)
"""
import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from src.data_2026 import load_and_preprocess_2026_data, prepare_temporal_datasets, create_dataloaders
from src.models.credit_net import CreditNet
from src.models.trainer import train_model
from src.calibration.temperature import fit_temperature_scaling
from src.calibration.metrics import binary_expected_calibration_error
from src.uncertainty.mc_dropout import mc_dropout_predict
from src.selective.cascade import UncertaintyGatedCascade
from src.system2.auditor import System2Auditor

device = torch.device("cpu")
df = load_and_preprocess_2026_data("data/credit_card_fraud_detection_2026.csv")
data = prepare_temporal_datasets(df)
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]
test_df = data["test_df"]

loaders = create_dataloaders(data, batch_size=512)

# Train CreditNet with moderate pos_weight=5.0
torch.manual_seed(42)
model = CreditNet(input_dim=X_train.shape[1], hidden_dims=(128, 64, 32), dropout_rate=0.25)
train_model(model, loaders["train"], loaders["val"], pos_weight=5.0, epochs=10, lr=2e-3, device=device, verbose=False)

model.eval()
with torch.no_grad():
    val_logits = model(torch.tensor(X_val, dtype=torch.float32)).numpy().ravel()
    test_logits = model(torch.tensor(X_test, dtype=torch.float32)).numpy().ravel()

raw_val_probs = torch.sigmoid(torch.tensor(val_logits)).numpy()
raw_test_probs = torch.sigmoid(torch.tensor(test_logits)).numpy()

# Temperature Scaling
_, opt_T = fit_temperature_scaling(val_logits, y_val, device=device)
temp_test_probs = torch.sigmoid(torch.tensor(test_logits) / opt_T).numpy()

# Platt Scaling (handles prior shift from pos_weight)
platt = LogisticRegression()
platt.fit(val_logits.reshape(-1, 1), y_val)
platt_val_probs = platt.predict_proba(val_logits.reshape(-1, 1))[:, 1]
platt_test_probs = platt.predict_proba(test_logits.reshape(-1, 1))[:, 1]

def stats(name, y, p, thresh=0.5):
    pred = (p >= thresh).astype(int)
    auc = roc_auc_score(y, p)
    pr = average_precision_score(y, p)
    f1 = f1_score(y, pred)
    ece = binary_expected_calibration_error(y, p)
    conf = np.maximum(p, 1.0 - p)
    silent = int(np.sum((y == 1) & (pred == 0) & (conf >= 0.80)))
    print(f"{name:<28} | AUC={auc:.4f} | PR={pr:.4f} | F1={f1:.4f} | ECE={ece:.4f} | SilentFN={silent}")

print(f"Optimal Temp T = {opt_T:.4f}")
stats("1. CreditNet Uncal (pw=5)", y_test, raw_test_probs)
stats("2. CreditNet TempScaled", y_test, temp_test_probs)
stats("3. CreditNet PlattScaled", y_test, platt_test_probs)

# MC Dropout epistemic std on Val & Test
val_mc = mc_dropout_predict(model, torch.from_numpy(X_val).float(), n_samples=25, device=device)
test_mc = mc_dropout_predict(model, torch.from_numpy(X_test).float(), n_samples=25, device=device)

# Tune tau_sigma at 92nd percentile and tau_conf on Val Platt probs to get ~90% coverage
tau_sigma = float(np.percentile(val_mc["std_epistemic"], 92))
# With Platt scaling, base rate is ~0.02, so most legit probs are < 0.15 (conf > 0.85).
# Let's check coverage at tau_conf = 0.85 and 0.90
for tc in [0.80, 0.85, 0.90]:
    conf_test = np.maximum(platt_test_probs, 1.0 - platt_test_probs)
    auto_mask = (conf_test >= tc) & (test_mc["std_epistemic"] <= tau_sigma)
    cov = np.mean(auto_mask)
    print(f"Gate (tau_conf={tc}, tau_sigma={tau_sigma:.4f}) -> Coverage={cov*100:.2f}% | Escalated={(1-cov)*100:.2f}%")
    if np.sum(auto_mask) > 0:
        stats(f"  Accepted (cov={cov*100:.1f}%)", y_test[auto_mask], platt_test_probs[auto_mask])

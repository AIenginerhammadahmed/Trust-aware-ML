"""
Independent verification of every number in the short paper.
Re-runs the audit pipeline (same seeds/config as run_audit_fixed_benchmark.py) and adds:
 - reproducibility check vs saved JSON
 - true single-transaction latency (batch=1) vs batch-amortized latency
 - bootstrap 95% CIs for ROC-AUC / PR-AUC / F1 differences
 - decomposition of the cascade's silent-FN reduction
 - pos_weight actually used
"""
import os, sys, time, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, torch, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from src.data_2026 import load_and_preprocess_2026_data, prepare_temporal_datasets, create_dataloaders
from src.models.credit_net import CreditNet
from src.models.trainer import train_model
from src.uncertainty.mc_dropout import mc_dropout_predict
from src.system2.serializer import serialize_transaction_case
from src.system2.auditor import System2Auditor

saved = json.load(open("experiments/results_2026/audit_benchmark_summary.json"))
df = load_and_preprocess_2026_data(); d = prepare_temporal_datasets(df)
Xtr, ytr, Xv, yv, Xte, yte = d["X_train"], d["y_train"], d["X_val"], d["y_val"], d["X_test"], d["y_test"]
L = create_dataloaders(d, batch_size=512)

pw_full = float(np.sum(ytr == 0) / np.sum(ytr == 1)); pw_used = pw_full * 0.5
print(f"pos_weight full ratio={pw_full:.2f} | actually used for CreditNet={pw_used:.2f} | LightGBM: none")

torch.manual_seed(42)
mlp = CreditNet(Xtr.shape[1], (128, 64, 32), 0.25)
train_model(mlp, L["train"], L["val"], pos_weight=pw_used, epochs=10, lr=2e-3, device=torch.device("cpu"), verbose=False)
clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1).fit(Xtr, ytr)
pv, pt = clf.predict_proba(Xv)[:, 1], clf.predict_proba(Xte)[:, 1]
vmc = mc_dropout_predict(mlp, torch.from_numpy(Xv).float(), n_samples=20)
tmc = mc_dropout_predict(mlp, torch.from_numpy(Xte).float(), n_samples=20)

th = saved["opt_thresh"]; band = saved["ambiguity_band"]
tau = float(np.percentile(vmc["std_epistemic"], 94))
auto = (np.abs(pt - th) >= band) & (tmc["std_epistemic"] <= tau)
print(f"[repro] tau_sigma rerun={tau:.4f} saved={saved['tau_sigma']:.4f} | coverage rerun={auto.mean():.4f} saved={saved['coverage']:.4f}")
print(f"[repro] LightGBM AUC rerun={roc_auc_score(yte, pt):.4f} saved={saved['models'][2]['roc_auc']:.4f}")

# cascade
aud = System2Auditor(); cp = pt.copy(); esc = np.where(~auto)[0]
for i in esc:
    r = d["test_df"].iloc[i]
    res = aud.audit_case(serialize_transaction_case(r, float(pt[i]), float(tmc["std_epistemic"][i]), "x"), r.to_dict(), float(pt[i]))
    cp[i] = 0.65 * pt[i] + 0.35 * res["fraud_probability"]
print(f"[repro] Cascade AUC rerun={roc_auc_score(yte, cp):.4f} saved={saved['models'][4]['roc_auc']:.4f}")

# silent FN decomposition
def silent(p): pred = p >= th; return (yte == 1) & (~pred) & (np.maximum(p, 1 - p) >= 0.8)
s_l, s_c = silent(pt), silent(cp)
print(f"[silentFN] LightGBM={s_l.sum()} Cascade={s_c.sum()} | LightGBM silent FNs inside escalated set={s_l[~auto].sum()}")
fixed = s_l & ~s_c
print(f"[silentFN] removed={fixed.sum()} -> of these, now correctly flagged (p>=th)={np.sum(fixed & (cp >= th))}, merely moved to low-confidence but still missed={np.sum(fixed & (cp < th))}")
print(f"[esc] frauds in escalated set={int(yte[~auto].sum())}/{int(yte.sum())} | LightGBM recall on esc={np.mean(pt[~auto][yte[~auto]==1]>=th):.3f} cascade recall on esc={np.mean(cp[~auto][yte[~auto]==1]>=th):.3f}")
print(f"[esc] FPs on escalated set: LightGBM={np.sum((pt[~auto]>=th)&(yte[~auto]==0))} cascade={np.sum((cp[~auto]>=th)&(yte[~auto]==0))}")

# Bootstrap CIs (paired)
rng = np.random.default_rng(0); B = 500; dA, dP, dF = [], [], []
for _ in range(B):
    ix = rng.integers(0, len(yte), len(yte)); y = yte[ix]
    dA.append(roc_auc_score(y, cp[ix]) - roc_auc_score(y, pt[ix]))
    dP.append(average_precision_score(y, cp[ix]) - average_precision_score(y, pt[ix]))
    dF.append(f1_score(y, cp[ix] >= th) - f1_score(y, pt[ix] >= th))
for n, v in [("dROC (cascade-LGB)", dA), ("dPR", dP), ("dF1", dF)]:
    print(f"[CI] {n}: mean={np.mean(v):+.4f} 95%CI=[{np.percentile(v,2.5):+.4f}, {np.percentile(v,97.5):+.4f}]")
aucs = [roc_auc_score(yte[ix], pt[ix]) for ix in (rng.integers(0, len(yte), len(yte)) for _ in range(300))]
print(f"[CI] LightGBM ROC-AUC 95%CI=[{np.percentile(aucs,2.5):.4f}, {np.percentile(aucs,97.5):.4f}]")

# Latency: batch-1 single transaction
x1 = Xte[:1]; xt1 = torch.from_numpy(x1).float()
t = time.perf_counter()
for _ in range(200): clf.predict_proba(x1)
lgb1 = (time.perf_counter() - t) / 200 * 1000
mlp.train(); t = time.perf_counter()
with torch.no_grad():
    for _ in range(50):
        for _ in range(20): mlp(xt1)
mc1 = (time.perf_counter() - t) / 50 * 1000
t = time.perf_counter(); clf.predict_proba(Xte); lgbB = (time.perf_counter() - t) / len(Xte) * 1000
print(f"[latency] LightGBM batch-amortized={lgbB:.4f} ms/tx | single-tx={lgb1:.3f} ms")
print(f"[latency] MC Dropout 20 passes single-tx={mc1:.3f} ms | System-1 single-tx total ~{lgb1+mc1:.3f} ms")

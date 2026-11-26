"""
Model training and evaluation routines with class-imbalance weighting and validation tracking.
"""

from typing import Dict, Any, Optional
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pos_weight: Optional[float] = None,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train CreditNet using BCEWithLogitsLoss with positive class weighting.
    Tracks validation metrics across epochs and restores best model weights.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    # Class-weighted binary cross-entropy
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device, dtype=torch.float32))
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
        "val_pr_auc": []
    }

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss_total = 0.0
        train_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_total / max(1, train_batches)

        # Validation
        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss_total += loss.item()
                val_batches += 1

                probs = torch.sigmoid(logits).cpu().numpy().ravel()
                val_preds.extend(probs)
                val_targets.extend(y_batch.cpu().numpy().ravel())

        avg_val_loss = val_loss_total / max(1, val_batches)
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)

        val_auc = roc_auc_score(val_targets, val_preds) if len(np.unique(val_targets)) > 1 else 0.5
        val_pr_auc = average_precision_score(val_targets, val_preds) if len(np.unique(val_targets)) > 1 else 0.0

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_auc"].append(val_auc)
        history["val_pr_auc"].append(val_pr_auc)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_weights = copy.deepcopy(model.state_dict())

        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
            print(f"Epoch [{epoch:02d}/{epochs:02d}] "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val ROC-AUC: {val_auc:.4f} | "
                  f"Val PR-AUC: {val_pr_auc:.4f}")

    # Restore best checkpoint
    model.load_state_dict(best_weights)

    return {
        "model": model,
        "history": history,
        "best_val_loss": best_val_loss
    }


def evaluate_predictions(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compute standard classification and ranking metrics."""
    preds = (probs >= threshold).astype(int)
    auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = average_precision_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.0
    f1 = f1_score(y_true, preds, zero_division=0)
    acc = (preds == y_true).mean()

    return {
        "roc_auc": float(auc),
        "pr_auc": float(pr_auc),
        "f1_score": float(f1),
        "accuracy": float(acc)
    }

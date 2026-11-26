"""
Data pipeline for the 2026 / 2024 Full-Year Credit Card Fraud Dataset.
Features: 100,000 transactions with interpretable behavioral, merchant, and device attributes.
Includes chronological temporal splitting (Train: Jan-Aug, Val/Calib: Sep-Oct, Test/Drift: Nov-Dec).
"""

from typing import Dict, List, Tuple, Optional, Any
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader


NUMERICAL_COLS = [
    "transaction_amount",
    "transaction_hour",
    "customer_age",
    "account_age_days",
    "credit_limit",
    "credit_score",
    "num_cards_held",
    "transactions_last_1h",
    "transactions_last_24h",
    "transactions_last_7d",
    "amount_last_1h",
    "amount_last_24h",
    "days_since_last_transaction",
    "distance_from_home_km",
    "distance_from_last_tx_km",
    "amount_vs_avg_ratio",
    "avg_transaction_amount_30d",
    "std_transaction_amount_30d",
    "num_declined_last_30d",
    "num_chargebacks_ever",
]

CATEGORICAL_COLS = [
    "merchant_category",
    "merchant_country",
    "customer_gender",
    "customer_state",
    "transaction_channel",
    "device_type",
    "browser",
    "email_domain",
    "transaction_day_of_week",
]

BINARY_COLS = [
    "card_present",
    "international_transaction",
    "high_risk_country",
    "used_chip",
    "used_pin",
    "is_weekend",
]

TARGET_COL = "is_fraud"


def load_and_preprocess_2026_data(
    filepath: str = "data/credit_card_fraud_detection_2026.csv",
) -> pd.DataFrame:
    """Load dataset, handle missing values, and sort chronologically."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    df = pd.read_csv(filepath)

    # Convert transaction_datetime to datetime object
    df["transaction_datetime"] = pd.to_datetime(df["transaction_datetime"])
    df = df.sort_values("transaction_datetime").reset_index(drop=True)

    # Missing values: 'browser' is missing for physical POS/ATM transactions
    df["browser"] = df["browser"].fillna("None/In-Store")

    # Clean any unexpected NaNs
    for col in NUMERICAL_COLS:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_COLS:
        if df[col].isnull().any():
            df[col] = df[col].fillna("Unknown")

    return df


def prepare_temporal_datasets(
    df: pd.DataFrame,
    val_months: Tuple[int, int] = (9, 10),      # Sep - Oct
    test_months: Tuple[int, int] = (11, 12),    # Nov - Dec (Holiday concept drift)
) -> Dict[str, Any]:
    """
    Split data chronologically to evaluate resilience against temporal concept drift.
    - Train: Months 1-8 (Jan-Aug)
    - Val / Calib: Months 9-10 (Sep-Oct)
    - Test: Months 11-12 (Nov-Dec)
    """
    months = df["transaction_datetime"].dt.month

    train_mask = (months < val_months[0])
    val_mask = (months >= val_months[0]) & (months <= val_months[1])
    test_mask = (months >= test_months[0]) & (months <= test_months[1])

    train_df = df[train_mask].copy().reset_index(drop=True)
    val_df = df[val_mask].copy().reset_index(drop=True)
    test_df = df[test_mask].copy().reset_index(drop=True)

    # Fit Encoders on Train only (strictly prevent data leakage)
    cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_cat = cat_encoder.fit_transform(train_df[CATEGORICAL_COLS])
    val_cat = cat_encoder.transform(val_df[CATEGORICAL_COLS])
    test_cat = cat_encoder.transform(test_df[CATEGORICAL_COLS])

    # Standardize numerical features on Train only
    scaler = StandardScaler()
    train_num = scaler.fit_transform(train_df[NUMERICAL_COLS])
    val_num = scaler.transform(val_df[NUMERICAL_COLS])
    test_num = scaler.transform(test_df[NUMERICAL_COLS])

    # Binary flags
    train_bin = train_df[BINARY_COLS].values.astype(np.float32)
    val_bin = val_df[BINARY_COLS].values.astype(np.float32)
    test_bin = test_df[BINARY_COLS].values.astype(np.float32)

    # Combined feature matrices
    X_train = np.hstack([train_num, train_cat, train_bin]).astype(np.float32)
    X_val = np.hstack([val_num, val_cat, val_bin]).astype(np.float32)
    X_test = np.hstack([test_num, test_cat, test_bin]).astype(np.float32)

    y_train = train_df[TARGET_COL].values.astype(np.int64)
    y_val = val_df[TARGET_COL].values.astype(np.int64)
    y_test = test_df[TARGET_COL].values.astype(np.int64)

    feature_names = NUMERICAL_COLS + CATEGORICAL_COLS + BINARY_COLS

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "scaler": scaler,
        "cat_encoder": cat_encoder,
        "feature_names": feature_names,
        "num_cols": NUMERICAL_COLS,
        "cat_cols": CATEGORICAL_COLS,
        "bin_cols": BINARY_COLS,
    }


def create_dataloaders(
    data: Dict[str, Any],
    batch_size: int = 512,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """Convert numpy matrices to PyTorch DataLoaders."""
    train_dataset = TensorDataset(
        torch.from_numpy(data["X_train"]),
        torch.from_numpy(data["y_train"]).float().unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(data["X_val"]),
        torch.from_numpy(data["y_val"]).float().unsqueeze(1)
    )
    test_dataset = TensorDataset(
        torch.from_numpy(data["X_test"]),
        torch.from_numpy(data["y_test"]).float().unsqueeze(1)
    )

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }

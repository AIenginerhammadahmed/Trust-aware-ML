"""
Data loading, preprocessing, and splitting utilities for credit card fraud detection.
Supports both Stratified 3-way split (Train / Val / Test) and Temporal Split.
"""

from typing import Tuple, Optional, Dict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_fraud_data(
    filepath: str,
    drop_na: bool = True
) -> pd.DataFrame:
    """Load credit card fraud dataset from CSV."""
    df = pd.read_csv(filepath)
    if drop_na:
        df = df.dropna().reset_index(drop=True)
    return df


def prepare_datasets(
    df: pd.DataFrame,
    split_strategy: str = "stratified",
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
    include_time: bool = True
) -> Dict[str, np.ndarray]:
    """
    Split and scale features for train, validation, and test sets.

    Args:
        df: DataFrame containing features and binary 'Class' column.
        split_strategy: 'stratified' (random stratified) or 'temporal' (ordered by Time).
        val_size: Fraction of total data allocated for validation.
        test_size: Fraction of total data allocated for testing.
        random_state: Seed for reproducibility.
        include_time: Whether to include the 'Time' feature.

    Returns:
        Dictionary containing X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_names.
    """
    feature_cols = [c for c in df.columns if c != "Class"]
    if not include_time and "Time" in feature_cols:
        feature_cols.remove("Time")

    X = df[feature_cols].values
    y = df["Class"].values.astype(int)

    if split_strategy == "temporal":
        # Sort chronologically by Time
        if "Time" in df.columns:
            sort_idx = np.argsort(df["Time"].values)
            X = X[sort_idx]
            y = y[sort_idx]

        n_total = len(df)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)
        n_train = n_total - n_val - n_test

        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
        X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    elif split_strategy == "stratified":
        # 3-way stratified split: Train / Val / Test
        # First split into train_val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )

        # Then split train_val into train and val
        val_relative = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_relative,
            stratify=y_train_val,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")

    # Standardize features using Train statistics only (prevents data leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": feature_cols
    }


def create_dataloaders(
    data_dict: Dict[str, np.ndarray],
    batch_size: int = 512,
    device: Optional[torch.device] = None
) -> Dict[str, DataLoader]:
    """Convert numpy arrays into PyTorch DataLoaders."""
    loaders = {}
    for split in ["train", "val", "test"]:
        X_tensor = torch.tensor(data_dict[f"X_{split}"], dtype=torch.float32)
        y_tensor = torch.tensor(data_dict[f"y_{split}"], dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(X_tensor, y_tensor)
        shuffle = (split == "train")
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loaders

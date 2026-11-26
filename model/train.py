"""
Backwards-compatible wrapper for training routines.
Core implementation is maintained in src.models.trainer.
"""

from src.models.trainer import train_model, evaluate_predictions

__all__ = ["train_model", "evaluate_predictions"]

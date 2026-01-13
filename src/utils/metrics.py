"""
Metrics calculation utilities
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    
    # Additional metrics
    mean_error = np.mean(y_pred - y_true)
    std_error = np.std(y_pred - y_true)
    
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
        "Mean Error": mean_error,
        "Std Error": std_error,
    }


def print_metrics(metrics: Dict[str, float], model_name: str = "") -> None:
    """
    Print metrics in a formatted way
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
    """
    print(f"\n{'='*60}")
    if model_name:
        print(f"Metrics for {model_name}")
    print(f"{'='*60}")
    for metric, value in metrics.items():
        if metric in ["MSE", "RMSE", "MAE"]:
            print(f"{metric:20s}: {value:>15.2f}")
        elif metric == "MAPE":
            print(f"{metric:20s}: {value:>15.2f}%")
        elif metric == "R2":
            print(f"{metric:20s}: {value:>15.4f}")
        else:
            print(f"{metric:20s}: {value:>15.4f}")
    print(f"{'='*60}\n")

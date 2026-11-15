"""
nn/baseline.py - Naive forecasting baselines for comparison:
1. "Same hour yesterday" baseline: hat_L(d, h) = L(d-1, h)
2. "Same hour last week" baseline: hat_L(d, h) = L(d-7, h)
"""

import numpy as np
import pandas as pd

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes MAE, RMSE, and MAPE metrics.
    y_true: shape (N, 24)
    y_pred: shape (N, 24)
    """
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    # Avoid zero division in MAPE
    eps = 1e-4
    mape = float(np.mean(np.abs(y_true - y_pred) / np.maximum(y_true, eps)) * 100.0)
    
    return {
        "MAE_kW": round(mae, 4),
        "RMSE_kW": round(rmse, 4),
        "MAPE_percent": round(mape, 2)
    }

def evaluate_baselines(df: pd.DataFrame) -> dict:
    """
    Evaluates both yesterday and last week baselines on a dataframe containing:
    target_h0..h23, yest_h0..h23, lastweek_h0..h23
    """
    target_cols = [f'target_h{h}' for h in range(24)]
    yest_cols = [f'yest_h{h}' for h in range(24)]
    lastweek_cols = [f'lastweek_h{h}' for h in range(24)]
    
    y_true = df[target_cols].values
    y_yest = df[yest_cols].values
    y_lastweek = df[lastweek_cols].values
    
    metrics_yest = compute_metrics(y_true, y_yest)
    metrics_lastweek = compute_metrics(y_true, y_lastweek)
    
    return {
        "Same_Hour_Yesterday": metrics_yest,
        "Same_Hour_Last_Week": metrics_lastweek
    }

"""
nn/evaluate.py - Prediction interface and evaluation utilities for the trained neural network.
"""

import os
import json
import numpy as np
import pandas as pd
import torch

from nn.model import LoadForecastMLP

class LoadForecaster:
    def __init__(self, model_dir: str = "nn"):
        self.model_dir = model_dir
        self.weights_path = os.path.join(model_dir, "best_model.pt")
        self.stats_path = os.path.join(model_dir, "norm_stats.json")
        
        with open(self.stats_path, "r") as f:
            stats = json.load(f)
            
        self.mean = np.array(stats["mean"], dtype=np.float32)
        self.std = np.array(stats["std"], dtype=np.float32)
        self.feature_cols = stats["feature_cols"]
        self.target_cols = stats["target_cols"]
        
        self.model = LoadForecastMLP(input_dim=len(self.feature_cols), output_dim=len(self.target_cols))
        self.model.load_state_dict(torch.load(self.weights_path, weights_only=True))
        self.model.eval()
        
    def predict(self, input_features: np.ndarray) -> np.ndarray:
        """
        Predicts 24-hour load given input features of shape (N, 53) or (53,).
        Returns predicted load in kW of shape (N, 24) or (24,).
        """
        is_1d = (input_features.ndim == 1)
        if is_1d:
            x = input_features.reshape(1, -1)
        else:
            x = input_features
            
        x_norm = (x - self.mean) / self.std
        with torch.no_grad():
            tensor_x = torch.from_numpy(x_norm.astype(np.float32))
            preds = self.model(tensor_x).numpy()
            
        if is_1d:
            return preds[0]
        return preds

    def predict_row(self, row: pd.Series) -> np.ndarray:
        feats = row[self.feature_cols].values.astype(np.float32)
        return self.predict(feats)

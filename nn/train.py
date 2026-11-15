"""
nn/train.py - Reproducible training script for the neural network load forecaster.
Compares performance on the final year test set against:
1. "Same hour yesterday"
2. "Same hour last week"
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from nn.model import LoadForecastMLP
from nn.baseline import compute_metrics, evaluate_baselines

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_features(df: pd.DataFrame, feature_cols: list, target_cols: list):
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)
    return X, y

def train_forecaster(data_dir: str = "data/processed",
                     save_dir: str = "nn",
                     epochs: int = 60,
                     batch_size: int = 32,
                     lr: float = 1e-3,
                     seed: int = 42):
    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Load data
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    calendar_cols = ['is_weekend', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos']
    yest_cols = [f'yest_h{h}' for h in range(24)]
    lastweek_cols = [f'lastweek_h{h}' for h in range(24)]
    target_cols = [f'target_h{h}' for h in range(24)]
    
    feature_cols = yest_cols + lastweek_cols + calendar_cols
    input_dim = len(feature_cols)
    output_dim = len(target_cols)
    
    print(f"Features: {input_dim} (24 yesterday + 24 last week + {len(calendar_cols)} calendar)")
    print(f"Target: {output_dim} hourly loads")
    
    X_train, y_train = prepare_features(train_df, feature_cols, target_cols)
    X_val, y_val = prepare_features(val_df, feature_cols, target_cols)
    X_test, y_test = prepare_features(test_df, feature_cols, target_cols)
    
    # Standardize features based on training set
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-6
    
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    X_test_norm = (X_test - mean) / std
    
    # Save normalization stats
    norm_stats = {
        "mean": mean.tolist()[0],
        "std": std.tolist()[0],
        "feature_cols": feature_cols,
        "target_cols": target_cols
    }
    with open(os.path.join(save_dir, "norm_stats.json"), "w") as f:
        json.dump(norm_stats, f)
        
    train_dataset = TensorDataset(torch.from_numpy(X_train_norm), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val_norm), torch.from_numpy(y_val))
    test_dataset = TensorDataset(torch.from_numpy(X_test_norm), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize Model, Loss, Optimizer
    model = LoadForecastMLP(input_dim=input_dim, output_dim=output_dim, hidden_dim=128, dropout_rate=0.1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    best_weights_path = os.path.join(save_dir, "best_model.pt")
    
    print("\n--- Starting Neural Network Training ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_x = torch.from_numpy(X_val_norm)
            val_y = torch.from_numpy(y_val)
            val_pred = model(val_x)
            val_loss = criterion(val_pred, val_y).item()
            
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_weights_path)
            mark = "*"
        else:
            mark = ""
            
        if epoch % 10 == 0 or mark == "*":
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss (MSE): {train_loss:.4f} | Val Loss (MSE): {val_loss:.4f} {mark}")
            
    # Load Best Model for Test Evaluation
    print("\nLoading best model checkpoint for evaluation on test set...")
    model.load_state_dict(torch.load(best_weights_path, weights_only=True))
    model.eval()
    
    with torch.no_grad():
        test_x = torch.from_numpy(X_test_norm)
        test_pred = model(test_x).numpy()
        
    nn_metrics = compute_metrics(y_test, test_pred)
    baseline_metrics = evaluate_baselines(test_df)
    
    results = {
        "Neural_Network_MLP": nn_metrics,
        "Same_Hour_Yesterday": baseline_metrics["Same_Hour_Yesterday"],
        "Same_Hour_Last_Week": baseline_metrics["Same_Hour_Last_Week"]
    }
    
    results_path = os.path.join(save_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n================== TEST SET EVALUATION RESULTS ==================")
    print(f"{'Model / Baseline':<25} | {'MAE (kW)':<10} | {'RMSE (kW)':<10} | {'MAPE (%)':<10}")
    print("-" * 65)
    for name, m in results.items():
        print(f"{name:<25} | {m['MAE_kW']:<10.4f} | {m['RMSE_kW']:<10.4f} | {m['MAPE_percent']:<10.2f}%")
    print("=" * 65)
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    train_forecaster(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)

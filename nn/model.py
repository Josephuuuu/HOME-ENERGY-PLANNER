"""
nn/model.py - PyTorch neural network for 24-hour day-ahead household load forecasting.
"""

import torch
import torch.nn as nn

class LoadForecastMLP(nn.Module):
    """
    Multi-layer Perceptron with residual connections and layer normalization
    for predicting 24-hour load profiles.
    Input features: 24 yesterday loads + 24 last week loads + 5 calendar features = 53 features.
    Output: 24 hourly load predictions (kW).
    """
    def __init__(self, input_dim: int = 53, output_dim: int = 24, hidden_dim: int = 128, dropout_rate: float = 0.1):
        super(LoadForecastMLP, self).__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.relu1 = nn.ReLU()
        
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.head = nn.Linear(64, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = self.relu1(h + self.block1(h))  # Residual connection
        h = self.block2(h)
        out = self.head(h)
        # Power load is non-negative
        return torch.relu(out)

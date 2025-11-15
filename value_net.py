"""
Value Network for OFC.
Learns to estimate expected final score from a given state.
OPTIMIZED for H200: Larger network (512 hidden units) with dropout for regularization.
"""
import torch
import torch.nn as nn


class ValueNet(nn.Module):
    """Neural network that estimates the value (expected final score) of a state."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        """
        Args:
            input_dim: Size of encoded state vector
            hidden_dim: Size of hidden layers (increased to 512 for better capacity)
            dropout: Dropout rate for regularization (0.1 = 10% dropout)
        """
        super().__init__()
        
        # Build network layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Batch of encoded states, shape (batch_size, input_dim)
        
        Returns:
            Value estimates, shape (batch_size, 1)
        """
        return self.net(x)


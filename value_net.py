"""
Value Network for OFC.
Learns to estimate expected final score from a given state.
"""
import torch
import torch.nn as nn


class ValueNet(nn.Module):
    """Neural network that estimates the value (expected final score) of a state."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        """
        Args:
            input_dim: Size of encoded state vector
            hidden_dim: Size of hidden layers
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Batch of encoded states, shape (batch_size, input_dim)
        
        Returns:
            Value estimates, shape (batch_size, 1)
        """
        return self.net(x)


"""
Value Head for PPO Critic

Lightweight neural network that estimates state values for advantage computation.
"""
import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """Lightweight value head for critic."""
    
    def __init__(self, input_dim: int = 4096, hidden_dim: int = 1024):
        """
        Initialize value head network.
        
        Args:
            input_dim: Dimension of input features (typically VLA hidden states)
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict state value.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            State values [batch_size]
        """
        return self.net(x).squeeze(-1)

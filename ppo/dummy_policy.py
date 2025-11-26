"""
Dummy policy for testing and demonstration.

Replace with actual VLA policy for real training.
"""

import numpy as np
from typing import Tuple


class DummyPolicy:
    """
    Dummy policy for demonstration (replace with actual VLA policy).
    
    This is a placeholder that returns random actions for testing the
    training infrastructure. In production, replace with an actual
    vision-language-action model.
    """
    
    def __init__(self, action_dim: int = 7):
        """
        Initialize dummy policy.
        
        Args:
            action_dim: Dimensionality of action space
        """
        self.action_dim = action_dim
    
    def get_action(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Get action, value, and log_prob for an observation.
        
        Args:
            obs: Observation from environment
            
        Returns:
            action: Random action in [-1, 1]^action_dim
            value: Dummy value estimate (always 0)
            log_prob: Dummy log probability (always 0)
        """
        action = np.random.uniform(-1, 1, self.action_dim).astype(np.float32)
        value = 0.0  # Dummy value
        log_prob = 0.0  # Dummy log prob
        return action, value, log_prob
    
    def get_value(self, obs: np.ndarray) -> float:
        """
        Get value estimate for an observation.
        
        Args:
            obs: Observation from environment
            
        Returns:
            Dummy value estimate (always 0)
        """
        return 0.0

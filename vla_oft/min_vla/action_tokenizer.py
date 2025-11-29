"""
action_tokenizer.py

Action discretization utilities for PPO training with OpenVLA.
Discretizes continuous robot actions into 256 bins and maps to vocabulary tokens.
"""

from typing import Union
import numpy as np
import torch


class ActionTokenizer:
    """
    Discretizes continuous robot actions into N bins per dimension and maps to vocabulary tokens.
    
    Maps actions to the last 256 tokens of the vocabulary (tokens 31744-32000 for vocab_size=32000).
    Uses uniform binning with bin centers for detokenization.
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        n_bins: int = 256,
        min_action: float = -1.0,
        max_action: float = 1.0,
    ):
        """
        Initialize action tokenizer.
        
        Args:
            vocab_size: Size of the vocabulary (default: 32000 for LLaMA)
            n_bins: Number of bins for discretization (default: 256)
            min_action: Minimum action value for clipping (default: -1.0)
            max_action: Maximum action value for clipping (default: 1.0)
        """
        self.vocab_size = vocab_size
        self.n_bins = n_bins
        self.min_action = min_action
        self.max_action = max_action
        
        # Create uniform bins
        self.bins = np.linspace(min_action, max_action, n_bins)
        
        # Compute bin centers (255 centers for 256 bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        
        # Action tokens occupy last n_bins positions in vocabulary
        self.action_token_begin_idx = vocab_size - n_bins
        self.action_token_end_idx = vocab_size
    
    def discretize_actions(self, actions: np.ndarray) -> np.ndarray:
        """
        Discretize continuous actions to token IDs.
        
        Args:
            actions: Continuous actions, shape (..., action_dim)
        
        Returns:
            token_ids: Discretized action token IDs, shape (..., action_dim)
        """
        # Clip actions to valid range
        actions = np.clip(actions, a_min=self.min_action, a_max=self.max_action)
        
        # Digitize to bin indices [1, n_bins]
        discretized = np.digitize(actions, self.bins)
        
        # Map to vocabulary token IDs: vocab_size - discretized
        token_ids = self.vocab_size - discretized
        
        return token_ids
    
    def detokenize_actions(self, token_ids: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Convert token IDs back to continuous actions using bin centers.
        
        Args:
            token_ids: Action token IDs, shape (..., action_dim)
        
        Returns:
            actions: Continuous actions, shape (..., action_dim)
        """
        # Convert to numpy if torch tensor
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().numpy()
        
        # Map token IDs back to bin indices
        discretized = self.vocab_size - token_ids
        
        # Clip to valid bin center indices [0, n_bins-2]
        # (we have n_bins-1 bin centers for n_bins bins)
        discretized = np.clip(discretized - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        
        # Look up bin centers
        actions = self.bin_centers[discretized]
        
        return actions
    
    def __repr__(self) -> str:
        return (
            f"ActionTokenizer(vocab_size={self.vocab_size}, n_bins={self.n_bins}, "
            f"range=[{self.min_action}, {self.max_action}], "
            f"tokens=[{self.action_token_begin_idx}, {self.action_token_end_idx}])"
        )

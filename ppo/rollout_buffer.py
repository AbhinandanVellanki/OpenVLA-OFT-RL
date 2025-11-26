"""
Rollout buffer for storing and processing trajectory data.

Supports GAE computation for advantage estimation.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data with computed advantages."""
    observations: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]
    dones: List[bool]
    values: List[float]
    log_probs: List[float]
    advantages: Optional[np.ndarray] = None
    returns: Optional[np.ndarray] = None
    
    def __init__(self):
        self.clear()
    
    def clear(self):
        """Clear all stored data."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.advantages = None
        self.returns = None
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ):
        """
        Add a transition to the buffer.
        
        Args:
            obs: Observation
            action: Action taken
            reward: Reward received
            done: Episode termination flag
            value: Value estimate
            log_prob: Log probability of action
        """
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
    
    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            last_value: Value estimate of the state after the last transition
            gamma: Discount factor
            gae_lambda: Lambda parameter for GAE
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]
            
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        
        self.advantages = advantages
        self.returns = advantages + values
    
    def get(self) -> Dict[str, np.ndarray]:
        """
        Get all data from the buffer as numpy arrays.
        
        Returns:
            Dictionary containing all buffer data
        """
        return {
            "observations": np.array(self.observations),
            "actions": np.array(self.actions),
            "rewards": np.array(self.rewards),
            "dones": np.array(self.dones),
            "values": np.array(self.values),
            "log_probs": np.array(self.log_probs),
            "advantages": self.advantages if self.advantages is not None else np.zeros(len(self.rewards)),
            "returns": self.returns if self.returns is not None else np.zeros(len(self.rewards)),
        }

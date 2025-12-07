"""
Generalized Advantage Estimation (GAE) for PPO.

Implements the GAE algorithm from:
"High-Dimensional Continuous Control Using Generalized Advantage Estimation"
by Schulman et al. (2015)
"""

import numpy as np
import torch
from typing import Tuple, Optional


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    normalize_advantages: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE) for a trajectory.

    GAE computes advantages using temporal difference (TD) residuals:
        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}

    Returns are computed as:
        returns_t = advantages_t + values_t

    This provides a lower-variance advantage estimate compared to Monte Carlo returns
    while maintaining lower bias than n-step returns.

    Args:
        rewards: Rewards array of shape (trajectory_length,)
        values: Value estimates array of shape (trajectory_length,)
        dones: Episode termination flags of shape (trajectory_length,)
        gamma: Discount factor for future rewards (default: 0.99)
        gae_lambda: GAE lambda parameter for bias-variance tradeoff (default: 0.95)
                   lambda=0: high bias, low variance (TD(0))
                   lambda=1: low bias, high variance (Monte Carlo)
        normalize_advantages: Whether to normalize advantages to mean=0, std=1

    Returns:
        advantages: GAE advantages of shape (trajectory_length,)
        returns: TD targets for value function training of shape (trajectory_length,)

    Example:
        >>> rewards = np.array([0.0, 0.0, 0.0, 1.0])
        >>> values = np.array([0.5, 0.6, 0.7, 0.8])
        >>> dones = np.array([False, False, False, True])
        >>> advantages, returns = compute_gae(rewards, values, dones)
        >>> print(f"Advantages shape: {advantages.shape}")
        >>> print(f"Returns shape: {returns.shape}")
    """
    assert rewards.shape == values.shape == dones.shape, (
        f"Shape mismatch: rewards {rewards.shape}, values {values.shape}, dones {dones.shape}"
    )

    traj_len = len(rewards)
    advantages = np.zeros(traj_len, dtype=np.float32)
    returns = np.zeros(traj_len, dtype=np.float32)

    # Compute GAE using backward pass
    gae = 0.0
    for t in reversed(range(traj_len)):
        if t == traj_len - 1:
            # Last timestep: no next value
            next_value = 0.0
            next_non_terminal = 0.0
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - float(dones[t])

        # TD residual: delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]

        # GAE accumulation: A_t = delta_t + gamma * lambda * (1 - done) * A_{t+1}
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[t] = gae

    # Returns for value function training: R_t = A_t + V(s_t)
    returns = advantages + values

    # Normalize advantages for training stability
    if normalize_advantages:
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages)
        if adv_std > 1e-6:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        else:
            # All advantages identical - skip normalization to prevent 0/0
            pass

    # Safety check for NaN/inf
    if np.any(np.isnan(advantages)) or np.any(np.isinf(advantages)):
        print("WARNING: NaN/inf detected in advantages after GAE computation")
        advantages = np.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)

    if np.any(np.isnan(returns)) or np.any(np.isinf(returns)):
        print("WARNING: NaN/inf detected in returns after GAE computation")
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    return advantages, returns


def compute_gae_batch(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    normalize_advantages: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute GAE for a batch of trajectories.

    This is a batched version that handles multiple trajectories simultaneously.
    Each trajectory is processed independently, resetting GAE at episode boundaries.

    Args:
        rewards: Rewards array of shape (num_trajectories, trajectory_length)
        values: Value estimates of shape (num_trajectories, trajectory_length)
        dones: Episode termination flags of shape (num_trajectories, trajectory_length)
        gamma: Discount factor for future rewards
        gae_lambda: GAE lambda parameter
        normalize_advantages: Whether to normalize advantages across all trajectories

    Returns:
        advantages: GAE advantages of shape (num_trajectories, trajectory_length)
        returns: TD targets of shape (num_trajectories, trajectory_length)
    """
    assert rewards.shape == values.shape == dones.shape, (
        f"Shape mismatch: rewards {rewards.shape}, values {values.shape}, dones {dones.shape}"
    )

    num_trajectories, traj_len = rewards.shape
    all_advantages = []
    all_returns = []

    # Process each trajectory independently
    for i in range(num_trajectories):
        advantages, returns = compute_gae(
            rewards[i],
            values[i],
            dones[i],
            gamma=gamma,
            gae_lambda=gae_lambda,
            normalize_advantages=False,  # Normalize globally after
        )
        all_advantages.append(advantages)
        all_returns.append(returns)

    advantages = np.array(all_advantages, dtype=np.float32)
    returns = np.array(all_returns, dtype=np.float32)

    # Global normalization across all trajectories
    if normalize_advantages:
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages)
        if adv_std > 1e-6:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

    return advantages, returns


def compute_gae_torch(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    normalize_advantages: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch version of GAE computation for GPU acceleration.

    Args:
        rewards: Rewards tensor of shape (trajectory_length,) on any device
        values: Value estimates tensor of shape (trajectory_length,)
        dones: Episode termination flags of shape (trajectory_length,)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        normalize_advantages: Whether to normalize advantages

    Returns:
        advantages: GAE advantages tensor of shape (trajectory_length,)
        returns: TD targets tensor of shape (trajectory_length,)
    """
    assert rewards.shape == values.shape == dones.shape

    device = rewards.device
    traj_len = len(rewards)
    advantages = torch.zeros(traj_len, dtype=torch.float32, device=device)

    # Compute GAE using backward pass
    gae = torch.tensor(0.0, device=device)
    for t in reversed(range(traj_len)):
        if t == traj_len - 1:
            next_value = torch.tensor(0.0, device=device)
            next_non_terminal = torch.tensor(0.0, device=device)
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t].float()

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[t] = gae

    returns = advantages + values

    if normalize_advantages:
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std > 1e-6:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

    # Safety check
    if torch.any(torch.isnan(advantages)) or torch.any(torch.isinf(advantages)):
        print("WARNING: NaN/inf detected in advantages (torch version)")
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)

    if torch.any(torch.isnan(returns)) or torch.any(torch.isinf(returns)):
        print("WARNING: NaN/inf detected in returns (torch version)")
        returns = torch.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    return advantages, returns

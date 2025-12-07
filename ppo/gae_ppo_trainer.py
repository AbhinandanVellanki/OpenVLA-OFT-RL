"""
GAE-based PPO Training Extensions

This module provides GAE (Generalized Advantage Estimation) functionality
to replace GRPO advantages in the OpenVLA PPO trainer.

Key additions:
1. Value head training with separate optimizer and learning rate
2. GAE advantage computation using learned value function
3. Value loss computation and optimization
4. Separate actor/critic learning rates

Usage:
    from vla_oft.min_vla.value_head import ValueHead
    from ppo.gae_ppo_trainer import GAEPPOExtensions, integrate_gae_into_trajectory_buffer

    value_head = ValueHead(input_dim=4096, hidden_dim=1024).to(device)
    gae_ext = GAEPPOExtensions(value_head, vla_model, ...)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional

from ppo.gae import compute_gae, compute_gae_batch


class GAEPPOExtensions:
    """
    Extensions for GAE-based PPO training.

    This class provides methods to:
    - Compute value estimates from VLA hidden states
    - Compute GAE advantages instead of GRPO
    - Train value head with separate optimizer
    - Compute value loss
    """

    def __init__(
        self,
        value_head: nn.Module,
        vla_model: nn.Module,
        actor_lr: float = 1e-5,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        value_loss_coef: float = 0.5,
        freeze_vla_for_critic: bool = False,
        device: torch.device = torch.device("cuda"),
    ):
        """
        Initialize GAE-PPO extensions.

        Args:
            value_head: Value head neural network (critic)
            vla_model: VLA model for extracting hidden states
            actor_lr: Learning rate for actor (VLA LoRA adapters)
            critic_lr: Learning rate for critic (value head)
            gamma: Discount factor for GAE
            gae_lambda: Lambda parameter for GAE
            value_loss_coef: Coefficient for value loss in total loss
            freeze_vla_for_critic: Whether to detach VLA hidden states for critic
            device: Device for computation
        """
        self.value_head = value_head
        self.vla_model = vla_model
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_loss_coef = value_loss_coef
        self.freeze_vla_for_critic = freeze_vla_for_critic
        self.device = device

        # Create separate optimizers for actor and critic
        # Actor: VLA LoRA adapters (low learning rate)
        vla_trainable_params = [p for p in vla_model.parameters() if p.requires_grad]
        self.actor_optimizer = optim.AdamW(vla_trainable_params, lr=actor_lr)

        # Critic: Value head (higher learning rate)
        self.critic_optimizer = optim.AdamW(value_head.parameters(), lr=critic_lr)

        print(f"GAE-PPO Optimizers Initialized:")
        print(f"  Actor LR: {actor_lr}")
        print(f"  Critic LR: {critic_lr}")
        print(f"  VLA trainable params: {sum(p.numel() for p in vla_trainable_params):,}")
        print(f"  Value head params: {sum(p.numel() for p in value_head.parameters()):,}")

    def compute_value_estimates(
        self,
        observations: Dict[str, torch.Tensor],
        instruction: str,
    ) -> torch.Tensor:
        """
        Compute value estimates for observations using value head.

        Args:
            observations: Dict with 'image' and 'proprio' tensors
            instruction: Task instruction string

        Returns:
            values: Tensor of shape (batch_size,) with value estimates
        """
        # Extract hidden states from VLA
        with torch.set_grad_enabled(not self.freeze_vla_for_critic):
            # Forward pass through VLA to get hidden states
            # This requires calling VLA's forward method with observations
            # Note: Implementation depends on VLA's get_hidden_states() method
            hidden_states = self.vla_model.get_hidden_states(
                observations,
                instruction,
            )

            if self.freeze_vla_for_critic:
                hidden_states = hidden_states.detach()

        # Pass through value head to get value estimates
        values = self.value_head(hidden_states).squeeze(-1)

        return values

    def compute_gae_advantages(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE advantages and returns.

        Args:
            rewards: Rewards array of shape (trajectory_length,)
            values: Value estimates array of shape (trajectory_length,)
            dones: Episode termination flags of shape (trajectory_length,)
            normalize: Whether to normalize advantages

        Returns:
            advantages: GAE advantages
            returns: TD targets for value training
        """
        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            normalize_advantages=normalize,
        )
        return advantages, returns

    def compute_value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MSE value loss between value predictions and returns.

        Args:
            values: Value predictions from critic
            returns: TD targets (advantages + values)

        Returns:
            value_loss: MSE loss
        """
        value_loss = nn.functional.mse_loss(values, returns)
        return value_loss

    def compute_total_loss(
        self,
        policy_loss: torch.Tensor,
        value_loss: torch.Tensor,
        entropy_loss: Optional[torch.Tensor] = None,
        entropy_coef: float = 0.01,
    ) -> torch.Tensor:
        """
        Compute total PPO loss combining policy, value, and entropy.

        Formula: L_total = L_policy + value_loss_coef * L_value - entropy_coef * H(π)

        Args:
            policy_loss: PPO clipped policy loss
            value_loss: MSE value loss
            entropy_loss: Entropy of policy (optional)
            entropy_coef: Coefficient for entropy bonus

        Returns:
            total_loss: Combined loss for backpropagation
        """
        total_loss = policy_loss + self.value_loss_coef * value_loss

        if entropy_loss is not None:
            total_loss = total_loss - entropy_coef * entropy_loss

        return total_loss

    def update_actor_critic(
        self,
        total_loss: torch.Tensor,
        max_grad_norm: float = 1.0,
    ) -> Dict[str, float]:
        """
        Perform gradient descent on both actor and critic.

        Args:
            total_loss: Combined loss (policy + value)
            max_grad_norm: Maximum gradient norm for clipping

        Returns:
            grad_info: Dictionary with gradient statistics
        """
        # Zero gradients
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        # Backward pass
        total_loss.backward()

        # Clip gradients
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.vla_model.parameters(), max_grad_norm
        )
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.value_head.parameters(), max_grad_norm
        )

        # Check for NaN/inf gradients
        actor_has_nan = any(
            torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
            for p in self.vla_model.parameters()
            if p.grad is not None
        )
        critic_has_nan = any(
            torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
            for p in self.value_head.parameters()
            if p.grad is not None
        )

        # Update parameters if gradients are valid
        if not actor_has_nan and not critic_has_nan:
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            success = True
        else:
            print("WARNING: NaN/inf detected in gradients, skipping update")
            success = False

        grad_info = {
            "actor_grad_norm": actor_grad_norm.item() if isinstance(actor_grad_norm, torch.Tensor) else actor_grad_norm,
            "critic_grad_norm": critic_grad_norm.item() if isinstance(critic_grad_norm, torch.Tensor) else critic_grad_norm,
            "actor_has_nan": actor_has_nan,
            "critic_has_nan": critic_has_nan,
            "update_success": success,
        }

        return grad_info


def integrate_gae_into_trajectory_buffer(trajectory_buffer, gae_extensions, use_gae=True):
    """
    Modify trajectory buffer to use GAE advantages instead of GRPO.

    This function patches the compute_advantages method of TrajectoryBuffer
    to use GAE when use_gae=True.

    Args:
        trajectory_buffer: TrajectoryBuffer instance
        gae_extensions: GAEPPOExtensions instance
        use_gae: Whether to use GAE (True) or GRPO (False)
    """
    if not use_gae:
        return  # Keep original GRPO behavior

    # Store original method
    original_compute_advantages = trajectory_buffer.compute_advantages

    # Define new GAE-based compute_advantages
    def compute_advantages_gae(self, normalize=True):
        """
        Compute GAE advantages instead of GRPO.

        This replaces the GRPO advantage computation with GAE.
        """
        if len(self.trajectories) == 0:
            print("WARNING: No trajectories to compute advantages for")
            return

        print(f"\nComputing GAE advantages for {len(self.trajectories)} trajectories...")

        for traj_idx, traj in enumerate(self.trajectories):
            traj_len = len(traj["rewards"])

            # Extract trajectory data
            rewards = np.array(traj["rewards"], dtype=np.float32)
            values = np.array(traj["values"], dtype=np.float32)
            dones = np.array(traj["dones"], dtype=bool)

            # Compute GAE advantages and returns
            advantages, returns = gae_extensions.compute_gae_advantages(
                rewards=rewards,
                values=values,
                dones=dones,
                normalize=False,  # Normalize globally after
            )

            # Store in trajectory
            traj["advantages"] = advantages.tolist()
            traj["returns"] = returns.tolist()

        # Global normalization across all trajectories
        if normalize:
            all_advantages = []
            for traj in self.trajectories:
                all_advantages.extend(traj["advantages"])

            all_advantages = np.array(all_advantages, dtype=np.float32)
            adv_mean = np.mean(all_advantages)
            adv_std = np.std(all_advantages)

            if adv_std > 1e-6:
                for traj in self.trajectories:
                    traj["advantages"] = [
                        (a - adv_mean) / (adv_std + 1e-8)
                        for a in traj["advantages"]
                    ]
                print(f"✓ Normalized advantages: mean={adv_mean:.4f}, std={adv_std:.4f}")
            else:
                print(f"⚠ Skipping normalization (std={adv_std:.4f} too small)")

        print("✓ GAE advantages computed")

    # Bind new method to instance (not class)
    import types
    trajectory_buffer.compute_advantages = types.MethodType(
        compute_advantages_gae, trajectory_buffer
    )

    print("✓ Trajectory buffer patched to use GAE advantages")


# Example usage:
#
# from vla_oft.min_vla.value_head import ValueHead
#
# # Create value head and GAE extensions
# value_head = ValueHead(input_dim=4096, hidden_dim=1024).to(device)
# gae_ext = GAEPPOExtensions(
#     value_head=value_head,
#     vla_model=actor.vla,
#     actor_lr=1e-5,
#     critic_lr=3e-4,
#     gamma=0.99,
#     gae_lambda=0.95,
#     value_loss_coef=0.5,
#     freeze_vla_for_critic=False,
#     device=device,
# )
#
# # Integrate into trajectory buffer
# integrate_gae_into_trajectory_buffer(
#     trajectory_buffer=trajectory_buffer,
#     gae_extensions=gae_ext,
#     use_gae=True,
# )
#
# # During rollout: compute and store value estimates
# values = gae_ext.compute_value_estimates(observations, instruction)
# trajectory_buffer.add(..., value=values)
#
# # Compute GAE advantages
# trajectory_buffer.compute_advantages(normalize=True)
#
# # During policy update: train both actor and critic
# values = gae_ext.compute_value_estimates(observations, instruction)
# value_loss = gae_ext.compute_value_loss(values, returns)
# total_loss = gae_ext.compute_total_loss(policy_loss, value_loss)
# grad_info = gae_ext.update_actor_critic(total_loss, max_grad_norm=1.0)

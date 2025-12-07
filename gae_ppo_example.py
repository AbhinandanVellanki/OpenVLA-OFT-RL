"""
Example: GAE-based PPO Training for OpenVLA

This script demonstrates how to integrate GAE (Generalized Advantage Estimation)
into the OpenVLA PPO training pipeline.

Key modifications from standard PPO:
1. Separate actor/critic optimizers with different learning rates
2. GAE advantage computation using learned value function
3. Value head training with MSE loss on TD targets
4. Combined policy + value loss optimization

Usage:
    # Single task with GAE
    python gae_ppo_example.py --task-id 0 --use-gae --timesteps 100000

    # Multi-task with GAE
    python gae_ppo_example.py --task-ids 0 1 2 3 --num-envs 4 --use-gae

    # Compare with GRPO (disable GAE)
    python gae_ppo_example.py --task-id 0 --timesteps 100000
"""

import argparse
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Example integration code
EXAMPLE_CODE = """
#
# GAE-PPO Integration Example
# ============================
#
# This example shows how to modify the existing OpenVLA_PPO.py training loop
# to use GAE-based advantage estimation instead of GRPO.
#

# Step 1: Import GAE modules
from vla_oft.min_vla.value_head import ValueHead
from ppo.gae_ppo_trainer import (
    GAEPPOExtensions,
    integrate_gae_into_trajectory_buffer,
)

# Step 2: Create value head (in __init__)
self.value_head = ValueHead(
    input_dim=4096,  # OpenVLA hidden size
    hidden_dim=1024,
).to(self.training_device)

# Step 3: Create GAE extensions (replaces single optimizer)
self.gae_ext = GAEPPOExtensions(
    value_head=self.value_head,
    vla_model=self.actor.vla,
    actor_lr=self.cfg.actor_lr,      # 1e-5
    critic_lr=self.cfg.critic_lr,     # 3e-4
    gamma=self.cfg.gamma,
    gae_lambda=self.cfg.gae_lambda,
    value_loss_coef=self.cfg.value_loss_coef,
    freeze_vla_for_critic=self.cfg.freeze_vla_for_critic,
    device=self.training_device,
)

# Step 4: Integrate GAE into trajectory buffer
integrate_gae_into_trajectory_buffer(
    trajectory_buffer=self.trajectory_buffer,
    gae_extensions=self.gae_ext,
    use_gae=self.cfg.use_gae,
)

# Step 5: During rollout collection (collect_rollouts method)
# Compute value estimates for each observation
with torch.no_grad():
    # Get VLA hidden states
    hidden_states = self.actor.vla.get_hidden_states(
        observations=obs_dict,
        instruction=instruction,
    )

    # Compute value estimate
    value = self.value_head(hidden_states).item()

# Add to trajectory buffer (value parameter is now used!)
self.trajectory_buffer.add(
    observation=obs_dict,
    action_response=action_response,
    action_ids=action_ids,
    reward=reward,
    done=done,
    log_prob=log_prob,
    value=value,  # Now used for GAE!
    input_ids=input_ids,
    attention_mask=attention_mask,
    pixel_values=pixel_values,
)

# Step 6: During policy update (update_policy method)
# Forward pass through VLA + value head
with torch.set_grad_enabled(True):
    # Get hidden states (gradients enabled for critic training)
    hidden_states = self.actor.vla.get_hidden_states(
        observations=batch_obs,
        instruction=instruction,
    )

    # Compute value estimates
    values = self.value_head(hidden_states).squeeze(-1)

    # Get returns from trajectory (computed by GAE)
    returns = torch.tensor(
        batch_returns, dtype=torch.float32, device=self.training_device
    )

    # Compute value loss
    value_loss = self.gae_ext.compute_value_loss(values, returns)

# Compute policy loss (unchanged)
policy_loss = compute_ppo_policy_loss(
    new_log_probs=new_log_probs,
    old_log_probs=old_log_probs,
    advantages=advantages,
    clip_ratio_high=self.cfg.clip_ratio_high,
    clip_ratio_low=self.cfg.clip_ratio_low,
)

# Compute total loss
total_loss = self.gae_ext.compute_total_loss(
    policy_loss=policy_loss,
    value_loss=value_loss,
    entropy_loss=None,  # Optional
    entropy_coef=self.cfg.entropy_coef,
)

# Update both actor and critic
grad_info = self.gae_ext.update_actor_critic(
    total_loss=total_loss,
    max_grad_norm=self.cfg.max_grad_norm,
)

# Log metrics
print(f"Policy Loss: {policy_loss.item():.4f}")
print(f"Value Loss: {value_loss.item():.4f}")
print(f"Total Loss: {total_loss.item():.4f}")
print(f"Actor Grad Norm: {grad_info['actor_grad_norm']:.4f}")
print(f"Critic Grad Norm: {grad_info['critic_grad_norm']:.4f}")

# Step 7: Save/load value head in checkpoints
# Save
torch.save({
    'vla_state_dict': self.actor.vla.state_dict(),
    'value_head_state_dict': self.value_head.state_dict(),
    'actor_optimizer_state_dict': self.gae_ext.actor_optimizer.state_dict(),
    'critic_optimizer_state_dict': self.gae_ext.critic_optimizer.state_dict(),
    'timesteps': self.total_timesteps,
}, checkpoint_path)

# Load
checkpoint = torch.load(checkpoint_path)
self.actor.vla.load_state_dict(checkpoint['vla_state_dict'])
self.value_head.load_state_dict(checkpoint['value_head_state_dict'])
self.gae_ext.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
self.gae_ext.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

"""


def main():
    parser = argparse.ArgumentParser(description="GAE-PPO Training Example")
    parser.add_argument("--task-suite", type=str, default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--task-ids", type=int, nargs="+", default=None)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--use-gae", action="store_true", help="Enable GAE")
    parser.add_argument("--config", type=str, default="configs/gae_ppo_config.yaml")
    args = parser.parse_args()

    print("="*80)
    print("GAE-PPO Integration Example")
    print("="*80)
    print("\nThis is an example script showing how to integrate GAE into OpenVLA PPO.")
    print("\nTo actually run training, you need to:")
    print("1. Copy the integration code below into OpenVLA_PPO.py")
    print("2. Modify the __init__, collect_rollouts, and update_policy methods")
    print("3. Run the modified training script")
    print("\n" + "="*80)
    print("Integration Code:")
    print("="*80)
    print(EXAMPLE_CODE)
    print("="*80)
    print("\nConfiguration:")
    print(f"  Task Suite: {args.task_suite}")
    print(f"  Task ID: {args.task_id}")
    print(f"  Task IDs: {args.task_ids}")
    print(f"  Num Envs: {args.num_envs}")
    print(f"  Timesteps: {args.timesteps}")
    print(f"  Use GAE: {args.use_gae}")
    print(f"  Config: {args.config}")
    print("="*80)

    print("\n✓ See ppo/gae_ppo_trainer.py for implementation details")
    print("✓ See ppo/gae.py for GAE computation")
    print("✓ See ppo/tests/test_gae.py for unit tests")
    print("✓ See configs/gae_ppo_config.yaml for config template")
    print("\nTo integrate into your training:")
    print("1. Import GAE modules in OpenVLA_PPO.py")
    print("2. Replace single optimizer with GAEPPOExtensions")
    print("3. Compute value estimates during rollout")
    print("4. Use GAE for advantage computation")
    print("5. Train value head with policy")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

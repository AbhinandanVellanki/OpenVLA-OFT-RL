"""
Example PPO training script using LIBERO environments.

This is a comprehensive template showing how to integrate LIBERO with RL training.
Demonstrates all features of the libero_rl package:
- Single and vectorized environments
- Different observation modes (image, image+state)
- Action normalization for VLA models
- Reward shaping options
- Task utilities and initial state management

Implements a PPOTrainer class for structured training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import random


@dataclass
class RolloutBuffer:
    """Simple buffer for storing rollout data."""
    observations: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]
    dones: List[bool]
    values: List[float]
    log_probs: List[float]
    
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
    
    def get(self) -> Dict[str, np.ndarray]:
        return {
            "observations": np.array(self.observations),
            "actions": np.array(self.actions),
            "rewards": np.array(self.rewards),
            "dones": np.array(self.dones),
            "values": np.array(self.values),
            "log_probs": np.array(self.log_probs),
        }


class PPOTrainer:
    """
    PPO Trainer for LIBERO environments.
    
    Implements Proximal Policy Optimization with:
    - Generalized Advantage Estimation (GAE)
    - Clipped surrogate objective
    - Value function learning
    - Entropy regularization
    """
    
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize PPO trainer.
        
        Args:
            actor: Policy network (outputs action distribution parameters)
            critic: Value network (outputs state values)
            actor_lr: Learning rate for actor
            critic_lr: Learning rate for critic
            clip_param: PPO clipping parameter (epsilon)
            gamma: Discount factor
            gae_lambda: Lambda for GAE
            entropy_coef: Entropy bonus coefficient
            value_loss_coef: Value loss coefficient
            max_grad_norm: Max gradient norm for clipping
            device: Device to run on
        """
        self.device = device

        # TODO: might not end up using a separate critic if we just implement a value head to the VLA
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        
        # TODO: can I just create the optimizer like this for the VLA?
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.clip_param = clip_param
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
    
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Rewards array [T]
            values: Value estimates [T]
            dones: Done flags [T]
            next_value: Value of next state after trajectory
            
        Returns:
            advantages: GAE advantages [T]
            returns: Discounted returns [T]
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value_t = values[t + 1]
                next_non_terminal = 1.0 - dones[t]
            
            delta = rewards[t] + self.gamma * next_value_t * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(
        self,
        rollout_buffer: 'RolloutBuffer',
        n_epochs: int = 4,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """
        Update policy using PPO.
        
        Args:
            rollout_buffer: Buffer containing rollout data
            n_epochs: Number of optimization epochs
            batch_size: Minibatch size
            
        Returns:
            Dictionary of training statistics
        """
        # Get data from buffer
        data = rollout_buffer.get()
        observations = torch.FloatTensor(data["observations"]).to(self.device)
        actions = torch.FloatTensor(data["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(data["log_probs"]).to(self.device)
        advantages = torch.FloatTensor(data["advantages"]).to(self.device)
        returns = torch.FloatTensor(data["returns"]).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training statistics
        stats = defaultdict(list)
        
        # Multiple epochs of optimization
        for epoch in range(n_epochs):
            # Generate random minibatch indices
            indices = np.arange(len(observations))
            np.random.shuffle(indices)
            
            # Process minibatches
            for start_idx in range(0, len(observations), batch_size):
                end_idx = min(start_idx + batch_size, len(observations))
                mb_indices = indices[start_idx:end_idx]
                
                mb_obs = observations[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                # Get current policy outputs (dummy for now - replace with actual actor)
                # In real implementation, actor outputs action distribution parameters
                new_log_probs = mb_old_log_probs  # Placeholder
                entropy = torch.zeros(1).to(self.device)  # Placeholder
                
                # PPO actor loss
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                actor_loss = actor_loss - self.entropy_coef * entropy.mean()
                
                # Update actor (placeholder - replace with actual update)
                # self.actor_optimizer.zero_grad()
                # actor_loss.backward()
                # nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                # self.actor_optimizer.step()
                
                # Get current value estimates (dummy for now - replace with actual critic)
                values = torch.zeros_like(mb_returns)  # Placeholder
                
                # Critic loss
                value_loss = nn.functional.mse_loss(values, mb_returns)
                
                # Update critic (placeholder - replace with actual update)
                # self.critic_optimizer.zero_grad()
                # value_loss.backward()
                # nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                # self.critic_optimizer.step()
                
                # Track statistics
                stats['actor_loss'].append(actor_loss.item())
                stats['value_loss'].append(value_loss.item())
                stats['entropy'].append(entropy.mean().item())
        
        # Return mean statistics
        return {k: np.mean(v) for k, v in stats.items()}


class DummyPolicy:
    """Dummy policy for demonstration (replace with actual VLA policy)."""
    
    def __init__(self, action_dim: int = 7):
        self.action_dim = action_dim
    
    def get_action(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Get action, value, and log_prob (dummy implementation)."""
        action = np.random.uniform(-1, 1, self.action_dim).astype(np.float32)
        value = 0.0  # Dummy value
        log_prob = 0.0  # Dummy log prob
        return action, value, log_prob
    
    def get_value(self, obs: np.ndarray) -> float:
        """Get value estimate (dummy implementation)."""
        return 0.0


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
        """Compute returns and advantages using GAE."""
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


def collect_rollouts(
    env,
    policy: DummyPolicy,
    buffer: RolloutBuffer,
    n_steps: int = 128,
) -> Dict[str, float]:
    """Collect rollout data from environment."""
    
    obs, info = env.reset()
    buffer.clear()
    
    episode_rewards = []
    episode_lengths = []
    current_episode_reward = 0
    current_episode_length = 0
    
    for step in range(n_steps):
        # Get action from policy
        action, value, log_prob = policy.get_action(obs)
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store transition
        buffer.add(obs, action, reward, done, value, log_prob)
        
        # Track episode stats
        current_episode_reward += reward
        current_episode_length += 1
        
        if done:
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            current_episode_reward = 0
            current_episode_length = 0
            obs, info = env.reset()
        else:
            obs = next_obs
    
    # Compute last value for GAE
    last_value = policy.get_value(obs)
    
    # Compute returns and advantages in buffer
    buffer.compute_returns_and_advantages(last_value)
    
    stats = {
        "mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
        "mean_length": np.mean(episode_lengths) if episode_lengths else 0.0,
        "num_episodes": len(episode_rewards),
    }
    
    return stats


def train_ppo(
    task_suite_name: str = "libero_spatial",
    task_id: int = 0,
    total_timesteps: int = 100000,
    n_steps: int = 128,
    learning_rate: float = 3e-4,
    obs_mode: str = "image",
    reward_shaper: str = "dense",
    action_normalization: str = "openvla",
):
    """
    Template PPO training loop for LIBERO with full feature integration.
    
    This demonstrates all libero_rl package features:
    - Configurable observation modes
    - Action normalization for VLA compatibility
    - Reward shaping options
    - Task information access
    
    Args:
        task_suite_name: LIBERO task suite ("libero_spatial", "libero_10", etc.)
        task_id: Task index within the suite
        total_timesteps: Total training timesteps
        n_steps: Steps per rollout collection
        learning_rate: Learning rate (not used in dummy policy)
        obs_mode: "image", "image_state", or "raw"
        reward_shaper: "sparse", "dense", or reward shaper config dict
        action_normalization: "none", "openvla", or "vla"
    """
    from libero_rl import make_libero_env, get_task, TASK_SUITES
    from libero_rl.utils.task_utils import get_max_episode_length, get_all_task_names
    
    print("=" * 60)
    print("LIBERO PPO Training with Full Feature Integration")
    print("=" * 60)
    
    # Show available task suites
    print(f"\nAvailable task suites: {TASK_SUITES}")
    print(f"Selected suite: {task_suite_name}")
    
    # Get task info before creating environment
    task = get_task(task_suite_name, task_id)
    max_steps = get_max_episode_length(task_suite_name)
    all_tasks = get_all_task_names(task_suite_name)
    
    print(f"\nTask Suite Info:")
    print(f"  - Total tasks in suite: {len(all_tasks)}")
    print(f"  - Selected task: {task.name}")
    print(f"  - Language instruction: {task.language}")
    print(f"  - Recommended max steps: {max_steps}")
    
    print(f"\nEnvironment Configuration:")
    print(f"  - Observation mode: {obs_mode}")
    print(f"  - Action normalization: {action_normalization}")
    print(f"  - Reward shaping: {reward_shaper}")
    
    # Create environment with full configuration
    env = make_libero_env(
        task_suite_name=task_suite_name,
        task_id=task_id,
        obs_mode=obs_mode,
        image_size=(224, 224),
        action_normalization=action_normalization,
        reward_shaper=reward_shaper,
    )
    
    print(f"\nEnvironment Details:")
    print(f"  - Task: {env.task_name}")
    print(f"  - Language: {env.task_language}")
    print(f"  - Action space: {env.action_space}")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Max episode length: {env.max_episode_length}")
    print(f"  - Num initial states: {env.num_init_states}")
    
    # Initialize policy and trainer
    policy = DummyPolicy(action_dim=7)
    buffer = RolloutBuffer()
    
    # Create dummy actor and critic for PPOTrainer
    # In real usage, replace with actual VLA actor and critic networks
    class DummyActor(nn.Module):
        def forward(self, x):
            return x
    
    class DummyCritic(nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 1)
    
    # Initialize PPO trainer
    trainer = PPOTrainer(
        actor=DummyActor(),
        critic=DummyCritic(),
        actor_lr=learning_rate,
        critic_lr=learning_rate,
    )
    
    # Training loop
    num_updates = total_timesteps // n_steps
    all_stats = defaultdict(list)
    
    print(f"\nStarting training for {total_timesteps} timesteps ({num_updates} updates)")
    print("-" * 60)
    
    for update in range(num_updates):
        # Collect rollouts
        stats = collect_rollouts(env, policy, buffer, n_steps)
        
        # Update policy using PPO trainer
        if len(buffer.observations) > 0:
            train_stats = trainer.update(buffer, n_epochs=4, batch_size=64)
            stats.update(train_stats)
        
        # Log stats
        for key, value in stats.items():
            all_stats[key].append(value)
        
        # Print progress
        if (update + 1) % 10 == 0:
            timesteps = (update + 1) * n_steps
            recent_rewards = all_stats['mean_reward'][-10:]
            recent_lengths = all_stats['mean_length'][-10:]
            
            log_msg = (f"Update {update + 1}/{num_updates} | "
                      f"Timesteps: {timesteps} | "
                      f"Mean Reward: {np.mean(recent_rewards):.3f} | "
                      f"Mean Length: {np.mean(recent_lengths):.1f} | "
                      f"Episodes: {stats['num_episodes']}")
            
            # Add training stats if available
            if 'actor_loss' in stats:
                log_msg += f" | Actor Loss: {stats['actor_loss']:.4f}"
            if 'value_loss' in stats:
                log_msg += f" | Value Loss: {stats['value_loss']:.4f}"
            
            print(log_msg)
    
    env.close()
    
    print("-" * 60)
    print("Training completed!")
    print(f"Final mean reward: {np.mean(all_stats['mean_reward'][-10:]):.3f}")
    print(f"Total episodes: {sum(all_stats['num_episodes'])}")
    
    return all_stats


def train_ppo_vectorized(
    task_suite_name: str = "libero_spatial",
    task_ids: List[int] = [0, 1, 2, 3],
    num_envs: int = 4,
    total_timesteps: int = 100000,
    n_steps: int = 128,
    obs_mode: str = "image",
    reward_shaper: str = "dense",
):
    """
    Template for vectorized PPO training with multiple parallel environments.
    
    Demonstrates:
    - Vectorized environment creation
    - Multi-task training (different task_ids)
    - Parallel rollout collection
    - Auto-reset functionality
    """
    from libero_rl import make_libero_env
    from libero_rl.utils.reward_shaping import create_reward_shaper
    
    print("=" * 60)
    print("LIBERO Vectorized PPO Training")
    print("=" * 60)
    
    print(f"\nCreating {num_envs} vectorized LIBERO environments")
    print(f"Task suite: {task_suite_name}")
    print(f"Task IDs: {task_ids}")
    
    # Create vectorized environment with auto-reset
    env = make_libero_env(
        task_suite_name=task_suite_name,
        task_id=task_ids,
        num_envs=num_envs,
        obs_mode=obs_mode,
        image_size=(224, 224),
        action_normalization="openvla",
        reward_shaper=reward_shaper,
        auto_reset=True,  # Automatically reset on episode termination
    )
    
    print(f"\nEnvironment Configuration:")
    print(f"  - Observation mode: {obs_mode}")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Action space: {env.action_space}")
    print(f"  - Task languages:")
    for i, lang in enumerate(env.task_languages):
        print(f"    Env {i}: {lang}")
    
    # Initialize policy
    policy = DummyPolicy(action_dim=7)
    
    # Training loop
    print(f"\nStarting training for {total_timesteps} timesteps")
    print("-" * 60)
    
    obs, info = env.reset()
    total_rewards = np.zeros(num_envs)
    episode_rewards = []
    episode_lengths = []
    episode_count = 0
    
    for step in range(total_timesteps // num_envs):
        # Get actions for all environments
        actions = np.stack([
            policy.get_action(obs[i])[0] for i in range(num_envs)
        ])
        
        # Step all environments (auto-reset handles episode boundaries)
        obs, rewards, terminated, truncated, info = env.step(actions)
        total_rewards += rewards
        
        # Track completed episodes
        done_mask = terminated | truncated
        if np.any(done_mask):
            for i in np.where(done_mask)[0]:
                ep_reward = total_rewards[i]
                episode_rewards.append(ep_reward)
                episode_lengths.append(info['step_count'][i] if 'step_count' in info else 0)
                episode_count += 1
                
                if episode_count % 5 == 0:
                    recent_rewards = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
                    print(f"Episode {episode_count} | "
                          f"Env {i} | "
                          f"Reward: {ep_reward:.3f} | "
                          f"Avg (last 10): {np.mean(recent_rewards):.3f}")
                
                total_rewards[i] = 0
        
        if episode_count >= 50:  # Stop after 50 episodes for demo
            break
    
    env.close()
    
    print("-" * 60)
    print("Training completed!")
    print(f"Total episodes: {episode_count}")
    print(f"Mean reward: {np.mean(episode_rewards):.3f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.1f}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
    }


def train_with_custom_reward_shaping():
    """
    Example showing custom reward shaping configuration.
    
    Demonstrates:
    - Creating custom reward shapers
    - Composite reward shaping
    - Dense reward signals
    """
    from libero_rl import make_libero_env
    
    print("=" * 60)
    print("Training with Custom Reward Shaping")
    print("=" * 60)
    
    # Option 1: Use built-in reward shapers
    print("\n1. Dense reward shaper (potential-based):")
    env1 = make_libero_env(
        task_suite_name="libero_spatial",
        task_id=0,
        obs_mode="image",
        reward_shaper="dense",
    )
    
    obs, _ = env1.reset()
    for _ in range(5):
        action = env1.action_space.sample()
        obs, reward, terminated, truncated, info = env1.step(action)
        print(f"  Step reward: {reward:.4f}")
        if terminated or truncated:
            break
    env1.close()
    
    # Option 2: Custom reward shaper with success bonus
    print("\n2. Success bonus reward shaper:")
    reward_config = {
        'type': 'success_bonus',
        'bonus': 10.0,
    }
    env2 = make_libero_env(
        task_suite_name="libero_spatial",
        task_id=0,
        obs_mode="image",
        reward_shaper=reward_config,
    )
    
    obs, _ = env2.reset()
    for i in range(5):
        action = env2.action_space.sample()
        obs, reward, terminated, truncated, info = env2.step(action)
        print(f"  Step {i+1} reward: {reward:.4f} (success bonus on completion)")
        if terminated or truncated:
            break
    env2.close()
    
    print("\n" + "=" * 60)


def demonstrate_observation_modes():
    """
    Example showing different observation modes.
    
    Demonstrates:
    - Image-only observations
    - Image + robot state observations
    - Raw dictionary observations
    """
    from libero_rl import make_libero_env
    
    print("=" * 60)
    print("Different Observation Modes")
    print("=" * 60)
    
    # Mode 1: Image only
    print("\n1. Image-only mode:")
    env1 = make_libero_env(
        task_suite_name="libero_spatial",
        task_id=0,
        obs_mode="image",
        image_size=(224, 224),
    )
    obs, _ = env1.reset()
    print(f"   Observation shape: {obs.shape}")
    print(f"   Observation space: {env1.observation_space}")
    env1.close()
    
    # Mode 2: Image + robot state
    print("\n2. Image + robot state mode:")
    env2 = make_libero_env(
        task_suite_name="libero_spatial",
        task_id=0,
        obs_mode="image_state",
        image_size=(224, 224),
    )
    obs, _ = env2.reset()
    print(f"   Observation keys: {obs.keys()}")
    print(f"   Image shape: {obs['image'].shape}")
    print(f"   Robot state shape: {obs['robot_state'].shape}")
    print(f"   Robot state (pos, quat, gripper): {obs['robot_state']}")
    env2.close()
    
    # Mode 3: Raw observations
    print("\n3. Raw observation mode (all sensors):")
    env3 = make_libero_env(
        task_suite_name="libero_spatial",
        task_id=0,
        obs_mode="raw",
    )
    obs, _ = env3.reset()
    print(f"   Available observation keys: {list(obs.keys())}")
    print(f"   Robot state keys: {[k for k in obs.keys() if 'robot' in k]}")
    print(f"   Object state keys: {[k for k in obs.keys() if 'object' in k]}")
    env3.close()
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PPO training on LIBERO with full feature demonstration"
    )
    parser.add_argument("--suite", type=str, default="libero_spatial",
                       help="Task suite name")
    parser.add_argument("--task", type=int, default=0,
                       help="Task ID within suite")
    parser.add_argument("--timesteps", type=int, default=10000,
                       help="Total training timesteps")
    parser.add_argument("--obs-mode", type=str, default="image",
                       choices=["image", "image_state", "raw"],
                       help="Observation mode")
    parser.add_argument("--reward-shaper", type=str, default="dense",
                       choices=["sparse", "dense"],
                       help="Reward shaping type")
    parser.add_argument("--action-norm", type=str, default="openvla",
                       choices=["none", "openvla", "vla"],
                       help="Action normalization")
    parser.add_argument("--vectorized", action="store_true",
                       help="Use vectorized environments")
    parser.add_argument("--demo-rewards", action="store_true",
                       help="Demonstrate reward shaping options")
    parser.add_argument("--demo-obs", action="store_true",
                       help="Demonstrate observation modes")
    
    args = parser.parse_args()
    
    if args.demo_rewards:
        train_with_custom_reward_shaping()
    elif args.demo_obs:
        demonstrate_observation_modes()
    elif args.vectorized:
        train_ppo_vectorized(
            task_suite_name=args.suite,
            task_ids=[args.task, args.task, args.task, args.task],  # Same task, 4 envs
            num_envs=4,
            total_timesteps=args.timesteps,
            obs_mode=args.obs_mode,
            reward_shaper=args.reward_shaper,
        )
    else:
        train_ppo(
            task_suite_name=args.suite,
            task_id=args.task,
            total_timesteps=args.timesteps,
            obs_mode=args.obs_mode,
            reward_shaper=args.reward_shaper,
            action_normalization=args.action_norm,
        )


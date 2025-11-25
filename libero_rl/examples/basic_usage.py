"""
Example script demonstrating LIBERO RL environment usage.

This script shows how to:
1. Create single and vectorized environments
2. Run basic rollouts
3. Use different observation modes
4. Apply reward shaping
"""

import numpy as np
from libero_rl.envs import LiberoEnv, LiberoVecEnv, make_libero_env


def example_single_env():
    """Example: Single environment with basic rollout."""
    print("=" * 60)
    print("Example 1: Single Environment")
    print("=" * 60)
    
    # Create environment
    env = make_libero_env(
        task_suite_name="libero_spatial",
        task_id=0,
        obs_mode="image",
        image_size=(224, 224),
    )
    
    print(f"Task: {env.task_name}")
    print(f"Language: {env.task_language}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Max episode length: {env.max_episode_length}")
    print(f"Num initial states: {env.num_init_states}")
    
    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"\nReset observation shape: {obs.shape}")
    print(f"Info: {list(info.keys())}")
    
    # Run a few steps with random actions
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode ended at step {step + 1}, success={info['success']}")
            break
    
    print(f"Total reward over {step + 1} steps: {total_reward}")
    
    env.close()
    print("Environment closed.\n")


def example_vectorized_env():
    """Example: Vectorized environment for parallel rollouts."""
    print("=" * 60)
    print("Example 2: Vectorized Environment")
    print("=" * 60)
    
    # Create vectorized environment with 4 parallel envs
    env = make_libero_env(
        task_suite_name="libero_spatial",
        task_id=[0, 1, 2, 3],
        num_envs=4,
        obs_mode="image",
        image_size=(224, 224),
        auto_reset=True,
    )
    
    print(f"Number of environments: {env.num_envs}")
    print(f"Task languages: {env.task_languages}")
    print(f"Action space: {env.single_action_space}")
    print(f"Observation space: {env.single_observation_space}")
    
    # Reset all environments
    obs, info = env.reset(seed=42)
    print(f"\nReset observation shape: {obs.shape}")
    
    # Run a few steps
    total_rewards = np.zeros(4)
    for step in range(10):
        # Sample actions for all environments
        actions = np.stack([env.single_action_space.sample() for _ in range(4)])
        obs, rewards, terminated, truncated, info = env.step(actions)
        total_rewards += rewards
        
        done_mask = terminated | truncated
        if np.any(done_mask):
            print(f"Step {step + 1}: Envs {np.where(done_mask)[0]} completed")
    
    print(f"Total rewards: {total_rewards}")
    
    env.close()
    print("Environment closed.\n")


def example_image_state_obs():
    """Example: Using image + state observations."""
    print("=" * 60)
    print("Example 3: Image + State Observations")
    print("=" * 60)
    
    env = make_libero_env(
        task_suite_name="libero_object",
        task_id=0,
        obs_mode="image_state",
        image_size=(224, 224),
    )
    
    print(f"Observation space: {env.observation_space}")
    
    obs, info = env.reset()
    print(f"\nObservation keys: {obs.keys()}")
    print(f"Image shape: {obs['image'].shape}")
    print(f"Robot state shape: {obs['robot_state'].shape}")
    
    env.close()
    print("Environment closed.\n")


def example_reward_shaping():
    """Example: Using custom reward shaping."""
    print("=" * 60)
    print("Example 4: Reward Shaping")
    print("=" * 60)
    
    # Create environment with dense reward shaping
    env = make_libero_env(
        task_suite_name="libero_spatial",
        task_id=0,
        reward_shaper="dense",
        reward_shaper_kwargs={
            "distance_scale": 1.0,
            "success_bonus": 10.0,
            "step_penalty": 0.01,
        },
    )
    
    obs, info = env.reset()
    
    # Run episode and compare rewards
    rewards = []
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        
        if terminated or truncated:
            break
    
    print(f"Reward statistics over episode:")
    print(f"  Min: {min(rewards):.4f}")
    print(f"  Max: {max(rewards):.4f}")
    print(f"  Mean: {np.mean(rewards):.4f}")
    print(f"  Total: {sum(rewards):.4f}")
    
    env.close()
    print("Environment closed.\n")


def example_evaluation():
    """Example: Setting up for evaluation."""
    print("=" * 60)
    print("Example 5: Evaluation Setup")
    print("=" * 60)
    
    from libero_rl.envs.make_env import make_libero_eval_env, get_task_info
    
    # Get task info
    task_info = get_task_info("libero_spatial", task_id=0)
    print(f"Task info: {task_info}")
    
    # Create evaluation environment
    eval_env = make_libero_eval_env(
        task_suite_name="libero_spatial",
        task_ids=[0, 1],
        num_trials_per_task=10,
        seed=42,
    )
    
    print(f"\nEvaluation environment created with {eval_env.num_envs} parallel envs")
    
    eval_env.close()
    print("Environment closed.\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("LIBERO RL Environment Examples")
    print("=" * 60 + "\n")
    
    try:
        example_single_env()
        example_vectorized_env()
        example_image_state_obs()
        example_reward_shaping()
        example_evaluation()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except ImportError as e:
        print(f"\nImport error: {e}")
        print("Make sure LIBERO and its dependencies are installed.")
        print("Run: pip install -e /path/to/LIBERO")
    
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()

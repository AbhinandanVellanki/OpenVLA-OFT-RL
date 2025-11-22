"""Simple test to verify libero_env.py works correctly."""

import numpy as np
import types
from libero_env import VLAEnv


def test_libero_env_basic():
    """Test basic functionality of VLAEnv."""
    # Create minimal config
    cfg = types.SimpleNamespace(
        seed=42,
        env_gpu_id=0,
        num_tasks_per_suite=1,
        n_rollout_threads=1,
        task_ids=None,
        max_env_length=0,  # Use default from task suite
        num_steps_wait=0,
        model_family="openvla",
        save_video=False,
        center_crop=False,
        exp_dir="/tmp/test_exp",
        num_trials_per_task=1,
        penalty_reward_value=-1.0,
        non_stop_penalty=False,
        verify_reward_value=1.0,
        task_suite_name="libero_spatial",
    )

    # Create environment
    env = VLAEnv(cfg, mode="train")
    
    # Test initialization
    assert env is not None
    assert env.action_space == (7,)
    assert env.observation_space["pixel_values"] == (3, 224, 224)
    assert env.observation_space["prompts"] == (1,)
    
    # Test reset - check shapes of outputs
    obs, info = env.reset()
    assert obs is not None
    assert hasattr(obs, "pixel_values")
    assert hasattr(obs, "prompts")
    
    # Check observation shapes
    assert isinstance(obs.pixel_values, list), f"Expected list, got {type(obs.pixel_values)}"
    assert len(obs.pixel_values) == env.env_num, f"Expected {env.env_num} observations, got {len(obs.pixel_values)}"
    assert isinstance(obs.prompts, list), f"Expected list, got {type(obs.prompts)}"
    assert len(obs.prompts) == env.env_num, f"Expected {env.env_num} prompts, got {len(obs.prompts)}"
    
    # Check pixel_values shape (should be PIL Images or arrays)
    for i, pixel_val in enumerate(obs.pixel_values):
        assert pixel_val is not None, f"pixel_values[{i}] is None"
        # Could be PIL Image or numpy array, both are valid
    
    # Check prompts are strings
    for i, prompt in enumerate(obs.prompts):
        assert isinstance(prompt, str), f"prompts[{i}] should be string, got {type(prompt)}"
        assert len(prompt) > 0, f"prompts[{i}] should not be empty"
    
    # Check info dictionary
    assert "task_description" in info
    assert "step_count" in info
    assert isinstance(info["step_count"], np.ndarray), "step_count should be numpy array"
    assert info["step_count"].shape == (env.env_num,), f"step_count shape should be ({env.env_num},), got {info['step_count'].shape}"
    assert np.all(info["step_count"] == 0), "step_count should be 0 after reset"
    
    # Test step - verify action input shape and output shapes
    action = np.zeros((env.env_num, 7), dtype=np.float32)
    assert action.shape == (env.env_num, 7), f"Action shape should be ({env.env_num}, 7), got {action.shape}"
    
    next_obs, rewards, dones, infos = env.step(action)
    
    # Verify next observation structure
    assert next_obs is not None
    assert hasattr(next_obs, "pixel_values")
    assert hasattr(next_obs, "prompts")
    
    # Check next observation shapes match initial observation
    assert isinstance(next_obs.pixel_values, list), "next_obs.pixel_values should be a list"
    assert len(next_obs.pixel_values) == env.env_num, f"Expected {env.env_num} next observations, got {len(next_obs.pixel_values)}"
    assert isinstance(next_obs.prompts, list), "next_obs.prompts should be a list"
    assert len(next_obs.prompts) == env.env_num, f"Expected {env.env_num} next prompts, got {len(next_obs.prompts)}"
    
    # Verify rewards shape
    assert isinstance(rewards, np.ndarray), f"rewards should be numpy array, got {type(rewards)}"
    assert rewards.shape == (env.env_num,), f"rewards shape should be ({env.env_num},), got {rewards.shape}"
    assert rewards.dtype in [np.float32, np.float64], f"rewards dtype should be float, got {rewards.dtype}"
    
    # Verify dones shape
    assert isinstance(dones, np.ndarray), f"dones should be numpy array, got {type(dones)}"
    assert dones.shape == (env.env_num,), f"dones shape should be ({env.env_num},), got {dones.shape}"
    assert dones.dtype == bool, f"dones dtype should be bool, got {dones.dtype}"
    
    # Verify infos structure
    assert "step_count" in infos
    assert isinstance(infos["step_count"], np.ndarray), "step_count should be numpy array"
    assert infos["step_count"].shape == (env.env_num,), f"step_count shape should be ({env.env_num},), got {infos['step_count'].shape}"
    assert np.all(infos["step_count"] >= 1), "step_count should be >= 1 after step"
    
    # Test multiple steps to verify state transitions
    print(f"  Testing multiple steps...")
    for step in range(3):
        action = np.random.randn(env.env_num, 7).astype(np.float32)
        next_obs, rewards, dones, infos = env.step(action)
        
        # Verify outputs maintain correct shapes
        assert len(next_obs.pixel_values) == env.env_num
        assert len(next_obs.prompts) == env.env_num
        assert rewards.shape == (env.env_num,)
        assert dones.shape == (env.env_num,)
        assert infos["step_count"].shape == (env.env_num,)
    
    # Clean up
    env.close()
    
    print("✓ All basic tests passed!")
    print("✓ Input/output shapes verified!")
    print("✓ Action -> next state transitions verified!")


if __name__ == "__main__":
    test_libero_env_basic()


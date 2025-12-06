"""
Test script to verify VLA action prediction is working correctly.
"""
import sys
import numpy as np
import torch
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "vla-oft"))

from vla_oft.min_vla.config import OpenVLAActorConfig
from vla_oft.min_vla.actor import OpenVLAActor
from libero_rl import make_libero_env
from libero_rl.utils.obs_utils import process_observation_for_vla
from libero_rl.utils.action_utils import process_action_for_libero
from libero_rl.utils.task_utils import get_task

def test_action_prediction():
    """Test that the VLA can predict actions and they look reasonable."""
    
    print("="*70)
    print("Testing VLA Action Prediction")
    print("="*70)
    
    # Initialize VLA
    vla_config = OpenVLAActorConfig()
    vla_config.use_multi_gpu = False  # Simpler for testing
    
    print("\n1. Loading VLA model...")
    actor = OpenVLAActor(vla_config)
    actor.vla.eval()
    print("   ✓ Model loaded")
    
    # Create environment
    print("\n2. Creating environment...")
    task = get_task("libero_spatial", 0)
    print(f"   Task: {task.name}")
    print(f"   Language: {task.language}")
    
    env = make_libero_env(
        task_suite_name="libero_spatial",
        task_id=0,
        num_envs=1,
        obs_mode="raw",
        seed=0,
        num_steps_wait=10,
    )
    print("   ✓ Environment created")
    
    # Reset and get observation
    print("\n3. Getting observation...")
    obs, info = env.reset()
    
    # Process observation
    processed_obs = process_observation_for_vla(
        obs,
        camera_name="agentview",
        resize_size=(224, 224),
        num_images=1,
        center_crop=True,
        crop_scale=0.9,
        return_pil=True,
    )
    
    print(f"   Image shape: {processed_obs['image'][0].size if isinstance(processed_obs['image'], list) else processed_obs['image'].size}")
    print(f"   Proprio shape: {processed_obs['proprio'].shape}")
    print(f"   Proprio values: {processed_obs['proprio']}")
    
    # Prepare for VLA
    actor_obs = {
        "image": processed_obs["image"],
        "proprio": processed_obs["proprio"],
    }
    
    # Predict actions (greedy)
    print("\n4. Predicting actions...")
    with torch.no_grad():
        actions = actor.predict_action(
            obs=actor_obs,
            task_label=task.language,
            unnorm_key="libero_spatial",
        )
    
    print(f"   Actions shape: {actions.shape}")
    print(f"   Actions (raw):")
    for i, act in enumerate(actions):
        print(f"     Action {i}: {act}")
    
    # Process first action
    print("\n5. Processing action for environment...")
    first_action = actions[0]
    processed_action = process_action_for_libero(first_action)
    print(f"   Processed action: {processed_action}")
    print(f"   Gripper value: {processed_action[-1]:.3f} ({'CLOSE' if processed_action[-1] > 0 else 'OPEN'})")
    
    # Execute in environment
    print("\n6. Executing action in environment...")
    obs, reward, terminated, truncated, info = env.step(processed_action)
    print(f"   Reward: {reward}")
    print(f"   Done: {terminated or truncated}")
    print(f"   Success: {info.get('success', False)}")
    
    # Try a few more steps
    print("\n7. Running 10 more steps with action chunking...")
    success = False
    for step in range(10):
        if step % 8 == 0:  # Requery every 8 steps
            processed_obs = process_observation_for_vla(
                obs,
                camera_name="agentview",
                resize_size=(224, 224),
                num_images=1,
                center_crop=True,
                crop_scale=0.9,
                return_pil=True,
            )
            actor_obs = {
                "image": processed_obs["image"],
                "proprio": processed_obs["proprio"],
            }
            with torch.no_grad():
                actions = actor.predict_action(
                    obs=actor_obs,
                    task_label=task.language,
                    unnorm_key="libero_spatial",
                )
        
        action = actions[step % 8]
        processed_action = process_action_for_libero(action)
        obs, reward, terminated, truncated, info = env.step(processed_action)
        
        if terminated or truncated:
            success = info.get('success', False)
            print(f"   Episode ended at step {step+1}")
            print(f"   Success: {success}")
            break
    
    env.close()
    
    print("\n" + "="*70)
    print("Test Summary:")
    print(f"  - Model can predict actions: ✓")
    print(f"  - Actions have reasonable values: {'✓' if np.abs(processed_action[:-1]).max() < 2 else '✗'}")
    print(f"  - Gripper command valid: {'✓' if -1 <= processed_action[-1] <= 1 else '✗'}")
    print(f"  - Episode completed: {'✓' if success else '✗ (failed)'}")
    print("="*70)

if __name__ == "__main__":
    test_action_prediction()

"""
Save example observations from LIBERO environment for inspection.

This script collects and saves sample observations (images and robot states)
from the LIBERO environment to help with debugging and visualization.
"""

import os
import numpy as np
from PIL import Image
from libero_rl import make_libero_env


def save_observations(
    task_suite_name="libero_spatial",
    task_id=0,
    num_observations=5,
    output_dir="libero_observations",
):
    """
    Collect and save observations from LIBERO environment.
    
    Args:
        task_suite_name: Name of the task suite
        task_id: Task ID within the suite
        num_observations: Number of observations to save
        output_dir: Directory to save observations
    """
    print(f"Collecting {num_observations} observations from {task_suite_name} task {task_id}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment with image + state observations
    env = make_libero_env(
        task_suite_name=task_suite_name,
        task_id=task_id,
        obs_mode="image_state",
        image_size=(224, 224),
    )
    
    print(f"\nTask: {env.task_name}")
    print(f"Language instruction: {env.task_language}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}\n")
    
    # Collect observations
    observations = []
    obs, info = env.reset()
    observations.append((obs, info))
    
    for i in range(num_observations - 1):
        # Take random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append((obs, info))
        
        if terminated or truncated:
            obs, info = env.reset()
    
    # Save observations
    print(f"Saving {len(observations)} observations to {output_dir}/")
    
    for i, (obs, info) in enumerate(observations):
        # Save image
        image = obs["image"]
        image_pil = Image.fromarray(image)
        image_path = os.path.join(output_dir, f"obs_{i:02d}_image.png")
        image_pil.save(image_path)
        print(f"  Saved: {image_path}")
        
        # Save robot state
        robot_state = obs["robot_state"]
        state_path = os.path.join(output_dir, f"obs_{i:02d}_state.txt")
        with open(state_path, "w") as f:
            f.write(f"Step: {i}\n")
            f.write(f"State ID: {info.get('state_id', 'N/A')}\n")
            f.write(f"Step Count: {info.get('step_count', 0)}\n")
            f.write(f"\nRobot State (shape={robot_state.shape}):\n")
            f.write(f"  End-effector position (xyz): {robot_state[0:3]}\n")
            f.write(f"  End-effector orientation (quat): {robot_state[3:7]}\n")
            f.write(f"  Gripper state: {robot_state[7:9]}\n")
        print(f"  Saved: {state_path}")
        
        # Save numpy arrays for programmatic access
        np.savez(
            os.path.join(output_dir, f"obs_{i:02d}.npz"),
            image=image,
            robot_state=robot_state,
            step=i,
            state_id=info.get('state_id', -1),
            step_count=info.get('step_count', 0),
        )
    
    env.close()
    
    print(f"\n✓ Successfully saved {num_observations} observations!")
    print(f"  Images: {output_dir}/obs_XX_image.png")
    print(f"  States: {output_dir}/obs_XX_state.txt")
    print(f"  NumPy: {output_dir}/obs_XX.npz")


if __name__ == "__main__":
    save_observations(
        task_suite_name="libero_spatial",
        task_id=0,
        num_observations=5,
        output_dir="libero_observations",
    )

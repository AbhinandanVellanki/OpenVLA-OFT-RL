"""
Factory functions for creating LIBERO environments.
"""

from typing import Dict, Any, Optional, List, Union, Callable
import numpy as np

from libero_rl.envs.libero_env import LiberoEnv
from libero_rl.envs.vec_env import LiberoVecEnv
from libero_rl.utils.reward_shaping import RewardShaper, create_reward_shaper


def make_libero_env(
    task_suite_name: str,
    task_id: Union[int, List[int]],
    num_envs: int = 1,
    task_order_index: int = 0,
    obs_mode: str = "image",
    image_size: tuple = (224, 224),
    camera_name: str = "agentview",
    center_crop: bool = True,
    crop_scale: float = 0.9,
    action_normalization: str = "openvla",
    max_episode_length: Optional[int] = None,
    reward_shaper: Optional[Union[str, RewardShaper]] = None,
    reward_shaper_kwargs: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    resolution: int = 256,
    num_steps_wait: int = 10,
    auto_reset: bool = True,
    state_sampler: Optional[Callable[[int, int], int]] = None,
    **kwargs,
) -> Union[LiberoEnv, LiberoVecEnv]:
    """
    Factory function to create LIBERO environments.
    
    Creates either a single environment or vectorized environment based on num_envs.
    
    Args:
        task_suite_name: LIBERO task suite name:
            - "libero_spatial": Spatial reasoning tasks (10 tasks)
            - "libero_object": Object manipulation tasks (10 tasks)
            - "libero_goal": Goal-conditioned tasks (10 tasks)
            - "libero_10": Combined benchmark (10 tasks)
            - "libero_90": Large benchmark (90 tasks)
        task_id: Task index or list of task indices
        num_envs: Number of parallel environments
        task_order_index: Task ordering for curriculum (0-20)
        obs_mode: Observation mode:
            - "raw": Full observation dict from LIBERO
            - "image": Processed image only
            - "image_state": Image and robot proprioception
        image_size: Target image size (H, W)
        camera_name: Camera for observations ("agentview" or "robot0_eye_in_hand")
        center_crop: Whether to apply center cropping
        crop_scale: Center crop scale factor
        action_normalization: Action normalization mode:
            - "none": No normalization
            - "openvla": Normalize + invert gripper for OpenVLA
            - "vla": Normalize gripper only
        max_episode_length: Maximum steps per episode
        reward_shaper: Reward shaper instance or type name ("sparse", "dense", etc.)
        reward_shaper_kwargs: Arguments for reward shaper creation
        seed: Random seed
        resolution: LIBERO render resolution
        num_steps_wait: Stabilization steps after reset
        auto_reset: Whether to auto-reset on termination (vec env only)
        state_sampler: Custom initial state sampler function
        **kwargs: Additional arguments
        
    Returns:
        LiberoEnv if num_envs=1 and task_id is int, else LiberoVecEnv
        
    Example:
        >>> # Single environment
        >>> env = make_libero_env("libero_spatial", task_id=0)
        >>> obs, info = env.reset()
        
        >>> # Vectorized environment
        >>> env = make_libero_env(
        ...     "libero_spatial",
        ...     task_id=[0, 1, 2, 3],
        ...     num_envs=4,
        ... )
        >>> obs, info = env.reset()
    """
    # Handle reward shaper
    if reward_shaper is None:
        shaper = None
    elif isinstance(reward_shaper, str):
        shaper_kwargs = reward_shaper_kwargs or {}
        shaper = create_reward_shaper(reward_shaper, **shaper_kwargs)
    else:
        shaper = reward_shaper
    
    # Convert single task_id to list for vectorized env
    if isinstance(task_id, int):
        task_ids = [task_id]
    else:
        task_ids = list(task_id)
    
    # Create single or vectorized environment
    if num_envs == 1 and len(task_ids) == 1:
        # Single environment
        return LiberoEnv(
            task_suite_name=task_suite_name,
            task_id=task_ids[0],
            task_order_index=task_order_index,
            obs_mode=obs_mode,
            image_size=image_size,
            camera_name=camera_name,
            center_crop=center_crop,
            crop_scale=crop_scale,
            action_normalization=action_normalization,
            max_episode_length=max_episode_length,
            reward_shaper=shaper,
            seed=seed,
            resolution=resolution,
            num_steps_wait=num_steps_wait,
        )
    else:
        # Vectorized environment
        return LiberoVecEnv(
            task_suite_name=task_suite_name,
            task_ids=task_ids,
            num_envs=num_envs,
            task_order_index=task_order_index,
            obs_mode=obs_mode,
            image_size=image_size,
            camera_name=camera_name,
            center_crop=center_crop,
            crop_scale=crop_scale,
            action_normalization=action_normalization,
            max_episode_length=max_episode_length,
            seed=seed,
            resolution=resolution,
            num_steps_wait=num_steps_wait,
            auto_reset=auto_reset,
            state_sampler=state_sampler,
        )


def make_libero_eval_env(
    task_suite_name: str,
    task_ids: List[int],
    num_trials_per_task: int = 50,
    seed: int = 42,
    **kwargs,
) -> LiberoVecEnv:
    """
    Create a LIBERO environment configured for evaluation.
    
    Sets up environment to iterate through all initial states for each task.
    
    Args:
        task_suite_name: LIBERO task suite name
        task_ids: List of task indices to evaluate
        num_trials_per_task: Number of trials per task
        seed: Random seed
        **kwargs: Additional arguments passed to make_libero_env
        
    Returns:
        LiberoVecEnv configured for evaluation
    """
    # Use sequential state sampling for reproducible evaluation
    state_counter = {"idx": 0}
    
    def sequential_state_sampler(task_id: int, num_states: int) -> int:
        state_id = state_counter["idx"] % min(num_states, num_trials_per_task)
        state_counter["idx"] += 1
        return state_id
    
    return make_libero_env(
        task_suite_name=task_suite_name,
        task_id=task_ids,
        num_envs=len(task_ids),
        seed=seed,
        auto_reset=False,  # Manual control for evaluation
        state_sampler=sequential_state_sampler,
        **kwargs,
    )


def list_available_task_suites() -> List[str]:
    """List available LIBERO task suites."""
    from libero_rl.utils.task_utils import TASK_SUITES
    return TASK_SUITES.copy()


def get_task_info(task_suite_name: str, task_id: int) -> Dict[str, Any]:
    """
    Get information about a specific task.
    
    Args:
        task_suite_name: Task suite name
        task_id: Task index
        
    Returns:
        Dictionary with task information
    """
    from libero_rl.utils.task_utils import (
        get_task,
        get_task_init_states,
        get_max_episode_length,
    )
    
    task = get_task(task_suite_name, task_id)
    init_states = get_task_init_states(task_suite_name, task_id)
    
    return {
        "name": task.name,
        "language": task.language,
        "problem": task.problem,
        "bddl_file": task.bddl_file,
        "num_init_states": len(init_states),
        "max_episode_length": get_max_episode_length(task_suite_name),
    }

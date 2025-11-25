"""
Single LIBERO environment wrapper with Gymnasium API.

Wraps LIBERO's OffScreenRenderEnv with a clean Gymnasium interface
for RL training.
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple, Union, Callable

from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path

from libero_rl.utils.task_utils import TaskConfig, get_task_bddl_file
from libero_rl.utils.obs_utils import (
    get_image_from_obs,
    get_robot_state_from_obs,
    preprocess_image,
    center_crop_image,
)
from libero_rl.utils.action_utils import (
    normalize_gripper_action,
    invert_gripper_action,
    clip_action,
    get_dummy_action,
)
from libero_rl.utils.reward_shaping import RewardShaper, SparseRewardShaper


class LiberoEnv(gym.Env):
    """
    Gymnasium-compatible wrapper for a single LIBERO environment.
    
    This wrapper provides:
    - Standard Gymnasium API (reset, step, render, close)
    - Configurable observation modes (raw dict, image-only, VLA-ready)
    - Action space normalization for VLA model compatibility
    - Pluggable reward shaping
    - Initial state management from pre-saved MuJoCo states
    
    Example:
        >>> env = LiberoEnv(
        ...     task_suite_name="libero_spatial",
        ...     task_id=0,
        ...     obs_mode="image",
        ... )
        >>> obs, info = env.reset()
        >>> action = env.action_space.sample()
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """
    
    metadata = {"render_modes": ["rgb_array"]}
    
    def __init__(
        self,
        task_suite_name: str,
        task_id: int,
        task_order_index: int = 0,
        obs_mode: str = "image",
        image_size: Tuple[int, int] = (224, 224),
        camera_name: str = "agentview",
        center_crop: bool = True,
        crop_scale: float = 0.9,
        action_normalization: str = "openvla",
        max_episode_length: Optional[int] = None,
        reward_shaper: Optional[RewardShaper] = None,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        resolution: int = 256,
        num_steps_wait: int = 10,
    ):
        """
        Initialize LIBERO environment.
        
        Args:
            task_suite_name: LIBERO task suite ("libero_spatial", "libero_object", 
                           "libero_goal", "libero_10", "libero_90")
            task_id: Task index within the suite
            task_order_index: Task ordering index (0-20 for curriculum)
            obs_mode: Observation mode:
                - "raw": Full observation dict from LIBERO
                - "image": Processed image only
                - "image_state": Image and robot proprioception
            image_size: Target image size (H, W) for preprocessing
            camera_name: Camera to use ("agentview" or "robot0_eye_in_hand")
            center_crop: Whether to apply center cropping
            crop_scale: Center crop scale factor
            action_normalization: Action normalization mode:
                - "none": No normalization
                - "openvla": Normalize + invert gripper for OpenVLA
                - "vla": Normalize gripper only
            max_episode_length: Maximum steps per episode (None = use default)
            reward_shaper: Custom reward shaper (None = sparse rewards)
            seed: Random seed
            render_mode: Render mode ("rgb_array" or None)
            resolution: LIBERO render resolution
            num_steps_wait: Steps to wait after reset for stabilization
        """
        super().__init__()
        
        # Store config
        self.task_suite_name = task_suite_name
        self.task_id = task_id
        self.obs_mode = obs_mode
        self.image_size = image_size
        self.camera_name = camera_name
        self.center_crop = center_crop
        self.crop_scale = crop_scale
        self.action_normalization = action_normalization
        self.render_mode = render_mode
        self.resolution = resolution
        self.num_steps_wait = num_steps_wait
        
        # Load task configuration
        self.task_config = TaskConfig(
            task_suite_name=task_suite_name,
            task_id=task_id,
            task_order_index=task_order_index,
            max_episode_length=max_episode_length,
        )
        self.max_episode_length = self.task_config.max_episode_length
        
        # Initialize reward shaper
        self.reward_shaper = reward_shaper or SparseRewardShaper()
        
        # Create underlying LIBERO environment
        self._create_env()
        
        # Define action space (7-DOF: delta_xyz, delta_rpy, gripper)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32,
        )
        
        # Define observation space based on mode
        self.observation_space = self._create_observation_space()
        
        # Episode tracking
        self._step_count = 0
        self._current_obs = None
        self._current_state_id = None
        
        # Random state
        self._rng = np.random.default_rng(seed)
        if seed is not None:
            self._env.seed(seed)
    
    def _create_env(self):
        """Create the underlying LIBERO OffScreenRenderEnv."""
        env_args = {
            "bddl_file_name": self.task_config.bddl_file,
            "camera_heights": self.resolution,
            "camera_widths": self.resolution,
            "camera_names": [self.camera_name, "robot0_eye_in_hand"],
            "has_renderer": False,
            "has_offscreen_renderer": True,
        }
        
        # Remove CUDA_VISIBLE_DEVICES to avoid MuJoCo issues
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        
        self._env = OffScreenRenderEnv(**env_args)
    
    def _create_observation_space(self) -> spaces.Space:
        """Create observation space based on mode."""
        if self.obs_mode == "raw":
            # Return raw dict - we define a simple image space as placeholder
            return spaces.Dict({
                "agentview_image": spaces.Box(
                    low=0, high=255,
                    shape=(self.resolution, self.resolution, 3),
                    dtype=np.uint8,
                ),
            })
        
        elif self.obs_mode == "image":
            return spaces.Box(
                low=0,
                high=255,
                shape=(self.image_size[0], self.image_size[1], 3),
                dtype=np.uint8,
            )
        
        elif self.obs_mode == "image_state":
            return spaces.Dict({
                "image": spaces.Box(
                    low=0, high=255,
                    shape=(self.image_size[0], self.image_size[1], 3),
                    dtype=np.uint8,
                ),
                "robot_state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(9,),  # eef_pos(3) + eef_quat(4) + gripper_qpos(2)
                    dtype=np.float32,
                ),
            })
        
        else:
            raise ValueError(f"Unknown obs_mode: {self.obs_mode}")
    
    def _process_observation(self, raw_obs: Dict[str, Any]) -> Any:
        """Process raw LIBERO observation based on mode."""
        if self.obs_mode == "raw":
            return raw_obs
        
        elif self.obs_mode == "image":
            img = get_image_from_obs(raw_obs, camera_name=self.camera_name)
            if self.center_crop:
                img = center_crop_image(img, self.crop_scale)
            img = preprocess_image(img, self.image_size)
            return img
        
        elif self.obs_mode == "image_state":
            img = get_image_from_obs(raw_obs, camera_name=self.camera_name)
            if self.center_crop:
                img = center_crop_image(img, self.crop_scale)
            img = preprocess_image(img, self.image_size)
            robot_state = get_robot_state_from_obs(raw_obs)
            return {
                "image": img,
                "robot_state": robot_state,
            }
        
        return raw_obs
    
    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """Process action based on normalization mode."""
        action = clip_action(action)
        
        if self.action_normalization == "openvla":
            # OpenVLA outputs gripper in [0, 1], needs inversion
            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)
        
        elif self.action_normalization == "vla":
            # Standard VLA normalization
            action = normalize_gripper_action(action, binarize=True)
        
        return action
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed for this episode
            options: Additional options:
                - "state_id": Specific initial state index to use
                
        Returns:
            observation: Initial observation
            info: Additional information
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._env.seed(seed)
        
        # Reset reward shaper
        self.reward_shaper.reset()
        
        # Select initial state
        if options and "state_id" in options:
            state_id = options["state_id"]
            init_state = self.task_config.get_init_state(state_id)
        else:
            state_id, init_state = self.task_config.sample_init_state(self._rng)
        
        self._current_state_id = state_id
        
        # Reset environment and set initial state
        self._env.reset()
        raw_obs = self._env.set_init_state(init_state)
        
        # Wait for stabilization
        dummy_action = get_dummy_action()
        for _ in range(self.num_steps_wait):
            raw_obs, _, _, _ = self._env.step(dummy_action)
        
        # Process observation
        self._current_obs = raw_obs
        obs = self._process_observation(raw_obs)
        
        # Reset step counter
        self._step_count = 0
        
        info = {
            "task_name": self.task_config.task.name,
            "task_language": self.task_config.language,
            "state_id": state_id,
            "step_count": 0,
        }
        
        return obs, info
    
    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: 7-DOF action array
            
        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether episode ended due to success
            truncated: Whether episode ended due to time limit
            info: Additional information
        """
        # Process action for LIBERO
        processed_action = self._process_action(action)
        
        # Store previous observation for reward shaping
        prev_obs = self._current_obs
        
        # Step environment
        raw_obs, base_reward, done, env_info = self._env.step(processed_action)
        self._current_obs = raw_obs
        self._step_count += 1
        
        # Check termination conditions
        success = self._env.check_success()
        terminated = success
        truncated = self._step_count >= self.max_episode_length
        
        # Compute shaped reward
        reward = self.reward_shaper.compute_reward(
            obs=prev_obs,
            action=action,
            next_obs=raw_obs,
            done=terminated or truncated,
            info=env_info if isinstance(env_info, dict) else {},
            base_reward=float(base_reward),
        )
        
        # Process observation
        obs = self._process_observation(raw_obs)
        
        info = {
            "task_name": self.task_config.task.name,
            "task_language": self.task_config.language,
            "state_id": self._current_state_id,
            "step_count": self._step_count,
            "success": success,
            "base_reward": base_reward,
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Returns:
            RGB array if render_mode is "rgb_array", else None
        """
        if self.render_mode == "rgb_array" and self._current_obs is not None:
            return get_image_from_obs(self._current_obs, camera_name=self.camera_name)
        return None
    
    def close(self):
        """Close the environment."""
        if hasattr(self, "_env"):
            self._env.close()
    
    @property
    def task_language(self) -> str:
        """Get the language instruction for the current task."""
        return self.task_config.language
    
    @property
    def task_name(self) -> str:
        """Get the name of the current task."""
        return self.task_config.task.name
    
    @property
    def num_init_states(self) -> int:
        """Get number of available initial states."""
        return self.task_config.num_init_states
    
    def get_state(self) -> np.ndarray:
        """Get current MuJoCo simulation state."""
        return self._env.get_sim_state()
    
    def set_state(self, state: np.ndarray):
        """Set MuJoCo simulation state."""
        self._current_obs = self._env.set_init_state(state)

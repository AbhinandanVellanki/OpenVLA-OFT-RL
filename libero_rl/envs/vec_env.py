"""
Vectorized LIBERO environment for parallel RL training.

Supports multiple parallel environments with subprocess isolation,
auto-reset on episode termination, and configurable initial state sampling.
"""

import os
import multiprocessing as mp
from multiprocessing import Pipe, Process
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from libero_rl.utils.task_utils import TaskConfig, get_max_episode_length
from libero_rl.utils.action_utils import get_dummy_action
from libero_rl.utils.obs_utils import (
    get_image_from_obs,
    get_robot_state_from_obs,
    preprocess_image,
    center_crop_image,
)


def _worker(
    pipe: mp.connection.Connection,
    task_suite_name: str,
    task_id: int,
    task_order_index: int,
    env_kwargs: Dict[str, Any],
    worker_id: int,
):
    """
    Worker process for vectorized environment.
    
    Runs in a separate process and communicates via pipe.
    """
    # Apply torch.load compatibility patch for PyTorch 2.6+ (LIBERO init states only)
    import torch
    _original_load = torch.load
    def _patched_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            if args and isinstance(args[0], (str, os.PathLike)):
                file_path = str(args[0])
                if '.pruned_init' in file_path or 'init_states' in file_path:
                    kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)
    torch.load = _patched_load
    
    # Import here to avoid issues with multiprocessing
    from libero.libero.envs import OffScreenRenderEnv
    from libero_rl.utils.task_utils import TaskConfig
    
    # Remove CUDA_VISIBLE_DEVICES
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    
    # Reconstruct task config in worker process
    task_config = TaskConfig(
        task_suite_name=task_suite_name,
        task_id=task_id,
        task_order_index=task_order_index,
    )
    
    # Create environment
    env_args = {
        "bddl_file_name": task_config.bddl_file,
        "camera_heights": env_kwargs.get("resolution", 256),
        "camera_widths": env_kwargs.get("resolution", 256),
        "camera_names": env_kwargs.get("camera_names", ["agentview", "robot0_eye_in_hand"]),
        "has_renderer": False,
        "has_offscreen_renderer": True,
    }
    
    env = OffScreenRenderEnv(**env_args)
    
    try:
        while True:
            cmd, data = pipe.recv()
            
            if cmd == "step":
                obs, reward, done, info = env.step(data)
                success = env.check_success()
                pipe.send((obs, reward, done, success, info))
            
            elif cmd == "reset":
                env.reset()
                pipe.send(True)
            
            elif cmd == "set_init_state":
                obs = env.set_init_state(data)
                pipe.send(obs)
            
            elif cmd == "seed":
                env.seed(data)
                pipe.send(True)
            
            elif cmd == "check_success":
                pipe.send(env.check_success())
            
            elif cmd == "get_sim_state":
                pipe.send(env.get_sim_state())
            
            elif cmd == "close":
                env.close()
                pipe.close()
                break
            
            else:
                raise ValueError(f"Unknown command: {cmd}")
    
    except KeyboardInterrupt:
        pass
    finally:
        try:
            env.close()
        except (AttributeError, Exception):
            pass  # Ignore errors during cleanup


class LiberoVecEnv(gym.vector.VectorEnv):
    """
    Vectorized LIBERO environment with subprocess parallelization.
    
    Each environment runs in a separate process for true parallelism.
    Supports:
    - Multiple tasks in parallel
    - Auto-reset on episode termination
    - Configurable initial state sampling
    - Observation preprocessing for VLA models
    
    Example:
        >>> env = LiberoVecEnv(
        ...     task_suite_name="libero_spatial",
        ...     task_ids=[0, 1, 2, 3],
        ...     num_envs=4,
        ... )
        >>> obs, info = env.reset()
        >>> actions = np.stack([env.single_action_space.sample() for _ in range(4)])
        >>> obs, rewards, terminated, truncated, info = env.step(actions)
    """
    
    def __init__(
        self,
        task_suite_name: str,
        task_ids: List[int],
        num_envs: Optional[int] = None,
        task_order_index: int = 0,
        obs_mode: str = "image",
        image_size: Tuple[int, int] = (224, 224),
        camera_name: str = "agentview",
        center_crop: bool = True,
        crop_scale: float = 0.9,
        action_normalization: str = "openvla",
        max_episode_length: Optional[int] = None,
        seed: int = 42,
        resolution: int = 256,
        num_steps_wait: int = 10,
        auto_reset: bool = True,
        state_sampler: Optional[Callable[[int, int], int]] = None,
    ):
        """
        Initialize vectorized LIBERO environment.
        
        Args:
            task_suite_name: LIBERO task suite name
            task_ids: List of task indices to use
            num_envs: Number of parallel environments (default: len(task_ids))
            task_order_index: Task ordering index
            obs_mode: Observation mode ("raw", "image", "image_state")
            image_size: Target image size
            camera_name: Camera name for observations
            center_crop: Whether to center crop images
            crop_scale: Center crop scale
            action_normalization: Action normalization mode
            max_episode_length: Maximum episode length
            seed: Random seed
            resolution: Render resolution
            num_steps_wait: Stabilization steps after reset
            auto_reset: Whether to auto-reset on termination
            state_sampler: Custom function to sample initial state indices
                          signature: (task_id, num_states) -> state_id
        """
        self.task_suite_name = task_suite_name
        self.task_ids = task_ids
        self.num_envs = num_envs or len(task_ids)
        self.obs_mode = obs_mode
        self.image_size = image_size
        self.camera_name = camera_name
        self.center_crop = center_crop
        self.crop_scale = crop_scale
        self.action_normalization = action_normalization
        self.resolution = resolution
        self.num_steps_wait = num_steps_wait
        self.auto_reset = auto_reset
        self.state_sampler = state_sampler
        
        if len(task_ids) < self.num_envs:
            raise ValueError(
                f"Not enough task_ids ({len(task_ids)}) for num_envs ({self.num_envs})"
            )
        
        # Load task configs
        self.task_configs = []
        self.task_languages = []
        for i in range(self.num_envs):
            task_id = task_ids[i % len(task_ids)]
            config = TaskConfig(
                task_suite_name=task_suite_name,
                task_id=task_id,
                task_order_index=task_order_index,
                max_episode_length=max_episode_length,
            )
            self.task_configs.append(config)
            self.task_languages.append(config.language)
        
        self.max_episode_length = self.task_configs[0].max_episode_length
        
        # Random state
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        
        # Episode tracking
        self._step_counts = np.zeros(self.num_envs, dtype=np.int32)
        self._current_state_ids = np.zeros(self.num_envs, dtype=np.int32)
        
        # Create action/observation spaces
        single_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        single_observation_space = self._create_single_observation_space()
        
        super().__init__(
            num_envs=self.num_envs,
            observation_space=single_observation_space,
            action_space=single_action_space,
        )
        
        # Start worker processes
        self._pipes = []
        self._processes = []
        self._start_workers()
    
    def _create_single_observation_space(self) -> spaces.Space:
        """Create observation space for a single environment."""
        if self.obs_mode == "raw":
            return spaces.Dict({
                "agentview_image": spaces.Box(
                    low=0, high=255,
                    shape=(self.resolution, self.resolution, 3),
                    dtype=np.uint8,
                ),
            })
        
        elif self.obs_mode == "image":
            return spaces.Box(
                low=0, high=255,
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
                    shape=(9,),
                    dtype=np.float32,
                ),
            })
        
        else:
            raise ValueError(f"Unknown obs_mode: {self.obs_mode}")
    
    def _start_workers(self):
        """Start worker processes."""
        # Use spawn to avoid MuJoCo fork issues
        ctx = mp.get_context("spawn")
        
        env_kwargs = {
            "resolution": self.resolution,
            "camera_names": [self.camera_name, "robot0_eye_in_hand"],
        }
        
        for i in range(self.num_envs):
            parent_conn, child_conn = Pipe()
            process = ctx.Process(
                target=_worker,
                args=(
                    child_conn,
                    self.task_configs[i].task_suite_name,
                    self.task_configs[i].task_id,
                    self.task_configs[i].task_order_index,
                    env_kwargs,
                    i,
                ),
                daemon=True,
            )
            process.start()
            child_conn.close()
            
            self._pipes.append(parent_conn)
            self._processes.append(process)
        
        # Seed all workers
        for i, pipe in enumerate(self._pipes):
            pipe.send(("seed", self._seed + i))
            pipe.recv()
    
    def _send_all(self, cmd: str, data_list: List[Any]):
        """Send command to all workers."""
        for pipe, data in zip(self._pipes, data_list):
            pipe.send((cmd, data))
    
    def _recv_all(self) -> List[Any]:
        """Receive from all workers."""
        return [pipe.recv() for pipe in self._pipes]
    
    def _process_observation(self, raw_obs: Dict[str, Any]) -> Any:
        """Process a single raw observation."""
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
            return {"image": img, "robot_state": robot_state}
        
        return raw_obs
    
    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """Process action for LIBERO."""
        from libero_rl.utils.action_utils import (
            normalize_gripper_action,
            invert_gripper_action,
            clip_action,
        )
        
        action = clip_action(action)
        
        if self.action_normalization == "openvla":
            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)
        elif self.action_normalization == "vla":
            action = normalize_gripper_action(action, binarize=True)
        
        return action
    
    def _sample_init_state(self, env_idx: int) -> Tuple[int, np.ndarray]:
        """Sample initial state for an environment."""
        config = self.task_configs[env_idx]
        
        if self.state_sampler is not None:
            state_id = self.state_sampler(
                self.task_ids[env_idx % len(self.task_ids)],
                config.num_init_states,
            )
            state_id = max(0, min(state_id, config.num_init_states - 1))
        else:
            state_id = self._rng.integers(0, config.num_init_states)
        
        return state_id, config.get_init_state(state_id)
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset all environments.
        
        Args:
            seed: Random seed
            options: Reset options
            
        Returns:
            observations: Stacked observations from all environments
            info: Dictionary with per-environment info
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._seed = seed
        
        # Reset step counts
        self._step_counts = np.zeros(self.num_envs, dtype=np.int32)
        
        # Reset all environments
        self._send_all("reset", [None] * self.num_envs)
        self._recv_all()
        
        # Sample and set initial states
        init_states = []
        for i in range(self.num_envs):
            state_id, state = self._sample_init_state(i)
            self._current_state_ids[i] = state_id
            init_states.append(state)
        
        self._send_all("set_init_state", init_states)
        raw_obs_list = self._recv_all()
        
        # Wait for stabilization
        dummy_action = get_dummy_action()
        for _ in range(self.num_steps_wait):
            self._send_all("step", [dummy_action] * self.num_envs)
            results = self._recv_all()
            raw_obs_list = [r[0] for r in results]
        
        # Process observations
        obs_list = [self._process_observation(obs) for obs in raw_obs_list]
        
        # Stack observations
        if self.obs_mode == "image":
            observations = np.stack(obs_list)
        elif self.obs_mode == "image_state":
            observations = {
                "image": np.stack([o["image"] for o in obs_list]),
                "robot_state": np.stack([o["robot_state"] for o in obs_list]),
            }
        else:
            observations = obs_list
        
        info = {
            "task_languages": self.task_languages.copy(),
            "state_ids": self._current_state_ids.copy(),
            "step_counts": self._step_counts.copy(),
        }
        
        return observations, info
    
    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Step all environments.
        
        Args:
            actions: Array of actions, shape (num_envs, 7)
            
        Returns:
            observations: Stacked observations
            rewards: Array of rewards
            terminated: Array of termination flags
            truncated: Array of truncation flags
            info: Dictionary with per-environment info
        """
        # Process actions
        processed_actions = [self._process_action(a) for a in actions]
        
        # Step all environments
        self._send_all("step", processed_actions)
        results = self._recv_all()
        
        # Unpack results
        raw_obs_list = [r[0] for r in results]
        rewards = np.array([r[1] for r in results], dtype=np.float32)
        dones = np.array([r[2] for r in results], dtype=bool)
        successes = np.array([r[3] for r in results], dtype=bool)
        
        # Update step counts
        self._step_counts += 1
        
        # Determine termination/truncation
        terminated = successes
        truncated = self._step_counts >= self.max_episode_length
        
        # Auto-reset completed environments
        done_mask = terminated | truncated
        if self.auto_reset and np.any(done_mask):
            done_indices = np.where(done_mask)[0]
            
            # Reset done environments
            for idx in done_indices:
                self._pipes[idx].send(("reset", None))
            for idx in done_indices:
                self._pipes[idx].recv()
            
            # Sample new initial states
            new_init_states = []
            for idx in done_indices:
                state_id, state = self._sample_init_state(idx)
                self._current_state_ids[idx] = state_id
                self._step_counts[idx] = 0
                new_init_states.append(state)
            
            # Set initial states
            for idx, state in zip(done_indices, new_init_states):
                self._pipes[idx].send(("set_init_state", state))
            new_obs_list = [self._pipes[idx].recv() for idx in done_indices]
            
            # Stabilization steps
            dummy_action = get_dummy_action()
            for _ in range(self.num_steps_wait):
                for idx in done_indices:
                    self._pipes[idx].send(("step", dummy_action))
                for i, idx in enumerate(done_indices):
                    result = self._pipes[idx].recv()
                    new_obs_list[i] = result[0]
            
            # Update observations for reset environments
            for i, idx in enumerate(done_indices):
                raw_obs_list[idx] = new_obs_list[i]
        
        # Process observations
        obs_list = [self._process_observation(obs) for obs in raw_obs_list]
        
        # Stack observations
        if self.obs_mode == "image":
            observations = np.stack(obs_list)
        elif self.obs_mode == "image_state":
            observations = {
                "image": np.stack([o["image"] for o in obs_list]),
                "robot_state": np.stack([o["robot_state"] for o in obs_list]),
            }
        else:
            observations = obs_list
        
        info = {
            "task_languages": self.task_languages.copy(),
            "state_ids": self._current_state_ids.copy(),
            "step_counts": self._step_counts.copy(),
            "successes": successes,
            "final_observation": raw_obs_list,  # For value bootstrapping
        }
        
        return observations, rewards, terminated, truncated, info
    
    def close(self):
        """Close all environments and worker processes."""
        for pipe in self._pipes:
            try:
                pipe.send(("close", None))
            except:
                pass
        
        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()

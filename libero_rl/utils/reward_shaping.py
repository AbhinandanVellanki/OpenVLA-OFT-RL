"""
Reward shaping utilities for LIBERO environments.

Provides pluggable reward shaping classes for dense reward signals
during RL training. LIBERO by default only provides sparse rewards (+1 on success).
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class RewardShaper(ABC):
    """Base class for reward shaping."""
    
    @abstractmethod
    def compute_reward(
        self,
        obs: Dict[str, Any],
        action: np.ndarray,
        next_obs: Dict[str, Any],
        done: bool,
        info: Dict[str, Any],
        base_reward: float,
    ) -> float:
        """
        Compute shaped reward.
        
        Args:
            obs: Current observation
            action: Action taken
            next_obs: Next observation
            done: Whether episode is done
            info: Additional info from environment
            base_reward: Original sparse reward from environment
            
        Returns:
            Shaped reward value
        """
        pass
    
    def reset(self):
        """Reset any internal state (called at episode start)."""
        pass


class SparseRewardShaper(RewardShaper):
    """
    Pass-through reward shaper that returns the original sparse reward.
    
    Optionally adds penalties for dummy actions or bonuses for success.
    """
    
    def __init__(
        self,
        success_bonus: float = 0.0,
        dummy_penalty: float = 0.0,
        step_penalty: float = 0.0,
    ):
        """
        Initialize sparse reward shaper.
        
        Args:
            success_bonus: Additional reward on task success
            dummy_penalty: Penalty for taking dummy/no-op actions
            step_penalty: Small penalty per step (encourages efficiency)
        """
        self.success_bonus = success_bonus
        self.dummy_penalty = dummy_penalty
        self.step_penalty = step_penalty
    
    def compute_reward(
        self,
        obs: Dict[str, Any],
        action: np.ndarray,
        next_obs: Dict[str, Any],
        done: bool,
        info: Dict[str, Any],
        base_reward: float,
    ) -> float:
        reward = base_reward
        
        # Add success bonus
        if base_reward > 0 and self.success_bonus > 0:
            reward += self.success_bonus
        
        # Add step penalty
        reward -= self.step_penalty
        
        # Add dummy action penalty
        if self.dummy_penalty != 0:
            from libero.utils.action_utils import is_dummy_action
            if is_dummy_action(action):
                reward -= self.dummy_penalty
        
        return reward


class DenseRewardShaper(RewardShaper):
    """
    Dense reward shaper using potential-based shaping.
    
    Computes rewards based on progress toward goals using object positions
    and robot end-effector state.
    """
    
    def __init__(
        self,
        distance_scale: float = 1.0,
        gripper_scale: float = 0.5,
        success_bonus: float = 10.0,
        step_penalty: float = 0.01,
        gamma: float = 0.99,
    ):
        """
        Initialize dense reward shaper.
        
        Args:
            distance_scale: Scale for distance-based rewards
            gripper_scale: Scale for gripper proximity rewards
            success_bonus: Bonus reward on task success
            step_penalty: Small penalty per step
            gamma: Discount factor for potential-based shaping
        """
        self.distance_scale = distance_scale
        self.gripper_scale = gripper_scale
        self.success_bonus = success_bonus
        self.step_penalty = step_penalty
        self.gamma = gamma
        self._prev_potential = None
    
    def _compute_potential(self, obs: Dict[str, Any]) -> float:
        """
        Compute potential function value from observation.
        
        Higher potential = closer to goal.
        """
        potential = 0.0
        
        # Get end-effector position
        eef_pos = obs.get("robot0_eef_pos", None)
        if eef_pos is None:
            return 0.0
        
        # Compute distance to objects of interest
        # Look for object positions in observation
        for key, value in obs.items():
            if key.endswith("_pos") and not key.startswith("robot"):
                obj_pos = value
                if isinstance(obj_pos, np.ndarray) and obj_pos.shape == (3,):
                    # Negative distance as potential (closer = higher)
                    dist = np.linalg.norm(eef_pos - obj_pos)
                    potential -= dist * self.distance_scale
        
        return potential
    
    def compute_reward(
        self,
        obs: Dict[str, Any],
        action: np.ndarray,
        next_obs: Dict[str, Any],
        done: bool,
        info: Dict[str, Any],
        base_reward: float,
    ) -> float:
        reward = 0.0
        
        # Potential-based shaping: r_shaped = gamma * phi(s') - phi(s)
        current_potential = self._compute_potential(next_obs)
        
        if self._prev_potential is not None:
            shaping_reward = self.gamma * current_potential - self._prev_potential
            reward += shaping_reward
        
        self._prev_potential = current_potential
        
        # Add success bonus
        if base_reward > 0:
            reward += self.success_bonus
        
        # Step penalty
        reward -= self.step_penalty
        
        return reward
    
    def reset(self):
        """Reset potential tracking at episode start."""
        self._prev_potential = None


class SuccessBonusShaper(RewardShaper):
    """
    Simple reward shaper that only adds a bonus on success.
    
    Useful when you want to keep sparse rewards but increase
    the magnitude of success signal.
    """
    
    def __init__(self, bonus: float = 10.0):
        """
        Initialize success bonus shaper.
        
        Args:
            bonus: Reward bonus on success
        """
        self.bonus = bonus
    
    def compute_reward(
        self,
        obs: Dict[str, Any],
        action: np.ndarray,
        next_obs: Dict[str, Any],
        done: bool,
        info: Dict[str, Any],
        base_reward: float,
    ) -> float:
        if base_reward > 0:
            return base_reward + self.bonus
        return base_reward


class CompositeRewardShaper(RewardShaper):
    """
    Combine multiple reward shapers with weights.
    """
    
    def __init__(self, shapers: List[tuple]):
        """
        Initialize composite shaper.
        
        Args:
            shapers: List of (weight, RewardShaper) tuples
        """
        self.shapers = shapers
    
    def compute_reward(
        self,
        obs: Dict[str, Any],
        action: np.ndarray,
        next_obs: Dict[str, Any],
        done: bool,
        info: Dict[str, Any],
        base_reward: float,
    ) -> float:
        total_reward = 0.0
        for weight, shaper in self.shapers:
            reward = shaper.compute_reward(
                obs, action, next_obs, done, info, base_reward
            )
            total_reward += weight * reward
        return total_reward
    
    def reset(self):
        """Reset all component shapers."""
        for _, shaper in self.shapers:
            shaper.reset()


class GripperProximityShaper(RewardShaper):
    """
    Reward shaper based on gripper proximity to objects.
    
    Encourages the robot to approach objects of interest.
    """
    
    def __init__(
        self,
        proximity_threshold: float = 0.1,
        proximity_reward: float = 0.1,
        grasp_reward: float = 0.5,
    ):
        """
        Initialize gripper proximity shaper.
        
        Args:
            proximity_threshold: Distance threshold for proximity reward
            proximity_reward: Reward when within threshold
            grasp_reward: Additional reward when gripper is closed near object
        """
        self.proximity_threshold = proximity_threshold
        self.proximity_reward = proximity_reward
        self.grasp_reward = grasp_reward
    
    def compute_reward(
        self,
        obs: Dict[str, Any],
        action: np.ndarray,
        next_obs: Dict[str, Any],
        done: bool,
        info: Dict[str, Any],
        base_reward: float,
    ) -> float:
        reward = base_reward
        
        eef_pos = next_obs.get("robot0_eef_pos", None)
        gripper_qpos = next_obs.get("robot0_gripper_qpos", None)
        
        if eef_pos is None:
            return reward
        
        # Check proximity to each object
        for key, value in next_obs.items():
            if key.endswith("_pos") and not key.startswith("robot"):
                obj_pos = value
                if isinstance(obj_pos, np.ndarray) and obj_pos.shape == (3,):
                    dist = np.linalg.norm(eef_pos - obj_pos)
                    
                    if dist < self.proximity_threshold:
                        reward += self.proximity_reward
                        
                        # Additional reward if gripper is closing
                        if gripper_qpos is not None:
                            gripper_closed = np.mean(gripper_qpos) < 0.04
                            if gripper_closed:
                                reward += self.grasp_reward
                        break  # Only reward for closest object
        
        return reward


def create_reward_shaper(
    shaper_type: str = "sparse",
    **kwargs,
) -> RewardShaper:
    """
    Factory function to create reward shapers.
    
    Args:
        shaper_type: Type of shaper ("sparse", "dense", "bonus", "proximity")
        **kwargs: Arguments passed to shaper constructor
        
    Returns:
        RewardShaper instance
    """
    shapers = {
        "sparse": SparseRewardShaper,
        "dense": DenseRewardShaper,
        "bonus": SuccessBonusShaper,
        "proximity": GripperProximityShaper,
    }
    
    if shaper_type not in shapers:
        raise ValueError(
            f"Unknown shaper type: {shaper_type}. "
            f"Available: {list(shapers.keys())}"
        )
    
    return shapers[shaper_type](**kwargs)

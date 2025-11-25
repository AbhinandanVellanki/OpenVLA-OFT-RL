"""
Action utilities for LIBERO environments.

Provides functions for action space conversion between VLA models and LIBERO.
"""

import numpy as np
from typing import Union, Optional


# LIBERO action space: 7-DOF (delta_xyz, delta_rpy, gripper)
ACTION_DIM = 7
ACTION_LOW = -1.0
ACTION_HIGH = 1.0


def get_dummy_action() -> np.ndarray:
    """
    Get a no-op/dummy action for LIBERO.
    
    The dummy action has zero delta for position/rotation and gripper open (-1).
    
    Returns:
        Dummy action array of shape (7,)
    """
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


def clip_action(action: np.ndarray) -> np.ndarray:
    """
    Clip action to valid range [-1, 1].
    
    Args:
        action: Action array of shape (..., 7)
        
    Returns:
        Clipped action array
    """
    return np.clip(action, ACTION_LOW, ACTION_HIGH)


def normalize_gripper_action(
    action: np.ndarray,
    binarize: bool = True,
) -> np.ndarray:
    """
    Normalize gripper action from [0, 1] to [-1, 1] range.
    
    VLA models typically output gripper in [0, 1] where:
        - 0 = open
        - 1 = close
    
    LIBERO expects gripper in [-1, 1] where:
        - -1 = open
        - +1 = close
    
    Args:
        action: Action array of shape (..., 7) with gripper in [0, 1]
        binarize: Whether to binarize gripper to {-1, +1}
        
    Returns:
        Action array with gripper in [-1, 1]
    """
    action = np.asarray(action, dtype=np.float32).copy()
    
    # Convert gripper from [0, 1] to [-1, 1]
    # gripper_new = 2 * gripper_old - 1
    action[..., -1] = 2.0 * action[..., -1] - 1.0
    
    if binarize:
        # Binarize: negative -> -1 (open), non-negative -> +1 (close)
        action[..., -1] = np.where(action[..., -1] < 0, -1.0, 1.0)
    
    return action


def denormalize_gripper_action(action: np.ndarray) -> np.ndarray:
    """
    Denormalize gripper action from [-1, 1] to [0, 1] range.
    
    Inverse of normalize_gripper_action.
    
    Args:
        action: Action array of shape (..., 7) with gripper in [-1, 1]
        
    Returns:
        Action array with gripper in [0, 1]
    """
    action = np.asarray(action, dtype=np.float32).copy()
    
    # Convert gripper from [-1, 1] to [0, 1]
    # gripper_new = (gripper_old + 1) / 2
    action[..., -1] = (action[..., -1] + 1.0) / 2.0
    
    return action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """
    Invert gripper action polarity.
    
    Some VLA models (e.g., OpenVLA) have inverted gripper conventions.
    This function flips the sign of the gripper action.
    
    Args:
        action: Action array of shape (..., 7)
        
    Returns:
        Action array with inverted gripper
    """
    action = np.asarray(action, dtype=np.float32).copy()
    action[..., -1] = -action[..., -1]
    return action


def is_dummy_action(action: np.ndarray, atol: float = 1e-6) -> Union[bool, np.ndarray]:
    """
    Check if action(s) are dummy/no-op actions.
    
    Args:
        action: Action array of shape (7,) or (batch, 7)
        atol: Absolute tolerance for comparison
        
    Returns:
        Boolean or boolean array indicating dummy actions
    """
    dummy = get_dummy_action()
    return np.all(np.isclose(action, dummy, atol=atol), axis=-1)


def scale_action(
    action: np.ndarray,
    position_scale: float = 1.0,
    rotation_scale: float = 1.0,
) -> np.ndarray:
    """
    Scale position and rotation components of action.
    
    Useful for adjusting action magnitude during training.
    
    Args:
        action: Action array of shape (..., 7)
        position_scale: Scale factor for position (xyz)
        rotation_scale: Scale factor for rotation (rpy)
        
    Returns:
        Scaled action array
    """
    action = np.asarray(action, dtype=np.float32).copy()
    
    # Scale position (first 3 elements)
    action[..., :3] *= position_scale
    
    # Scale rotation (elements 3-6)
    action[..., 3:6] *= rotation_scale
    
    return clip_action(action)


def add_action_noise(
    action: np.ndarray,
    noise_std: float = 0.1,
    noise_type: str = "gaussian",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Add exploration noise to action.
    
    Args:
        action: Action array of shape (..., 7)
        noise_std: Standard deviation of noise
        noise_type: Type of noise ("gaussian" or "uniform")
        rng: Random number generator
        
    Returns:
        Noisy action array (clipped to valid range)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    action = np.asarray(action, dtype=np.float32).copy()
    
    if noise_type == "gaussian":
        noise = rng.normal(0, noise_std, size=action.shape).astype(np.float32)
    elif noise_type == "uniform":
        noise = rng.uniform(-noise_std, noise_std, size=action.shape).astype(np.float32)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    # Don't add noise to gripper (keep discrete)
    noise[..., -1] = 0.0
    
    return clip_action(action + noise)


def action_to_delta_pose(action: np.ndarray) -> dict:
    """
    Convert action array to delta pose dictionary.
    
    Args:
        action: Action array of shape (7,)
        
    Returns:
        Dictionary with delta_pos, delta_rot, gripper
    """
    return {
        "delta_pos": action[:3],
        "delta_rot": action[3:6],
        "gripper": action[6],
    }


def delta_pose_to_action(
    delta_pos: np.ndarray,
    delta_rot: np.ndarray,
    gripper: float,
) -> np.ndarray:
    """
    Convert delta pose components to action array.
    
    Args:
        delta_pos: Position delta (3,)
        delta_rot: Rotation delta in RPY (3,)
        gripper: Gripper action scalar
        
    Returns:
        Action array of shape (7,)
    """
    return np.concatenate([
        delta_pos,
        delta_rot,
        [gripper],
    ]).astype(np.float32)

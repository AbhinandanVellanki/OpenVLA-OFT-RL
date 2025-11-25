"""
Observation utilities for LIBERO environments.

Provides functions to extract and preprocess observations from LIBERO environments.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
from PIL import Image


def get_image_from_obs(
    obs: Dict[str, Any],
    camera_name: str = "agentview",
    rotate: bool = True,
) -> np.ndarray:
    """
    Extract image from LIBERO observation dictionary.
    
    LIBERO images need to be rotated 180 degrees to match the expected orientation
    for VLA models trained on RLDS datasets.
    
    Args:
        obs: Observation dictionary from LIBERO environment
        camera_name: Camera to extract image from ("agentview" or "robot0_eye_in_hand")
        rotate: Whether to rotate image 180 degrees (default True for VLA compatibility)
        
    Returns:
        Image array of shape (H, W, 3), dtype uint8
    """
    img_key = f"{camera_name}_image"
    if img_key not in obs:
        raise KeyError(
            f"Camera '{camera_name}' not found in observation. "
            f"Available keys: {list(obs.keys())}"
        )
    
    img = obs[img_key]
    
    # Rotate 180 degrees (flip both axes) to match VLA training data orientation
    if rotate:
        img = img[::-1, ::-1]
    
    return img.astype(np.uint8)


def get_robot_state_from_obs(obs: Dict[str, Any]) -> np.ndarray:
    """
    Extract robot proprioceptive state from observation.
    
    Extracts end-effector position, quaternion, and gripper state.
    
    Args:
        obs: Observation dictionary from LIBERO environment
        
    Returns:
        Robot state array of shape (10,):
            - eef_pos (3): End-effector position
            - eef_quat (4): End-effector quaternion (xyzw)
            - gripper_qpos (2): Gripper joint positions
            - joint_pos (1): First joint position (for state tracking)
    """
    state_parts = []
    
    # End-effector position (3)
    if "robot0_eef_pos" in obs:
        state_parts.append(obs["robot0_eef_pos"])
    
    # End-effector quaternion (4)
    if "robot0_eef_quat" in obs:
        state_parts.append(obs["robot0_eef_quat"])
    
    # Gripper joint positions (2)
    if "robot0_gripper_qpos" in obs:
        state_parts.append(obs["robot0_gripper_qpos"])
    
    if len(state_parts) == 0:
        raise KeyError(
            "No robot state keys found in observation. "
            f"Available keys: {list(obs.keys())}"
        )
    
    return np.concatenate(state_parts).astype(np.float32)


def preprocess_image(
    img: np.ndarray,
    resize_size: Tuple[int, int] = (224, 224),
    method: str = "lanczos",
) -> np.ndarray:
    """
    Preprocess image for VLA model input.
    
    Args:
        img: Input image array (H, W, 3)
        resize_size: Target size (height, width)
        method: Interpolation method ("lanczos", "bilinear", "nearest")
        
    Returns:
        Resized image array of shape (resize_size[0], resize_size[1], 3)
    """
    pil_img = Image.fromarray(img.astype(np.uint8))
    
    resample_methods = {
        "lanczos": Image.LANCZOS,
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST,
        "bicubic": Image.BICUBIC,
    }
    resample = resample_methods.get(method.lower(), Image.LANCZOS)
    
    # Resize (PIL uses width, height order)
    pil_img = pil_img.resize((resize_size[1], resize_size[0]), resample=resample)
    
    return np.array(pil_img, dtype=np.uint8)


def center_crop_image(
    img: np.ndarray,
    crop_scale: float = 0.9,
) -> np.ndarray:
    """
    Center crop an image by a scale factor.
    
    Args:
        img: Input image array (H, W, 3)
        crop_scale: Fraction of image area to keep (0.9 = keep 90% of area)
        
    Returns:
        Cropped image array
    """
    h, w = img.shape[:2]
    
    # Calculate crop size (sqrt because area scales quadratically)
    crop_h = int(h * np.sqrt(crop_scale))
    crop_w = int(w * np.sqrt(crop_scale))
    
    # Calculate crop offsets (center crop)
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    
    return img[start_h:start_h + crop_h, start_w:start_w + crop_w]


def process_observation_for_vla(
    obs: Dict[str, Any],
    camera_name: str = "agentview",
    resize_size: Tuple[int, int] = (224, 224),
    center_crop: bool = True,
    crop_scale: float = 0.9,
) -> Dict[str, Any]:
    """
    Process LIBERO observation for VLA model input.
    
    Args:
        obs: Raw observation dictionary from LIBERO
        camera_name: Camera to use for image
        resize_size: Target image size
        center_crop: Whether to apply center cropping
        crop_scale: Center crop scale factor
        
    Returns:
        Processed observation dictionary with:
            - "image": Preprocessed image (H, W, 3)
            - "robot_state": Robot proprioceptive state
    """
    # Extract and process image
    img = get_image_from_obs(obs, camera_name=camera_name, rotate=True)
    
    if center_crop:
        img = center_crop_image(img, crop_scale=crop_scale)
    
    img = preprocess_image(img, resize_size=resize_size)
    
    # Extract robot state
    robot_state = get_robot_state_from_obs(obs)
    
    return {
        "image": img,
        "robot_state": robot_state,
    }


def stack_observations(obs_list: list) -> Dict[str, np.ndarray]:
    """
    Stack a list of observation dictionaries into batched arrays.
    
    Args:
        obs_list: List of observation dictionaries
        
    Returns:
        Dictionary with stacked arrays
    """
    if len(obs_list) == 0:
        return {}
    
    keys = obs_list[0].keys()
    stacked = {}
    
    for key in keys:
        arrays = [obs[key] for obs in obs_list]
        stacked[key] = np.stack(arrays, axis=0)
    
    return stacked

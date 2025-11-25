"""
LIBERO RL Environment Wrapper Package

A Gymnasium-compatible environment wrapper for LIBERO robotics benchmark
designed for online RL training (PPO, SAC, etc.).
"""

# Monkey-patch torch.load for PyTorch 2.6+ compatibility with LIBERO
# The patch only affects calls that load .pruned_init files (LIBERO's initial states)
# Your model loading code remains unaffected
import torch
import os

_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    """
    Wrapper for torch.load that sets weights_only=False for LIBERO initial state files.
    
    This patch only affects loading of LIBERO's .pruned_init files which contain
    simulation states. Your model checkpoints (.pt, .pth, .ckpt) are unaffected.
    """
    if 'weights_only' not in kwargs:
        # Check if this is loading a LIBERO initial state file
        if args and isinstance(args[0], (str, os.PathLike)):
            file_path = str(args[0])
            if '.pruned_init' in file_path or 'init_states' in file_path:
                kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

from libero_rl.envs import LiberoEnv, LiberoVecEnv, make_libero_env
from libero_rl.utils.task_utils import (
    get_benchmark,
    get_task,
    get_task_bddl_file,
    get_task_init_states,
    TASK_SUITES,
)

__version__ = "0.1.0"

__all__ = [
    # Environments
    "LiberoEnv",
    "LiberoVecEnv",
    "make_libero_env",
    # Task utilities
    "get_benchmark",
    "get_task",
    "get_task_bddl_file",
    "get_task_init_states",
    "TASK_SUITES",
]

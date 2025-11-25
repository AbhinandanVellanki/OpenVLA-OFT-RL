# LIBERO RL Environment Wrapper

A Gymnasium-compatible environment wrapper for the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) robotics benchmark, designed for online RL training (PPO, SAC, etc.).

## Features

- **Gymnasium API**: Standard `reset()`, `step()`, `render()`, `close()` interface
- **Vectorized Environments**: Parallel environments with subprocess isolation
- **Flexible Observations**: Raw dict, image-only, or image+state modes
- **Action Normalization**: Built-in support for OpenVLA and other VLA models
- **Reward Shaping**: Pluggable reward shapers (sparse, dense, proximity-based)
- **Task Management**: Easy access to all LIBERO task suites and initial states

## Installation

1. Install LIBERO first:
```bash
cd /path/to/LIBERO
pip install -e .
```

2. Install dependencies:
```bash
pip install gymnasium numpy pillow torch
```

## Quick Start

### Single Environment

```python
from libero.envs import make_libero_env

# Create environment
env = make_libero_env(
    task_suite_name="libero_spatial",
    task_id=0,
    obs_mode="image",
    image_size=(224, 224),
)

# Standard Gymnasium API
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

### Vectorized Environment

```python
from libero.envs import make_libero_env

# Create 4 parallel environments with different tasks
env = make_libero_env(
    task_suite_name="libero_spatial",
    task_id=[0, 1, 2, 3],
    num_envs=4,
    obs_mode="image",
    auto_reset=True,
)

obs, info = env.reset()
actions = np.stack([env.single_action_space.sample() for _ in range(4)])
obs, rewards, terminated, truncated, info = env.step(actions)
env.close()
```

## Configuration Options

### Observation Modes

- `"raw"`: Full observation dictionary from LIBERO
- `"image"`: Processed image only (default)
- `"image_state"`: Image and robot proprioception

### Action Normalization

- `"none"`: No normalization
- `"openvla"`: Normalize gripper [0,1]→[-1,1] + invert for OpenVLA
- `"vla"`: Normalize gripper only

### Reward Shapers

```python
# Sparse rewards (default)
env = make_libero_env(..., reward_shaper="sparse")

# Dense rewards with distance-based shaping
env = make_libero_env(
    ...,
    reward_shaper="dense",
    reward_shaper_kwargs={
        "distance_scale": 1.0,
        "success_bonus": 10.0,
        "step_penalty": 0.01,
    }
)
```

## Available Task Suites

| Suite | Tasks | Description |
|-------|-------|-------------|
| `libero_spatial` | 10 | Spatial reasoning tasks |
| `libero_object` | 10 | Object manipulation tasks |
| `libero_goal` | 10 | Goal-conditioned tasks |
| `libero_10` | 10 | Combined benchmark |
| `libero_90` | 90 | Large benchmark |

## API Reference

### LiberoEnv

```python
class LiberoEnv(gym.Env):
    """Single LIBERO environment."""
    
    @property
    def task_language(self) -> str:
        """Get language instruction."""
    
    @property
    def task_name(self) -> str:
        """Get task name."""
    
    @property
    def num_init_states(self) -> int:
        """Get number of available initial states."""
    
    def get_state(self) -> np.ndarray:
        """Get current MuJoCo state."""
    
    def set_state(self, state: np.ndarray):
        """Set MuJoCo state."""
```

### LiberoVecEnv

```python
class LiberoVecEnv(gym.vector.VectorEnv):
    """Vectorized LIBERO environment."""
    
    # Attributes
    task_languages: List[str]  # Language instructions for each env
    num_envs: int  # Number of parallel environments
```

### Utility Functions

```python
from libero.utils.task_utils import (
    get_benchmark,           # Get benchmark object
    get_task,                # Get task by ID
    get_task_init_states,    # Load initial states
    get_max_episode_length,  # Get recommended episode length
    TASK_SUITES,             # List of available suites
)

from libero.utils.action_utils import (
    normalize_gripper_action,  # [0,1] → [-1,1]
    invert_gripper_action,     # Flip gripper sign
    get_dummy_action,          # No-op action
)

from libero.utils.obs_utils import (
    get_image_from_obs,        # Extract image
    get_robot_state_from_obs,  # Extract proprioception
    preprocess_image,          # Resize image
)
```

## Examples

See `examples/` directory:
- `basic_usage.py`: Basic environment usage examples
- `save_observations.py`: Save environment observations for inspection

## Project Structure

```
libero_rl/
├── __init__.py
├── envs/
│   ├── __init__.py
│   ├── libero_env.py      # Single environment
│   ├── vec_env.py         # Vectorized environment
│   └── make_env.py        # Factory functions
├── utils/
│   ├── __init__.py
│   ├── task_utils.py      # Task/benchmark loading
│   ├── obs_utils.py       # Observation processing
│   ├── action_utils.py    # Action processing
│   └── reward_shaping.py  # Reward shapers
└── examples/
    ├── __init__.py
    ├── basic_usage.py     # Basic environment examples
    └── save_observations.py  # Save observations for inspection
```

## License

See LIBERO repository for license information.

"""
Task utilities for LIBERO benchmark.

Provides functions to load tasks, benchmarks, BDDL files, and initial states
from the LIBERO package.
"""

import os
from typing import List, Optional, Dict, Any, Tuple
import torch
import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.benchmark import Task

# Available task suites in LIBERO
TASK_SUITES = [
    "libero_spatial",
    "libero_object", 
    "libero_goal",
    "libero_10",
    "libero_90",
]

# Task-specific maximum episode lengths
TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def get_benchmark(task_suite_name: str, task_order_index: int = 0):
    """
    Get a benchmark object for the specified task suite.
    
    Args:
        task_suite_name: Name of the task suite (e.g., "libero_spatial", "libero_10")
        task_order_index: Index of task ordering (0-20 for 10-task suites)
        
    Returns:
        Benchmark object with task definitions
    """
    benchmark_dict = benchmark.get_benchmark_dict()
    if task_suite_name.lower() not in benchmark_dict:
        raise ValueError(
            f"Unknown task suite: {task_suite_name}. "
            f"Available: {list(benchmark_dict.keys())}"
        )
    return benchmark_dict[task_suite_name.lower()](task_order_index=task_order_index)


def get_task(task_suite_name: str, task_id: int, task_order_index: int = 0) -> Task:
    """
    Get a specific task from a task suite.
    
    Args:
        task_suite_name: Name of the task suite
        task_id: Index of the task within the suite
        task_order_index: Index of task ordering
        
    Returns:
        Task namedtuple with name, language, problem, problem_folder, 
        bddl_file, init_states_file
    """
    bm = get_benchmark(task_suite_name, task_order_index)
    if task_id < 0 or task_id >= bm.get_num_tasks():
        raise ValueError(
            f"Task ID {task_id} out of range. "
            f"Suite {task_suite_name} has {bm.get_num_tasks()} tasks."
        )
    return bm.get_task(task_id)


def get_task_bddl_file(task: Task) -> str:
    """
    Get the full path to the BDDL file for a task.
    
    Args:
        task: Task namedtuple
        
    Returns:
        Absolute path to the BDDL file
    """
    return os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file
    )


def get_task_init_states(task_suite_name: str, task_id: int, task_order_index: int = 0) -> np.ndarray:
    """
    Load pre-saved initial states (MuJoCo states) for a task.
    
    Args:
        task_suite_name: Name of the task suite
        task_id: Index of the task within the suite
        task_order_index: Index of task ordering
        
    Returns:
        Array of initial states, shape (num_states, state_dim)
    """
    bm = get_benchmark(task_suite_name, task_order_index)
    init_states = bm.get_task_init_states(task_id)
    if isinstance(init_states, torch.Tensor):
        init_states = init_states.numpy()
    return init_states


def get_task_language(task: Task) -> str:
    """
    Get the language instruction for a task.
    
    Args:
        task: Task namedtuple
        
    Returns:
        Natural language instruction string
    """
    return task.language


def get_max_episode_length(task_suite_name: str) -> int:
    """
    Get the recommended maximum episode length for a task suite.
    
    Args:
        task_suite_name: Name of the task suite
        
    Returns:
        Maximum number of steps per episode
    """
    return TASK_MAX_STEPS.get(task_suite_name.lower(), 300)


def get_num_tasks(task_suite_name: str, task_order_index: int = 0) -> int:
    """
    Get the number of tasks in a task suite.
    
    Args:
        task_suite_name: Name of the task suite
        task_order_index: Index of task ordering
        
    Returns:
        Number of tasks in the suite
    """
    bm = get_benchmark(task_suite_name, task_order_index)
    return bm.get_num_tasks()


def get_all_task_names(task_suite_name: str, task_order_index: int = 0) -> List[str]:
    """
    Get names of all tasks in a task suite.
    
    Args:
        task_suite_name: Name of the task suite
        task_order_index: Index of task ordering
        
    Returns:
        List of task names
    """
    bm = get_benchmark(task_suite_name, task_order_index)
    return bm.get_task_names()


def get_all_task_languages(task_suite_name: str, task_order_index: int = 0) -> List[str]:
    """
    Get language instructions for all tasks in a task suite.
    
    Args:
        task_suite_name: Name of the task suite
        task_order_index: Index of task ordering
        
    Returns:
        List of language instructions
    """
    bm = get_benchmark(task_suite_name, task_order_index)
    return [bm.get_task(i).language for i in range(bm.get_num_tasks())]


class TaskConfig:
    """Configuration for a LIBERO task."""
    
    def __init__(
        self,
        task_suite_name: str,
        task_id: int,
        task_order_index: int = 0,
        max_episode_length: Optional[int] = None,
    ):
        """
        Initialize task configuration.
        
        Args:
            task_suite_name: Name of the task suite
            task_id: Index of the task within the suite
            task_order_index: Index of task ordering
            max_episode_length: Override for maximum episode length
        """
        self.task_suite_name = task_suite_name
        self.task_id = task_id
        self.task_order_index = task_order_index
        
        # Load task info
        self.benchmark = get_benchmark(task_suite_name, task_order_index)
        self.task = get_task(task_suite_name, task_id, task_order_index)
        self.bddl_file = get_task_bddl_file(self.task)
        self.init_states = get_task_init_states(task_suite_name, task_id, task_order_index)
        self.language = get_task_language(self.task)
        
        # Episode length
        if max_episode_length is not None:
            self.max_episode_length = max_episode_length
        else:
            self.max_episode_length = get_max_episode_length(task_suite_name)
    
    @property
    def num_init_states(self) -> int:
        """Number of available initial states."""
        return len(self.init_states)
    
    def get_init_state(self, state_id: int) -> np.ndarray:
        """Get a specific initial state by index."""
        return self.init_states[state_id]
    
    def sample_init_state(self, rng: Optional[np.random.Generator] = None) -> Tuple[int, np.ndarray]:
        """
        Sample a random initial state.
        
        Args:
            rng: Random number generator (uses numpy default if None)
            
        Returns:
            Tuple of (state_id, state_array)
        """
        if rng is None:
            state_id = np.random.randint(0, self.num_init_states)
        else:
            state_id = rng.integers(0, self.num_init_states)
        return state_id, self.init_states[state_id]
    
    def __repr__(self) -> str:
        return (
            f"TaskConfig(suite={self.task_suite_name}, task_id={self.task_id}, "
            f"name={self.task.name}, language='{self.language}')"
        )

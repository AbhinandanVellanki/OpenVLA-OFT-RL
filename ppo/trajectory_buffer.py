"""
trajectory_buffer.py

Trajectory-based rollout buffer for PPO training with OpenVLA.
Stores complete episodes with variable lengths using masking.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import torch


class TrajectoryBuffer:
    """
    Buffer for storing trajectory-based rollouts.
    
    Stores complete episodes with padding for variable lengths.
    Uses finish_step markers to indicate episode completion.
    """
    
    def __init__(self):
        self.clear()
    
    def clear(self):
        """Clear all stored data."""
        self.trajectories: List[Dict[str, Any]] = []
        self.current_trajectory: Dict[str, List[Any]] = {
            'observations': [],
            'responses': [],  # Action token IDs
            'input_ids': [],
            'attention_mask': [],
            'pixel_values': [],
            'proprio': [],
            'actions': [],  # Continuous actions (for environment)
            'rewards': [],
            'dones': [],
            'values': [],
            'old_log_probs': [],
        }
        self.episode_step = 0
    
    def add(
        self,
        obs: Dict[str, Any],
        responses: torch.Tensor,  # Action token IDs
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        proprio: Optional[np.ndarray],
        action: np.ndarray,  # Continuous action
        reward: float,
        done: bool,
        value: float,
        old_log_prob: torch.Tensor,
    ):
        """
        Add a single timestep to the current trajectory.
        
        When done=True, finalizes the trajectory and starts a new one.
        """
        self.current_trajectory['observations'].append(obs)
        self.current_trajectory['responses'].append(responses)
        self.current_trajectory['input_ids'].append(input_ids)
        self.current_trajectory['attention_mask'].append(attention_mask)
        self.current_trajectory['pixel_values'].append(pixel_values)
        self.current_trajectory['proprio'].append(proprio)
        self.current_trajectory['actions'].append(action)
        self.current_trajectory['rewards'].append(reward)
        self.current_trajectory['dones'].append(done)
        self.current_trajectory['values'].append(value)
        self.current_trajectory['old_log_probs'].append(old_log_prob)
        
        self.episode_step += 1
        
        if done:
            # Finalize trajectory and add to buffer
            trajectory = {
                'observations': self.current_trajectory['observations'].copy(),
                'responses': torch.stack(self.current_trajectory['responses']).detach(),
                'input_ids': torch.stack(self.current_trajectory['input_ids']).detach(),
                'attention_mask': torch.stack(self.current_trajectory['attention_mask']).detach(),
                'pixel_values': torch.stack(self.current_trajectory['pixel_values']).detach(),
                'proprio': np.stack(self.current_trajectory['proprio']) if self.current_trajectory['proprio'][0] is not None else None,
                'actions': np.stack(self.current_trajectory['actions']),
                'rewards': np.array(self.current_trajectory['rewards']),
                'dones': np.array(self.current_trajectory['dones']),
                'values': np.array(self.current_trajectory['values']),
                'old_log_probs': torch.stack(self.current_trajectory['old_log_probs']).detach(),
                'finish_step': self.episode_step - 1,  # Last step index
                'traj_len': self.episode_step,
            }
            self.trajectories.append(trajectory)
            
            # Reset for next trajectory
            self.current_trajectory = {k: [] for k in self.current_trajectory.keys()}
            self.episode_step = 0
    
    def finalize_partial_trajectory(self):
        """
        Finalize current partial trajectory if it exists.
        
        Called at end of rollout collection if trajectory is incomplete.
        """
        if self.episode_step > 0:
            trajectory = {
                'observations': self.current_trajectory['observations'].copy(),
                'responses': torch.stack(self.current_trajectory['responses']).detach(),
                'input_ids': torch.stack(self.current_trajectory['input_ids']).detach(),
                'attention_mask': torch.stack(self.current_trajectory['attention_mask']).detach(),
                'pixel_values': torch.stack(self.current_trajectory['pixel_values']).detach(),
                'proprio': np.stack(self.current_trajectory['proprio']) if self.current_trajectory['proprio'][0] is not None else None,
                'actions': np.stack(self.current_trajectory['actions']),
                'rewards': np.array(self.current_trajectory['rewards']),
                'dones': np.array(self.current_trajectory['dones']),
                'values': np.array(self.current_trajectory['values']),
                'old_log_probs': torch.stack(self.current_trajectory['old_log_probs']).detach(),
                'finish_step': self.episode_step - 1,
                'traj_len': self.episode_step,
            }
            self.trajectories.append(trajectory)
            
            # Reset
            self.current_trajectory = {k: [] for k in self.current_trajectory.keys()}
            self.episode_step = 0
    
    def generate_traj_mask(self, traj_len: int, finish_step: int, device: torch.device) -> torch.Tensor:
        """
        Generate trajectory mask from finish_step.
        
        Args:
            traj_len: Length of trajectory (may include padding)
            finish_step: Index of final valid step
            device: Device for tensor
        
        Returns:
            mask: Boolean mask, shape (traj_len,), True for valid steps
        """
        mask = torch.zeros(traj_len, dtype=torch.bool, device=device)
        mask[:finish_step + 1] = True
        return mask
    
    def compute_advantages(
        self,
        gamma: float = 0.99,
        verifier_gamma: float = 1.0,
    ):
        """
        Compute GRPO advantages for all trajectories.
        
        Uses sparse rewards at finish_step only.
        GRPO: A[i] = sum(gamma^t * r[t] for t in [i, finish_step])
        
        Args:
            gamma: Discount factor (unused in GRPO with sparse rewards)
            verifier_gamma: Verifier discount (1.0 for no discounting)
        """
        for traj in self.trajectories:
            traj_len = traj['traj_len']
            finish_step = traj['finish_step']
            rewards = traj['rewards']
            
            # Compute returns (reward-to-go from each step)
            returns = np.zeros(traj_len, dtype=np.float32)
            
            # Only reward at finish_step is non-zero (sparse rewards)
            # Propagate backward with gamma
            returns[finish_step] = rewards[finish_step]
            for t in range(finish_step - 1, -1, -1):
                returns[t] = rewards[t] + verifier_gamma * returns[t + 1]
            
            # GRPO: advantages = returns (no value baseline)
            advantages = returns.copy()
            
            traj['returns'] = returns
            traj['advantages'] = advantages
        
        # Normalize advantages across all trajectories
        all_advantages = np.concatenate([t['advantages'] for t in self.trajectories])
        adv_mean = all_advantages.mean()
        adv_std = all_advantages.std() + 1e-8
        
        for traj in self.trajectories:
            traj['advantages'] = (traj['advantages'] - adv_mean) / adv_std
    
    def get(self) -> Dict[str, Any]:
        """
        Get all stored trajectories as a dictionary.
        
        Returns:
            data: Dictionary containing all trajectory data
        """
        if not self.trajectories:
            return {
                'observations': [],
                'responses': [],
                'input_ids': [],
                'attention_mask': [],
                'pixel_values': [],
                'proprio': [],
                'actions': [],
                'rewards': [],
                'returns': [],
                'advantages': [],
                'old_log_probs': [],
                'finish_steps': [],
                'traj_lens': [],
            }
        
        # Flatten trajectories into single arrays
        data = {
            'observations': [obs for traj in self.trajectories for obs in traj['observations']],
            'responses': torch.cat([traj['responses'] for traj in self.trajectories]),
            'input_ids': torch.cat([traj['input_ids'] for traj in self.trajectories]),
            'attention_mask': torch.cat([traj['attention_mask'] for traj in self.trajectories]),
            'pixel_values': torch.cat([traj['pixel_values'] for traj in self.trajectories]),
            'proprio': np.concatenate([traj['proprio'] for traj in self.trajectories if traj['proprio'] is not None]) if self.trajectories[0]['proprio'] is not None else None,
            'actions': np.concatenate([traj['actions'] for traj in self.trajectories]),
            'rewards': np.concatenate([traj['rewards'] for traj in self.trajectories]),
            'returns': np.concatenate([traj['returns'] for traj in self.trajectories]),
            'advantages': np.concatenate([traj['advantages'] for traj in self.trajectories]),
            'old_log_probs': torch.cat([traj['old_log_probs'] for traj in self.trajectories]),
            'finish_steps': [traj['finish_step'] for traj in self.trajectories],
            'traj_lens': [traj['traj_len'] for traj in self.trajectories],
        }
        
        return data
    
    def __len__(self) -> int:
        """Return total number of complete trajectories."""
        return len(self.trajectories)

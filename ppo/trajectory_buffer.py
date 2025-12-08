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
            'l1_actions': [],  # L1 regression actions (for BC targets)
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
        l1_action: Optional[np.ndarray],  # L1 regression action (for BC)
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
        self.current_trajectory['l1_actions'].append(l1_action)
        self.current_trajectory['rewards'].append(reward)
        self.current_trajectory['dones'].append(done)
        self.current_trajectory['values'].append(value)
        self.current_trajectory['old_log_probs'].append(old_log_prob)
        
        self.episode_step += 1
        
        if done:
            # Finalize trajectory and add to buffer
            # Pad input_ids and attention_mask to max length in trajectory for stacking
            input_ids_list = self.current_trajectory['input_ids']
            attention_mask_list = self.current_trajectory['attention_mask']
            
            # Find max length
            max_len = max(ids.shape[1] for ids in input_ids_list)
            
            # Pad all tensors to max length
            padded_input_ids = []
            padded_attention_masks = []
            for ids, mask in zip(input_ids_list, attention_mask_list):
                pad_len = max_len - ids.shape[1]
                if pad_len > 0:
                    # Pad with 0 for input_ids and 0 for attention_mask (ignore padding)
                    ids = torch.nn.functional.pad(ids, (0, pad_len), value=0)
                    mask = torch.nn.functional.pad(mask, (0, pad_len), value=0)
                padded_input_ids.append(ids)
                padded_attention_masks.append(mask)
            
            # Pad l1_actions to consistent shape before stacking
            if any(a is not None for a in self.current_trajectory['l1_actions']):
                expected_shape = (self.action_chunk, self.action_dim) if hasattr(self, 'action_chunk') and hasattr(self, 'action_dim') else None
                def pad_action(a):
                    if a is None:
                        return np.zeros(expected_shape, dtype=np.float32) if expected_shape else np.zeros_like(self.current_trajectory['l1_actions'][0])
                    if expected_shape and a.shape != expected_shape:
                        pad_width = [(0, expected_shape[i] - a.shape[i]) for i in range(len(expected_shape))]
                        return np.pad(a, pad_width, mode='constant')
                    return a
                padded_actions = [pad_action(a) for a in self.current_trajectory['l1_actions']]
                l1_actions = np.stack(padded_actions)
            else:
                l1_actions = None

            trajectory = {
                'observations': self.current_trajectory['observations'].copy(),
                'responses': torch.stack(self.current_trajectory['responses']).detach(),
                'input_ids': torch.stack(padded_input_ids).detach(),
                'attention_mask': torch.stack(padded_attention_masks).detach(),
                'pixel_values': torch.stack(self.current_trajectory['pixel_values']).detach(),
                'proprio': np.stack(self.current_trajectory['proprio']) if self.current_trajectory['proprio'][0] is not None else None,
                'actions': np.stack(self.current_trajectory['actions']),
                'l1_actions': l1_actions,
                # ...existing code...
            }

            trajectory = {
                # ...existing code...
                'l1_actions': l1_actions,
                # ...existing code...
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
            # Pad input_ids and attention_mask to max length in trajectory for stacking
            input_ids_list = self.current_trajectory['input_ids']
            attention_mask_list = self.current_trajectory['attention_mask']
            
            # Find max length
            max_len = max(ids.shape[1] for ids in input_ids_list)
            
            # Pad all tensors to max length
            padded_input_ids = []
            padded_attention_masks = []
            for ids, mask in zip(input_ids_list, attention_mask_list):
                pad_len = max_len - ids.shape[1]
                if pad_len > 0:
                    # Pad with 0 for input_ids and 0 for attention_mask (ignore padding)
                    ids = torch.nn.functional.pad(ids, (0, pad_len), value=0)
                    mask = torch.nn.functional.pad(mask, (0, pad_len), value=0)
                padded_input_ids.append(ids)
                padded_attention_masks.append(mask)
            
            # Robust l1_actions stacking: only if all are not None and shapes match
            l1_actions_list = self.current_trajectory['l1_actions']
            if l1_actions_list and all(a is not None for a in l1_actions_list):
                shapes = [np.shape(a) for a in l1_actions_list]
                if all(s == shapes[0] for s in shapes):
                    l1_actions = np.stack(l1_actions_list)
                else:
                    l1_actions = None
            else:
                l1_actions = None

            trajectory = {
                'observations': self.current_trajectory['observations'].copy(),
                'responses': torch.stack(self.current_trajectory['responses']).detach(),
                'input_ids': torch.stack(padded_input_ids).detach(),
                'attention_mask': torch.stack(padded_attention_masks).detach(),
                'pixel_values': torch.stack(self.current_trajectory['pixel_values']).detach(),
                'proprio': np.stack(self.current_trajectory['proprio']) if self.current_trajectory['proprio'][0] is not None else None,
                'actions': np.stack(self.current_trajectory['actions']),
                'l1_actions': l1_actions,
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
            # For sparse binary rewards (0 or 1), use ABSOLUTE advantages
            # This avoids negative advantages that cause policy instability
            advantages = returns.copy()
            
            traj['returns'] = returns
            traj['advantages'] = advantages
        
        # Collect all advantages for statistics (but DON'T normalize for sparse rewards)
        all_advantages = np.concatenate([t['advantages'] for t in self.trajectories])
        
        # Check for NaN or inf in advantages
        if np.any(np.isnan(all_advantages)) or np.any(np.isinf(all_advantages)):
            print(f"⚠️  WARNING: Found NaN or inf in advantages before normalization!")
            print(f"   NaN count: {np.isnan(all_advantages).sum()}")
            print(f"   Inf count: {np.isinf(all_advantages).sum()}")
            # Replace NaN/inf with zeros
            all_advantages = np.nan_to_num(all_advantages, nan=0.0, posinf=0.0, neginf=0.0)
        
        adv_mean = all_advantages.mean()
        adv_std = all_advantages.std()
        
        # Debug: Print advantage statistics
        print(f"\n📊 Advantage Statistics (ABSOLUTE - No Normalization):")
        print(f"   Mean: {adv_mean:.6f}")
        print(f"   Std: {adv_std:.6f}")
        print(f"   Min: {all_advantages.min():.6f}")
        print(f"   Max: {all_advantages.max():.6f}")
        print(f"   Total samples: {len(all_advantages)}")
        
        # CRITICAL: For sparse binary rewards (0 or 1), DO NOT normalize!
        # Normalization creates negative advantages which confuse the policy.
        # Instead, use absolute advantages:
        #   - Successful trajectories: advantage = 1.0 (increase log prob)
        #   - Failed trajectories: advantage = 0.0 (no gradient)
        print(f"\n✓ Using ABSOLUTE advantages (no normalization) for sparse rewards")
        print(f"  - Successful steps: advantage ≈ {all_advantages.max():.2f}")
        print(f"  - Failed steps: advantage ≈ {all_advantages.min():.2f}")
        print(f"  - This ensures policy only increases prob of successful actions\n")
        
        # Final safety check
        for traj in self.trajectories:
            if np.any(np.isnan(traj['advantages'])) or np.any(np.isinf(traj['advantages'])):
                print(f"⚠️  ERROR: NaN/inf in advantages! Setting to zeros.")
                traj['advantages'] = np.nan_to_num(traj['advantages'], nan=0.0, posinf=0.0, neginf=0.0)
    
    def get(self) -> Dict[str, Any]:
        """
        Get all stored trajectories as a dictionary.
        Handles partial chunks robustly: only executed actions are returned, and variable-length chunks are supported.
        Returns:
            data: Dictionary containing all trajectory data
        """
        import torch
        import numpy as np
        if not self.trajectories:
            return {
                'observations': [],
                'responses': torch.empty(0),
                'input_ids': torch.empty(0, dtype=torch.long),
                'attention_mask': torch.empty(0, dtype=torch.long),
                'pixel_values': torch.empty(0),
                'proprio': np.empty((0,)),
                'actions': np.empty((0,)),
                'l1_actions': np.empty((0,)),
                'rewards': np.empty((0,)),
                'returns': np.empty((0,)),
                'advantages': np.empty((0,)),
                'old_log_probs': torch.empty(0),
                'finish_steps': [],
                'traj_lens': []
            }
                        'old_log_probs': torch.empty(0),
                        'finish_steps': [],
                        'traj_lens': []
                    }
        """
        Get all stored trajectories as a dictionary.
        Handles partial chunks robustly: only executed actions are returned, and variable-length chunks are supported.
        Returns:
            data: Dictionary containing all trajectory data
        """

        if not self.trajectories:
            import torch
            import numpy as np
            return {
                'observations': [],
                'responses': torch.empty(0),
                'input_ids': torch.empty(0, dtype=torch.long),
                'attention_mask': torch.empty(0, dtype=torch.long),
                'pixel_values': torch.empty(0),
                'proprio': np.empty((0,)),
                'actions': np.empty((0,)),
                'l1_actions': np.empty((0,)),
                'rewards': np.empty((0,)),
                'returns': np.empty((0,)),
                'advantages': np.empty((0,)),
                'old_log_probs': torch.empty(0),
                'finish_steps': [],
                'traj_lens': []
            }

        # Pad sequence tensors to max length for cat
        def pad_sequence_list(seq_list, pad_value=0):
            # seq_list: list of tensors of the same rank, but possibly different shape in any dimension
            # Pad all dimensions to the max size in that dimension across the batch
            # Example: for 3D tensors, pad (D0, D1, D2) to (max0, max1, max2)
            import torch
            max_shape = []
            ndims = seq_list[0].ndim
            for d in range(ndims):
                max_shape.append(max(x.shape[d] for x in seq_list))
            padded = []
            for x in seq_list:
                pad_widths = []
                for d in reversed(range(ndims)):
                    pad_amt = max_shape[d] - x.shape[d]
                    pad_widths.extend([0, pad_amt])  # pad only on the right
                x = torch.nn.functional.pad(x, pad_widths, value=pad_value)
                padded.append(x)
            return torch.stack(padded, dim=0)

            trajectory = {
                'observations': self.current_trajectory['observations'].copy(),
                'responses': torch.stack(self.current_trajectory['responses']).detach(),
                'input_ids': torch.stack(padded_input_ids).detach(),
                'attention_mask': torch.stack(padded_attention_masks).detach(),
                'pixel_values': torch.stack(self.current_trajectory['pixel_values']).detach(),
                'proprio': np.stack(self.current_trajectory['proprio']) if self.current_trajectory['proprio'][0] is not None else None,
                'actions': np.stack(self.current_trajectory['actions']),
                'l1_actions': l1_actions,
                'rewards': np.array(self.current_trajectory['rewards']),
                'dones': np.array(self.current_trajectory['dones']),
                'values': np.array(self.current_trajectory['values']),
                'old_log_probs': torch.stack(self.current_trajectory['old_log_probs']).detach(),
                'finish_step': self.episode_step - 1,
                'traj_len': self.episode_step
            }
            self.trajectories.append(trajectory)
    

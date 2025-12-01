"""
OpenVLA PPO Training for LIBERO Spatial Tasks

Integrates OpenVLA actor with PPO for policy fine-tuning in LIBERO environments.
Adds a lightweight value head critic for advantage estimation.

Usage:
    # Single task training (quick test)
    python OpenVLA_PPO.py --task-suite libero_spatial --task-id 0 --timesteps 10000
    
    # Multi-task training (4 parallel environments)
    python OpenVLA_PPO.py --task-suite libero_spatial --task-ids 0 1 2 3 --num-envs 4 --timesteps 100000
    
    # Multi-task with multi-GPU setup
    python OpenVLA_PPO.py --task-ids 0 1 2 3 --use-multi-gpu --timesteps 200000
"""
import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm

# Suppress all warnings for clean console output
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Configure tqdm for clean output
tqdm.monitor_interval = 0  # Disable monitor thread warnings

# # Add vla-oft to path for imports
# vla_oft_path = Path(__file__).parent / "vla-oft"
# sys.path.insert(0, str(vla_oft_path))

from vla_oft.min_vla.config import OpenVLAActorConfig
from vla_oft.min_vla.actor import OpenVLAActor
from vla_oft.min_vla.action_tokenizer import ActionTokenizer
from ppo.config import PPOConfig
from ppo.trajectory_buffer import TrajectoryBuffer
from ppo.core_algos import logprobs_from_logits, compute_policy_loss, apply_mask_with_grad_control
from libero_rl.utils.obs_utils import process_observation_for_vla
from libero_rl.utils.task_utils import get_task, get_max_episode_length, get_all_task_names


class OpenVLAPPO:
    """
    PPO trainer for OpenVLA actor using GRPO (value-free advantages).
    
    Integrates:
    - OpenVLAActor from vla-oft/min_vla
    - TrajectoryBuffer with GRPO advantages
    - LIBERO utils from libero_rl/utils
    
    Note: GRPO (Group Relative Policy Optimization) computes advantages
    directly from sparse rewards without requiring a value function.
    """
    
    def __init__(
        self,
        vla_config: OpenVLAActorConfig,
        ppo_config: PPOConfig,
    ):
        self.cfg = ppo_config
        self.vla_config = vla_config
        
        # Respect multi-GPU configuration from vla_config
        if vla_config.use_multi_gpu:
            self.device = torch.device(vla_config.device)  # GPU 0 for VLA backbone
            self.training_device = torch.device(vla_config.training_device)  # GPU 1 for training
            print(f"Multi-GPU Setup:")
            print(f"  - VLA backbone on {self.device}")
            print(f"  - Training components on {self.training_device}")
        else:
            self.device = torch.device(ppo_config.device)
            self.training_device = self.device
            print(f"Single-GPU Setup: All components on {self.device}")
        
        # Initialize OpenVLA actor
        print("Initializing OpenVLA actor...")
        self.actor = OpenVLAActor(vla_config)
        
        # Enable gradient checkpointing to save memory during training
        if hasattr(self.actor.vla.language_model, 'gradient_checkpointing_enable'):
            self.actor.vla.language_model.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing for memory efficiency")
        
        # Initialize action tokenizer
        print("Initializing action tokenizer...")
        self.action_tokenizer = ActionTokenizer(
            vocab_size=32000,  # LLaMA vocab size
            n_bins=256,
            min_action=-1.0,
            max_action=1.0,
        )
        
        # Add bin_centers and vocab_size to VLA model for compatibility
        self.actor.vla.bin_centers = self.action_tokenizer.bin_centers
        self.actor.vla.vocab_size = self.action_tokenizer.vocab_size
        print(f"  {self.action_tokenizer}")
        

        # Apply LoRA or freeze VLA backbone based on config
        if vla_config.use_lora and not vla_config.freeze_vla_backbone:
            print("\n" + "="*70)
            print("Applying LoRA Adapters to VLA Model")
            print("="*70)
            
            try:
                from peft import LoraConfig, get_peft_model
                
                # Configure LoRA (following openvla-oft finetune.py)
                lora_config = LoraConfig(
                    r=vla_config.lora_rank,
                    lora_alpha=min(vla_config.lora_rank, 16),  # Cap at 16 like in finetune.py
                    lora_dropout=vla_config.lora_dropout,
                    target_modules="all-linear",  # Apply to all linear layers
                    init_lora_weights="gaussian",
                )
                
                # Apply LoRA to entire VLA model (not just language_model)
                self.actor.vla = get_peft_model(self.actor.vla, lora_config)
                
                # Print trainable parameters
                self.actor.vla.print_trainable_parameters()
                
                print(f"LoRA Configuration:")
                print(f"  - Rank (r): {vla_config.lora_rank}")
                print(f"  - Alpha (α): {min(vla_config.lora_rank, 16)}")
                print(f"  - Dropout: {vla_config.lora_dropout}")
                print(f"  - Target: all-linear layers")
                print("="*70 + "\n")
                
            except ImportError:
                print("WARNING: peft library not found!")
                print("    Install with: pip install peft")
                print("    Falling back to freezing VLA backbone")
                print("="*70 + "\n")
                vla_config.freeze_vla_backbone = True
                vla_config.use_lora = False
        
        if vla_config.freeze_vla_backbone:
            print("\n" + "="*70)
            print("Freezing VLA Backbone (Vision + Language Model)")
            print("="*70)
            
            # Freeze vision backbone
            for param in self.actor.vla.vision_backbone.parameters():
                param.requires_grad = False
            
            # Freeze language model
            for param in self.actor.vla.language_model.parameters():
                param.requires_grad = False
            
            # Keep trainable: action_head, proprio_projector
            trainable_components = []
            if self.actor.action_head:
                trainable_components.append("action_head")
            if self.actor.proprio_projector:
                trainable_components.append("proprio_projector")
            
            print(f"Frozen: vision_backbone, language_model")
            print(f"Trainable: {', '.join(trainable_components)}")
            print("="*70 + "\n")
            
            # Set VLA to eval mode (no dropout, batchnorm in eval mode)
            self.actor.vla.eval()
        
        # Verify configuration for PPO training
        if not vla_config.use_tokenized_actions:
            raise ValueError(
                "PPO requires use_tokenized_actions=True. "
                "Set this in OpenVLAActorConfig for RL training."
            )
        
        # Warn if L1 action head is loaded (not needed for PPO)
        if self.actor.l1_action_head is not None:
            print("\n" + "="*70)
            print("WARNING: L1 Regression Action Head Loaded")
            print("="*70)
            print("The L1 regression head is loaded but PPO uses tokenized actions.")
            print("The L1 head will NOT be used for action prediction or training.")
            print("")
            print("To save memory (~668MB), set in config:")
            print("  load_l1_action_head = False")
            print("")
            print("The L1 head is only needed for:")
            print("  - Supervised learning with L1 loss")
            print("  - Inference comparison with original checkpoint")
            print("="*70 + "\n")
        
        # Initialize optimizer for trainable VLA components
        # Note: L1 action head is explicitly EXCLUDED from optimizer for PPO
        # Even if loaded, it won't be trained or used
        vla_trainable_params = [p for p in self.actor.vla.parameters() if p.requires_grad]
        proprio_proj_params = list(self.actor.proprio_projector.parameters()) if self.actor.proprio_projector else []
        
        # Only include L1 head if explicitly not frozen and loaded (rare case for hybrid training)
        l1_head_params = []
        if (self.actor.l1_action_head is not None and 
            not vla_config.freeze_l1_action_head and 
            vla_config.load_l1_action_head):
            l1_head_params = list(self.actor.l1_action_head.parameters())
            print("Including L1 action head in optimizer (hybrid training mode)")
        
        actor_params = vla_trainable_params + proprio_proj_params + l1_head_params
        self.actor_optimizer = optim.AdamW(actor_params, lr=ppo_config.actor_lr)
        self.max_grad_norm = ppo_config.max_grad_norm
        
        # Trajectory buffer for storing complete episodes
        self.trajectory_buffer = TrajectoryBuffer()
        
        # Reference policy for KL penalty (frozen copy of actor)
        if ppo_config.kl_coef > 0:
            print("Creating reference policy for KL penalty...")
            import copy
            self.ref_vla = copy.deepcopy(self.actor.vla)
            self.ref_vla.eval()
            # Freeze reference policy
            for param in self.ref_vla.parameters():
                param.requires_grad = False
            print("  Reference policy created (frozen)")
        else:
            self.ref_vla = None
        
        # Training stats
        self.global_step = 0
        self.episode_count = 0
        
        # Multi-task tracking
        self.is_vectorized = ppo_config.num_envs > 1
        
        print(f"Initialized OpenVLA-PPO (GRPO - Value-Free Advantages)")
        print(f"  - Mode: {'Multi-task vectorized' if self.is_vectorized else 'Single-task'}")
        print(f"  - Num envs: {ppo_config.num_envs}")
        print(f"  - Actor trainable params: {sum(p.numel() for p in actor_params):,}")
        print(f"  - Action prediction: Tokenized (PPO mode)")
        print(f"  - Action bins: {self.action_tokenizer.n_bins}")
        print(f"  - L1 regression head: {'Loaded ({})'.format('frozen' if vla_config.freeze_l1_action_head else 'trainable') if self.actor.l1_action_head else 'Not loaded'}")
        print(f"  - Rollout temperature: {ppo_config.rollout_temperature}")
        print(f"  - Clip ratios: high={ppo_config.clip_ratio_high}, low={ppo_config.clip_ratio_low}")
        print(f"  - GRPO gamma: {ppo_config.verifier_gamma}")
        print(f"  - KL coefficient: {ppo_config.kl_coef}")
        print(f"  - Device mode: {'Multi-GPU' if vla_config.use_multi_gpu else 'Single-GPU'}")
        if vla_config.use_multi_gpu:
            print(f"    * VLA backbone ({vla_config.device}): {'LoRA adapters' if vla_config.use_lora else 'frozen'}")
            print(f"    * Training components ({vla_config.training_device})")
    

    
    def get_action(
        self,
        obs: Dict[str, Any],
        task_prompt: str,
        temperature: float = 1.6,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Dict[str, Any]]:
        """
        Get action and log_prob from policy using tokenized action prediction.
        
        This method uses the VLA's language model logits to predict action tokens,
        which are then detokenized to continuous actions. This is the correct
        approach for PPO training (not L1 regression).
        
        Args:
            obs: Observation dictionary with 'image' and 'proprio'
            task_prompt: Task description string
            temperature: Sampling temperature for action tokens
                        (1.6 for rollout exploration, 0.0 for greedy eval)
        
        Returns:
            action: (action_dim,) numpy array (continuous action for env)
            info: Dictionary containing:
                - log_prob: log probability tensor (sum over action dims)
                - responses: action token IDs tensor
                - input_ids, attention_mask, pixel_values: for replay
                - proprio: proprioception array
        """
        # Verify we're using tokenized actions (required for PPO)
        if not self.vla_config.use_tokenized_actions:
            raise RuntimeError(
                "get_action() requires use_tokenized_actions=True. "
                "This is required for PPO training."
            )
        
        # Get action token logits with gradients
        action_data = self.predict_action_tokens_with_grad(
            obs, task_prompt, temperature=temperature, sample=True
        )
        
        # Extract continuous action for environment
        action = action_data['continuous_action']
        
        # Compile info dictionary
        info = {
            'log_prob': action_data['log_prob'],
            'responses': action_data['responses'],
            'input_ids': action_data['input_ids'],
            'attention_mask': action_data['attention_mask'],
            'pixel_values': action_data['pixel_values'],
            'proprio': obs.get('proprio', None),
        }
        
        return action, info
    
    def predict_action_tokens_with_grad(
        self,
        obs: Dict[str, Any],
        task_prompt: str,
        temperature: float = 1.6,
        sample: bool = True,
    ) -> Dict[str, Any]:
        """
        Get action token predictions with gradients enabled for PPO training.
        
        This is the core action prediction method for PPO. It:
        1. Runs full VLA forward pass (vision + language model)
        2. Extracts logits for action tokens (last 256 vocab tokens)
        3. Samples or argmaxes token IDs
        4. Detokenizes to continuous actions for environment
        5. Computes log probabilities for PPO loss
        
        Note: This does NOT use the L1 regression action head. Actions are
        predicted purely through the language model's token logits.
        
        Args:
            obs: Observation with 'image' and 'proprio'
            task_prompt: Task description
            temperature: Sampling temperature (1.6 for rollout, 0 for greedy)
            sample: If True, sample from distribution; if False, use argmax
        
        Returns:
            Dictionary containing:
                - logits: Action token logits, shape (action_dim * action_chunk, 256)
                - responses: Sampled token IDs, shape (action_dim * action_chunk,)
                - log_prob: Log probability tensor (sum over action dims)
                - continuous_action: Detokenized action, shape (action_dim,)
                - input_ids, attention_mask, pixel_values: For replay
        """
        # Import constants
        from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, IGNORE_INDEX
        
        # Prepare observation
        image = obs["image"]
        proprio = obs.get("proprio", None)
        
        # Clip proprio to expected dimension
        if proprio is not None and len(proprio) > 8:
            proprio = proprio[:8]
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            from PIL import Image
            image = Image.fromarray(image)
        
        # Build prompt
        prompt = f"In: What action should the robot take to {task_prompt.lower()}?\nOut:"
        
        # Process inputs
        inputs = self.actor.processor(prompt, image).to(self.device, dtype=torch.bfloat16)
        
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]
        attention_mask = inputs["attention_mask"]
        
        # Add empty token if needed (from predict_action)
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )
        
        # Create fake labels
        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX
        
        # Get number of prompt tokens
        NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1
        
        # Prepare inputs (add action tokens)
        input_ids, attention_mask = self.actor.vla._prepare_input_for_action_prediction(input_ids, attention_mask)
        labels = self.actor.vla._prepare_labels_for_action_prediction(labels, input_ids)
        
        # Get input embeddings and action masks
        input_embeddings = self.actor.vla.get_input_embeddings()(input_ids)
        all_actions_mask = self.actor.vla._process_action_masks(labels)
        
        # Extract language embeddings
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )
        
        # Process vision features
        use_film = False  # Our model doesn't use FiLM
        projected_patch_embeddings = self.actor.vla._process_vision_features(pixel_values, language_embeddings, use_film)
        
        # Add proprio if available
        if self.actor.proprio_projector is not None and proprio is not None:
            proprio_tensor = torch.from_numpy(proprio).to(projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype)
            projected_patch_embeddings = self.actor.vla._process_proprio_features(
                projected_patch_embeddings, proprio_tensor, self.actor.proprio_projector
            )
        
        # Calculate number of patches
        NUM_PATCHES = self.actor.vla.vision_backbone.get_num_patches() * self.actor.vla.vision_backbone.get_num_images_in_input()
        if self.actor.proprio_projector is not None and proprio is not None:
            NUM_PATCHES += 1
        
        # Zero out action token embeddings
        all_actions_mask = all_actions_mask.unsqueeze(-1)
        input_embeddings = input_embeddings * ~all_actions_mask
        
        # Build multimodal embeddings
        multimodal_embeddings, multimodal_attention_mask = self.actor.vla._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )
        
        # Forward through language model
        language_model_output = self.actor.vla.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Extract logits for action tokens
        # Get logits at action token positions
        action_logits = language_model_output.logits[
            :,
            NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK,
            :,
        ]
        
        # Extract last 256 tokens (action vocabulary)
        # Following reference: logits[..., -256-64:-64]
        action_token_logits = action_logits[..., -256-64:-64]  # Shape: (batch, seq_len, 256)
        
        # Sample or take argmax
        if sample and temperature > 0:
            # Apply temperature and sample
            scaled_logits = action_token_logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            
            # Sample token indices [0, 255]
            probs_flat = probs.reshape(-1, probs.shape[-1])
            sampled_indices_flat = torch.multinomial(probs_flat, num_samples=1)
            sampled_indices = sampled_indices_flat.view(action_token_logits.shape[0], -1)
            
            # Convert to vocab token IDs (last 256 tokens)
            responses = sampled_indices + (self.action_tokenizer.vocab_size - 256)
        else:
            # Greedy decoding (temperature=0 or sample=False)
            sampled_indices = action_token_logits.argmax(dim=-1)
            responses = sampled_indices + (self.action_tokenizer.vocab_size - 256)
        
        # Compute log probabilities
        log_probs_per_token = logprobs_from_logits(action_token_logits, sampled_indices)
        log_prob = log_probs_per_token.sum(dim=-1)  # Sum over action dimensions
        
        # Detokenize to continuous action
        responses_np = responses[0].detach().cpu().numpy()  # (action_dim * action_chunk,)
        continuous_actions = self.action_tokenizer.detokenize_actions(responses_np)
        continuous_actions = continuous_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
        
        # Take first action from chunk
        continuous_action = continuous_actions[0]  # (action_dim,)
        
        return {
            'logits': action_token_logits,
            'responses': responses[0],  # (action_dim * action_chunk,)
            'log_prob': log_prob[0],  # Scalar tensor
            'continuous_action': continuous_action,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
        }
    
    def collect_rollouts(
        self,
        env,
        task_prompts: Union[str, List[str]],
    ) -> Dict[str, float]:
        """
        Collect trajectory-based rollouts with sparse rewards.
        
        Collects complete episodes, assigns sparse rewards only at episode completion,
        and stores action token IDs for PPO training.
        
        Args:
            env: Single or vectorized LIBERO environment
            task_prompts: Single task prompt string or list of prompts (one per env)
        """
        self.actor.vla.eval()
        
        obs, info = env.reset()
        self.trajectory_buffer.clear()
        
        # Convert single prompt to list for uniform handling
        if isinstance(task_prompts, str):
            task_prompts = [task_prompts]
        
        # Episode tracking
        episode_rewards = []
        episode_lengths = []
        episode_successes = []
        
        if self.is_vectorized:
            current_episode_lengths = np.zeros(self.cfg.num_envs, dtype=int)
        else:
            current_episode_length = 0
        
        # Collect n_steps or until enough complete trajectories
        steps_collected = 0
        
        pbar = tqdm(
            total=self.cfg.n_steps,
            desc="Collecting rollouts",
            leave=False,
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        # Use torch.no_grad() during rollout to save memory
        with torch.no_grad():
            while steps_collected < self.cfg.n_steps:
                if self.is_vectorized:
                    # Vectorized environment - process all envs
                    for env_idx in range(self.cfg.num_envs):
                        # Process observation
                        processed_obs = process_observation_for_vla(
                            obs[env_idx],
                            camera_name="agentview",
                            resize_size=self.cfg.image_size,
                        )
                        
                        actor_obs = {
                            "image": processed_obs["image"],
                            "proprio": processed_obs["robot_state"],
                        }
                        
                        # Get action with tokenization
                        action, action_info = self.get_action(
                            actor_obs, 
                            task_prompts[env_idx],
                            temperature=self.cfg.rollout_temperature,
                        )
                        
                        # Step environment
                        next_obs, reward, terminated, truncated, infos = env.step(action)
                        done = terminated or truncated
                        
                        # Sparse rewards: zero intermediate rewards, only at finish
                        sparse_reward = 0.0
                        if done:
                            success = infos.get("success", [False] * self.cfg.num_envs)[env_idx]
                            sparse_reward = 1.0 if success else 0.0
                            episode_successes.append(float(success))
                            episode_lengths.append(current_episode_lengths[env_idx] + 1)
                            current_episode_lengths[env_idx] = 0
                        else:
                            current_episode_lengths[env_idx] += 1
                        
                        # Add to trajectory buffer
                        self.trajectory_buffer.add(
                            obs=actor_obs,
                            responses=action_info['responses'],
                            input_ids=action_info['input_ids'],
                            attention_mask=action_info['attention_mask'],
                            pixel_values=action_info['pixel_values'],
                            proprio=action_info['proprio'],
                            action=action,
                            reward=sparse_reward,
                            done=done,
                            old_log_prob=action_info['log_prob'],
                        )
                        
                        steps_collected += 1
                        pbar.update(1)
                        pbar.set_postfix({
                            'eps': len(episode_successes),
                            'succ': f"{np.mean(episode_successes)*100:.1f}%" if episode_successes else "0.0%",
                            'len': f"{np.mean(episode_lengths):.0f}" if episode_lengths else "0"
                        }, refresh=False)
                        
                    obs = next_obs
                    self.global_step += self.cfg.num_envs
                    
                else:
                    # Single environment
                    processed_obs = process_observation_for_vla(
                        obs,
                        camera_name="agentview",
                        resize_size=self.cfg.image_size,
                    )
                    
                    actor_obs = {
                        "image": processed_obs["image"],
                        "proprio": processed_obs["robot_state"],
                    }
                    
                    # Get action with tokenization
                    action, action_info = self.get_action(
                        actor_obs,
                        task_prompts[0],
                        temperature=self.cfg.rollout_temperature,
                    )
                    
                    # Step environment
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    # Sparse rewards: zero intermediate rewards, only at finish
                    sparse_reward = 0.0
                    if done:
                        success = info.get("success", False)
                        sparse_reward = 1.0 if success else 0.0
                        episode_successes.append(float(success))
                        episode_lengths.append(current_episode_length + 1)
                        current_episode_length = 0
                        obs, info = env.reset()
                    else:
                        current_episode_length += 1
                        obs = next_obs
                    
                    # Add to trajectory buffer
                    self.trajectory_buffer.add(
                        obs=actor_obs,
                        responses=action_info['responses'],
                        input_ids=action_info['input_ids'],
                        attention_mask=action_info['attention_mask'],
                        pixel_values=action_info['pixel_values'],
                        proprio=action_info['proprio'],
                        action=action,
                        reward=sparse_reward,
                        done=done,
                        value=0.0,  # GRPO mode: must provide value even if unused
                        old_log_prob=action_info['log_prob'],
                    )
                    
                    steps_collected += 1
                    pbar.update(1)
                    if done:
                        pbar.set_postfix({
                            'eps': len(episode_successes),
                            'succ': f"{np.mean(episode_successes)*100:.1f}%" if episode_successes else "0.0%",
                            'len': f"{np.mean(episode_lengths):.0f}" if episode_lengths else "0"
                        }, refresh=False)
                    self.global_step += 1
        
        pbar.close()
        
        # Finalize any partial trajectories
        self.trajectory_buffer.finalize_partial_trajectory()
        
        # Compute GRPO advantages
        self.trajectory_buffer.compute_advantages(
            gamma=self.cfg.gamma,
            verifier_gamma=self.cfg.verifier_gamma,
        )
        
        # Compute statistics
        stats = {
            "rollout/mean_reward": np.mean(episode_successes) if episode_successes else 0.0,  # Success rate as reward
            "rollout/mean_length": np.mean(episode_lengths) if episode_lengths else 0.0,
            "rollout/success_rate": np.mean(episode_successes) if episode_successes else 0.0,
            "rollout/num_episodes": len(episode_successes),
            "rollout/num_trajectories": len(self.trajectory_buffer),
        }
        
        return stats
    
    def update_policy(self, task_prompt: str) -> Dict[str, float]:
        """
        Update policy using PPO with GRPO advantages and gradient accumulation.
        
        Implements PPO policy gradient with:
        - Action token log probabilities
        - Asymmetric clipped surrogate objective
        - GRPO advantages (value-free)
        - Per-sample gradient accumulation
        - Optional KL penalty from reference policy
        
        Note: GRPO computes advantages directly from sparse rewards,
        no value function needed.
        """
        self.actor.vla.train()
        # L1 action head stays in eval mode (not used for PPO)
        if self.actor.l1_action_head is not None:
            self.actor.l1_action_head.eval()
        
        # Get data from trajectory buffer
        data = self.trajectory_buffer.get()
        
        if len(data['observations']) == 0:
            return {"train/no_data": 1.0}
        
        # Keep data on rollout device initially
        # Move all data tensors to training_device for safe indexing
        advantages = torch.FloatTensor(data["advantages"]).to(self.training_device)
        returns = torch.FloatTensor(data["returns"]).to(self.training_device)
        old_log_probs = data["old_log_probs"].to(self.training_device)
        responses = data["responses"].to(self.training_device)
        
        # Training statistics
        stats = defaultdict(list)
        
        # Multiple epochs of optimization
        epoch_pbar = tqdm(
            range(self.cfg.n_epochs),
            desc="Policy update",
            leave=False,
            ncols=100,
            bar_format='{l_bar}{bar}| Epoch {n_fmt}/{total_fmt}'
        )
        
        for epoch in epoch_pbar:
            # Generate random minibatch indices
            indices = torch.randperm(len(advantages))
            
            # Calculate number of minibatches
            num_minibatches = (len(advantages) + self.cfg.batch_size - 1) // self.cfg.batch_size
            
            # Progress bar for minibatches within epoch
            minibatch_pbar = tqdm(
                range(0, len(advantages), self.cfg.batch_size),
                desc=f"  Epoch {epoch+1} minibatches",
                leave=False,
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} batches',
                total=num_minibatches
            )
            
            # Process minibatches
            for start_idx in minibatch_pbar:
                end_idx = min(start_idx + self.cfg.batch_size, len(advantages))
                mb_indices = indices[start_idx:end_idx]
                
                # Ensure all minibatch tensors are on training_device
                # mb_indices must remain on CPU for indexing
                mb_advantages = advantages[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_responses = responses[mb_indices]
                
                # ==================== ACTOR UPDATE (PPO Policy Gradient) ====================
                # Batch process samples with gradient accumulation
                self.actor_optimizer.zero_grad()
                
                total_policy_loss = 0.0
                total_clipfrac = 0.0
                total_approx_kl = 0.0
                
                # Collect all action data in a batch to minimize GPU transfers
                batch_logits = []
                batch_log_probs = []
                
                for i, idx in enumerate(mb_indices):
                    obs = data["observations"][idx.item()]
                    
                    # Get action token logits from current policy
                    action_data = self.predict_action_tokens_with_grad(
                        obs, task_prompt, temperature=self.cfg.rollout_temperature, sample=False
                    )
                    
                    # Extract logits for the 256 action tokens
                    logits = action_data['logits'][0]  # (seq_len, 256)
                    batch_logits.append(logits)
                    
                    # Clear intermediate data
                    del action_data
                
                # Compute losses in batch for efficiency
                accumulated_loss = 0.0
                for i, (logits, idx) in enumerate(zip(batch_logits, mb_indices)):
                    response = mb_responses[i]
                    old_log_prob = mb_old_log_probs[i]
                    advantage = mb_advantages[i]
                    
                    # Map response tokens to indices in [0, 255]
                    response_indices = response - (self.action_tokenizer.vocab_size - 256)
                    
                    # Ensure logits and response_indices are on the same device
                    logits = logits.to(self.training_device)
                    response_indices = response_indices.to(self.training_device)
                    log_prob = logprobs_from_logits(logits.unsqueeze(0), response_indices.unsqueeze(0))
                    log_prob = log_prob.sum()  # Sum over action dimensions
                    
                    # Compute PPO loss for this sample
                    ratio = torch.exp(log_prob - old_log_prob)
                    clip_high = torch.clamp(ratio, 1 - self.cfg.clip_ratio_low, 1 + self.cfg.clip_ratio_high)
                    clip_low = torch.clamp(ratio, 1 - self.cfg.clip_ratio_high, 1 + self.cfg.clip_ratio_low)
                    clipped_ratio = torch.where(advantage > 0, clip_high, clip_low)
                    policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
                    
                    # Accumulate loss (normalize by batch size for gradient averaging)
                    accumulated_loss += policy_loss / len(mb_indices)
                    
                    # Track statistics
                    total_policy_loss += policy_loss.item()
                    with torch.no_grad():
                        clipfrac = ((ratio - clipped_ratio).abs() > 1e-6).float().mean()
                        approx_kl = (old_log_prob - log_prob).mean()
                        total_clipfrac += clipfrac.item()
                        total_approx_kl += approx_kl.item()
                
                # Single backward pass for entire minibatch
                accumulated_loss.backward()
                
                # Clear batch data
                del batch_logits, accumulated_loss
                torch.cuda.empty_cache()
                
                # Gradient clipping
                actor_params = (
                    [p for p in self.actor.vla.parameters() if p.requires_grad] +
                    (list(self.actor.l1_action_head.parameters()) if self.actor.l1_action_head and not self.vla_config.freeze_l1_action_head else []) +
                    (list(self.actor.proprio_projector.parameters()) if self.actor.proprio_projector else [])
                )
                nn.utils.clip_grad_norm_(actor_params, self.max_grad_norm)
                
                # Optimizer step
                self.actor_optimizer.step()
                
                # Track averaged statistics for this minibatch
                n_samples = len(mb_indices)
                stats['train/policy_loss'].append(total_policy_loss / n_samples)
                stats['train/clipfrac'].append(total_clipfrac / n_samples)
                stats['train/approx_kl'].append(total_approx_kl / n_samples)
                
                # Update minibatch progress bar with current stats
                minibatch_pbar.set_postfix({
                    'loss': f"{total_policy_loss / n_samples:.4f}",
                    'clip': f"{total_clipfrac / n_samples:.3f}",
                }, refresh=False)
                
                # Clear CUDA cache after each minibatch to prevent fragmentation
                torch.cuda.empty_cache()
            
            minibatch_pbar.close()
            
            # Update epoch progress bar with average stats
            epoch_pbar.set_postfix({
                'avg_loss': f"{np.mean([s for s in stats['train/policy_loss'][-num_minibatches:]]):.4f}",
                'avg_clip': f"{np.mean([s for s in stats['train/clipfrac'][-num_minibatches:]]):.3f}",
            }, refresh=False)
        
        epoch_pbar.close()
        # ...existing code...
        return {k: np.mean(v) for k, v in stats.items()}
    
    def validate(self, env, task_prompt: str) -> Dict[str, float]:
        """
        Run validation episodes to measure success rate.
        
        Uses greedy (argmax) action selection for deterministic evaluation.
        """
        # Set to eval mode first
        self.actor.vla.eval()
        if self.actor.l1_action_head is not None:
            self.actor.l1_action_head.eval()
        
        # Aggressive memory cleanup before validation to prevent OOM
        import gc
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        
        val_rewards = []
        val_successes = []
        
        for ep in tqdm(
            range(self.cfg.val_episodes),
            desc="Validation",
            leave=False,
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} episodes'
        ):
            obs, info = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Process observation
                processed_obs = process_observation_for_vla(
                    obs,
                    camera_name="agentview",
                    resize_size=self.cfg.image_size,
                )
                actor_obs = {
                    "image": processed_obs["image"],
                    "proprio": processed_obs["robot_state"],
                }
                
                # Get action with greedy selection (eval_temperature=0)
                action, action_info = self.get_action(
                    actor_obs,
                    task_prompt,
                    temperature=self.cfg.eval_temperature,
                )
                
                # Step
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            val_rewards.append(episode_reward)
            val_successes.append(float(info.get("success", False)))
            
            # Clear cache frequently during validation to prevent memory buildup
            if (ep + 1) % 2 == 0:  # Every 2 episodes
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        # Final cleanup after validation
        import gc
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        
        return {
            "val/mean_reward": np.mean(val_rewards),
            "val/success_rate": np.mean(val_successes),
        }
    
    def train(self):
        """Main training loop."""
        from libero_rl import make_libero_env
        
        # Determine task IDs and prompts
        if self.cfg.task_ids is not None:
            # Multi-task mode
            task_ids = self.cfg.task_ids
            tasks = [get_task(self.cfg.task_suite, tid) for tid in task_ids]
            task_prompts = [task.language for task in tasks]
            task_names = [task.name for task in tasks]
        else:
            # Single task mode
            task_ids = self.cfg.task_id
            task = get_task(self.cfg.task_suite, self.cfg.task_id)
            task_prompts = task.language
            task_names = task.name
        
        print(f"\n{'='*70}")
        print(f"Starting PPO Training")
        print(f"{'='*70}")
        if self.is_vectorized:
            print(f"Mode: Multi-task vectorized ({self.cfg.num_envs} environments)")
            print(f"Tasks:")
            for i, (name, lang) in enumerate(zip(task_names, task_prompts)):
                print(f"  Env {i}: {name}")
                print(f"         {lang}")
        else:
            print(f"Mode: Single-task")
            print(f"Task: {task_names}")
            print(f"Language: {task_prompts}")
        print(f"Total timesteps: {self.cfg.total_timesteps}")
        print(f"{'='*70}\n")
        
        # Initialize wandb
        if self.cfg.use_wandb:
            # get date-time string
            from datetime import datetime
            dt_string = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{self.cfg.task_suite}_tasks{'_'.join(map(str, task_ids)) if self.is_vectorized else str(self.cfg.task_id)}_PPO_{dt_string}"
            wandb.init(
                project="OFT_RL",
                entity=self.cfg.wandb_entity,
                config={**vars(self.cfg)},
                name=run_name,
            )
        
        # Create environment
        env = make_libero_env(
            task_suite_name=self.cfg.task_suite,
            task_id=task_ids,
            num_envs=self.cfg.num_envs,
            obs_mode="raw",  # We'll process manually
            action_normalization="none",  # VLA handles normalization
            auto_reset=True if self.is_vectorized else False,
        )
        
        # Run initial validation to establish baseline
        print("\n" + "="*70)
        print("Running initial validation to establish baseline...")
        print("="*70)
        val_prompt = task_prompts[0] if isinstance(task_prompts, list) else task_prompts
        initial_val_stats = self.validate(env, val_prompt)
        print(f"Initial Success Rate: {initial_val_stats['val/success_rate']:.2%}")
        print(f"Initial Mean Reward: {initial_val_stats['val/mean_reward']:.3f}")
        print("="*70 + "\n")
        
        # Log initial validation to wandb
        if self.cfg.use_wandb:
            wandb.log({
                "val/success_rate": initial_val_stats['val/success_rate'],
                "val/mean_reward": initial_val_stats['val/mean_reward'],
            }, step=0)
        
        # Training loop
        num_updates = self.cfg.total_timesteps // (self.cfg.n_steps * self.cfg.num_envs) if self.is_vectorized else self.cfg.total_timesteps // self.cfg.n_steps
        
        pbar = tqdm(
            range(num_updates),
            desc="Training",
            unit="update",
            ncols=120,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        for update in pbar:
            # Collect rollouts
            rollout_stats = self.collect_rollouts(env, task_prompts)
            
            # Update policy (pass first task_prompt for value computation)
            # In multi-task, we use a generic prompt for value updates
            update_prompt = task_prompts[0] if isinstance(task_prompts, list) else task_prompts
            train_stats = self.update_policy(update_prompt)
            
            # Combine stats
            stats = {**rollout_stats, **train_stats}
            stats["global_step"] = self.global_step
            
            # Validation
            if self.global_step % self.cfg.val_interval == 0:
                # For multi-task, validate on first task (could extend to all tasks)
                val_prompt = task_prompts[0] if isinstance(task_prompts, list) else task_prompts
                val_stats = self.validate(env, val_prompt)
                stats.update(val_stats)
            
            # Update progress bar with stats
            pbar_stats = {
                "step": self.global_step,
                "succ": f"{stats['rollout/success_rate']:.1%}",
                "traj": stats.get('rollout/num_trajectories', 0),
            }
            if "train/policy_loss" in stats:
                pbar_stats["π_loss"] = f"{stats['train/policy_loss']:.4f}"
            if "train/clipfrac" in stats:
                pbar_stats["clip"] = f"{stats['train/clipfrac']:.3f}"
            if "val/success_rate" in stats:
                pbar_stats["val"] = f"{stats['val/success_rate']:.1%}"
            pbar.set_postfix(pbar_stats, refresh=True)
            
            # Logging
            if update % self.cfg.log_interval == 0:
                log_msg = f"Update {update}/{num_updates} | Step {self.global_step} | "
                log_msg += f"Success: {stats['rollout/success_rate']:.2%} | "
                log_msg += f"Trajectories: {stats.get('rollout/num_trajectories', 0)} | "
                if "train/policy_loss" in stats:
                    log_msg += f"Policy Loss: {stats['train/policy_loss']:.4f} | "
                if "train/clipfrac" in stats:
                    log_msg += f"Clip Frac: {stats['train/clipfrac']:.3f} | "
                if "train/approx_kl" in stats:
                    log_msg += f"KL: {stats['train/approx_kl']:.4f} | "
                if "val/success_rate" in stats:
                    log_msg += f"Val Success: {stats['val/success_rate']:.2%}"
                tqdm.write(log_msg)  # Use tqdm.write to avoid progress bar disruption
                
                if self.cfg.use_wandb:
                    wandb.log(stats, step=self.global_step)
            
            # Save checkpoint
            if self.global_step % self.cfg.save_interval == 0:
                self.save_checkpoint(f"checkpoint_{self.global_step}.pt")
        
        pbar.close()
        env.close()
        
        if self.cfg.use_wandb:
            wandb.finish()
        
        print("\n" + "="*70)
        print("Training completed!")
        print("="*70)
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint with proper LoRA adapter handling."""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            "global_step": self.global_step,
            "episode_count": self.episode_count,
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "vla_config": vars(self.vla_config),
            "ppo_config": vars(self.cfg),
        }
        
        # Handle VLA model saving based on whether LoRA is used
        if self.vla_config.use_lora and not self.vla_config.freeze_vla_backbone:
            # Save only LoRA adapter weights (much smaller - ~400MB vs ~15GB)
            try:
                from peft import get_peft_model_state_dict
                lora_state_dict = get_peft_model_state_dict(self.actor.vla)
                checkpoint["lora_adapters"] = lora_state_dict
                checkpoint["is_lora"] = True
                print(f"  → Saving LoRA adapters ({len(lora_state_dict)} tensors)")
            except Exception as e:
                print(f"  Warning: Failed to save LoRA adapters: {e}")
                print(f"  Falling back to full model save")
                checkpoint["vla_state_dict"] = self.actor.vla.state_dict()
                checkpoint["is_lora"] = False
        else:
            # Save full VLA state (frozen backbone case)
            checkpoint["vla_state_dict"] = self.actor.vla.state_dict()
            checkpoint["is_lora"] = False
        
        # Save L1 regression action head if loaded (optional)
        if self.actor.l1_action_head:
            checkpoint["l1_action_head_state_dict"] = self.actor.l1_action_head.state_dict()
            checkpoint["had_l1_head"] = True
        else:
            checkpoint["had_l1_head"] = False
        
        # Save proprio projector (always needed)
        if self.actor.proprio_projector:
            checkpoint["proprio_projector_state_dict"] = self.actor.proprio_projector.state_dict()
        
        torch.save(checkpoint, checkpoint_dir / filename)
        
        # Calculate and display checkpoint size
        checkpoint_size_mb = (checkpoint_dir / filename).stat().st_size / (1024 * 1024)
        print(f"Saved checkpoint: {filename} ({checkpoint_size_mb:.1f} MB)")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint with proper LoRA adapter handling."""
        checkpoint_dir = Path("checkpoints")
        checkpoint_path = checkpoint_dir / filename
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint: {filename}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Restore training state
        self.global_step = checkpoint["global_step"]
        self.episode_count = checkpoint["episode_count"]
        
        # Load VLA model based on checkpoint type
        if checkpoint.get("is_lora", False):
            # Load LoRA adapters
            try:
                from peft import set_peft_model_state_dict
                set_peft_model_state_dict(self.actor.vla, checkpoint["lora_adapters"])
                print("  Loaded LoRA adapters")
            except Exception as e:
                print(f"  Warning: Failed to load LoRA adapters: {e}")
        else:
            # Load full VLA state
            self.actor.vla.load_state_dict(checkpoint["vla_state_dict"])
            print("  Loaded full VLA model")
        
        # Load L1 regression action head if present in checkpoint and config allows
        if "l1_action_head_state_dict" in checkpoint:
            if self.actor.l1_action_head is not None:
                self.actor.l1_action_head.load_state_dict(checkpoint["l1_action_head_state_dict"])
                print("  Loaded L1 regression action head")
            else:
                print("  Warning: Checkpoint contains L1 head but config has load_l1_action_head=False")
                print("           Set load_l1_action_head=True in config to load it")
        elif checkpoint.get("had_l1_head", False):
            print("  Note: Checkpoint indicates L1 head was present during training but not saved")
        
        if self.actor.proprio_projector and "proprio_projector_state_dict" in checkpoint:
            self.actor.proprio_projector.load_state_dict(checkpoint["proprio_projector_state_dict"])
            print("  Loaded proprio projector")
        
        # Load optimizer
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        print("  Loaded optimizer")
        
        print(f"Checkpoint loaded successfully (step {self.global_step}, episode {self.episode_count})")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenVLA PPO Training")
    parser.add_argument("--task-suite", type=str, default="libero_spatial",
                       help="LIBERO task suite")
    parser.add_argument("--task-id", type=int, default=0,
                       help="Task ID within suite (used if --task-ids not specified)")
    parser.add_argument("--task-ids", type=int, nargs="+", default=None,
                       help="Multiple task IDs for multi-task training (e.g., --task-ids 0 1 2 3)")
    parser.add_argument("--num-envs", type=int, default=1,
                       help="Number of parallel environments (must match len(task-ids) if specified)")
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Total training timesteps")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable wandb logging")
    parser.add_argument("--use-multi-gpu", action="store_true",
                       help="Use multi-GPU setup")
    
    args = parser.parse_args()
    
    # VLA configuration - uses defaults from OpenVLAActorConfig
    vla_config = OpenVLAActorConfig()
    
    # Only override if explicitly provided via command line
    if args.use_multi_gpu:
        vla_config.use_multi_gpu = True
    
    # PPO configuration - device matches VLA training device from config
    ppo_config = PPOConfig(
        total_timesteps=args.timesteps,
        task_suite=args.task_suite,
        task_id=args.task_id,
        task_ids=args.task_ids,
        num_envs=args.num_envs,
        device=vla_config.training_device,  # Use same device as VLA training components
        use_wandb=not args.no_wandb,
        wandb_entity='deeprl_ais'
    )
    
    # Initialize and train
    trainer = OpenVLAPPO(vla_config, ppo_config)
    trainer.train()


if __name__ == "__main__":
    main()

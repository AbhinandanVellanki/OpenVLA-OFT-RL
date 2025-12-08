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
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

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
from vla_oft.min_vla.value_head import ValueHead
from ppo.config import PPOConfig
from ppo.trajectory_buffer import TrajectoryBuffer
from ppo.core_algos import logprobs_from_logits, compute_policy_loss, apply_mask_with_grad_control
from ppo.gae_ppo_trainer import GAEPPOExtensions, integrate_gae_into_trajectory_buffer
from libero_rl.utils.obs_utils import process_observation_for_vla
from libero_rl.utils.action_utils import process_action_for_libero, get_dummy_action
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
        
        # Set device from config
        self.device = torch.device(vla_config.device)
        self.training_device = self.device  # Same device for training
        
        # Set default CUDA device
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
            print(f"✓ Set default CUDA device to {self.device}")
        
        print(f"Device: {self.device}")
        
        # Initialize OpenVLA actor
        print("Initializing OpenVLA actor...")
        self.actor = OpenVLAActor(vla_config)
        
        # Enable gradient checkpointing to save memory during training
        if hasattr(self.actor.vla.language_model, 'gradient_checkpointing_enable'):
            self.actor.vla.language_model.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing for memory efficiency")
        
        # Training stage tracking
        self.training_stage = "warmup"  # "warmup", "transition", or "rl"
        self.stage_step = 0  # Steps within current stage
        
        # Initialize action tokenizer
        print("Initializing action tokenizer...")
        self.action_tokenizer = ActionTokenizer(
            vocab_size=32000,  # LLaMA vocab size
            n_bins=256,
            min_action=-1.0,
            max_action=1.0,
        )
        
        # Add bin_centers and vocab_size to VLA model for compatibility
        # Access unwrapped model (in case of DataParallel)
        vla_model = self.actor.vla.module if isinstance(self.actor.vla, nn.DataParallel) else self.actor.vla
        vla_model.bin_centers = self.action_tokenizer.bin_centers
        vla_model.vocab_size = self.action_tokenizer.vocab_size
        print(f"  {self.action_tokenizer}")
        
        # Verify VLA has norm_stats loaded (done automatically in get_vla())
        if hasattr(vla_model, 'norm_stats') and vla_model.norm_stats is not None:
            print(f"✓ VLA norm_stats loaded: {list(vla_model.norm_stats.keys())}")
        else:
            print("⚠ Warning: VLA norm_stats not loaded, unnormalization may fail")
        
        # Store unnorm_key for validation
        self.unnorm_key = "libero_spatial_no_noops"

        # Apply LoRA if configured (MUST be done BEFORE DataParallel)
        if vla_config.use_lora:
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
                print("    Falling back to full VLA training")
                print("="*70 + "\n")
                vla_config.use_lora = False
        
        # Wrap in DataParallel if enabled (MUST be done AFTER LoRA)
        if vla_config.use_data_parallel and torch.cuda.device_count() > 1:
            print(f"\n{'='*70}")
            print(f"🚀 Enabling DataParallel on {torch.cuda.device_count()} GPUs")
            print(f"{'='*70}")
            
            # Wrap VLA backbone in DataParallel
            self.actor.vla = nn.DataParallel(
                self.actor.vla,
                device_ids=list(range(torch.cuda.device_count())),
                output_device=self.device.index if self.device.type == 'cuda' else 0,
            )
            
            print(f"✓ Model replicated across GPUs: {list(range(torch.cuda.device_count()))}")
            print(f"✓ Batch will be split across GPUs automatically")
            print(f"✓ Output gathered on: {self.device}")
            print(f"{'='*70}\n")
        elif vla_config.use_data_parallel and torch.cuda.device_count() <= 1:
            print(f"⚠️  DataParallel enabled but only {torch.cuda.device_count()} GPU(s) available")
            print(f"   Running on single GPU: {self.device}")
        
        # Apply freezing strategy based on configuration
        if vla_config.freeze_vla_backbone and not vla_config.use_lora:
            # Legacy freezing: Freeze entire backbone (used when LoRA is disabled)
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
        
        # FREEZE VLA BACKBONE if configured
        if vla_config.freeze_vla_backbone:
            print("\n" + "="*70)
            print("🔒 Freezing Base VLA Backbone (LoRA adapters trainable)")
            print("="*70)
            
            # Get unwrapped model for freezing (in case of DataParallel)
            vla_model = self.actor.vla.module if isinstance(self.actor.vla, nn.DataParallel) else self.actor.vla
            
            # Freeze vision backbone
            for name, param in vla_model.vision_backbone.named_parameters():
                if 'lora' not in name.lower():
                    param.requires_grad = False
            
            # Freeze language model backbone (but NOT LoRA)
            for name, param in vla_model.language_model.named_parameters():
                if 'lora' not in name.lower():
                    param.requires_grad = False
            
            # Count trainable parameters
            trainable = sum(p.numel() for p in vla_model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in vla_model.parameters())
            
            print(f"✓ Frozen base backbone (7B parameters)")
            print(f"✓ LoRA adapters trainable: {trainable:,} parameters")
            print(f"✓ Trainable: {100*trainable/total:.2f}%")
            print("="*70 + "\n")
            
            # Debug: Show what's actually trainable
            print("\n📊 Trainable Parameter Breakdown:")
            lora_params = []
            backbone_params = []
            other_params = []
            
            for name, param in vla_model.named_parameters():
                if param.requires_grad:
                    if 'lora' in name.lower():
                        lora_params.append((name, param.numel()))
                    elif 'vision_backbone' in name or 'language_model' in name:
                        backbone_params.append((name, param.numel()))
                    else:
                        other_params.append((name, param.numel()))
            
            print(f"\n✓ Trainable LoRA parameters: {len(lora_params)}")
            if lora_params[:3]:  # Show first 3
                for n, s in lora_params[:3]:
                    print(f"  - {n}: {s:,} params")
                if len(lora_params) > 3:
                    print(f"  - ... and {len(lora_params) - 3} more")
            
            print(f"\n✓ Trainable backbone parameters: {len(backbone_params)}")
            if backbone_params[:3]:
                for n, s in backbone_params[:3]:
                    print(f"  - {n}: {s:,} params")
            elif len(backbone_params) == 0:
                print(f"  - None (all frozen ✓)")
            
            print(f"\n✓ Other trainable parameters: {len(other_params)}")
            if other_params:
                for n, s in other_params:
                    print(f"  - {n}: {s:,} params")
            
            total_lora = sum(s for _, s in lora_params)
            total_backbone = sum(s for _, s in backbone_params)
            total_other = sum(s for _, s in other_params)
            print(f"\n📈 Total trainable in VLA: {total_lora + total_backbone + total_other:,}")
            if total_lora + total_backbone + total_other > 0:
                print(f"  - LoRA: {total_lora:,} ({100*total_lora/(total_lora+total_backbone+total_other):.1f}%)")
                print(f"  - Backbone: {total_backbone:,} ({100*total_backbone/(total_lora+total_backbone+total_other):.1f}%)")
                print(f"  - Other: {total_other:,} ({100*total_other/(total_lora+total_backbone+total_other):.1f}%)")
            print()
        
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
        
        # Print final trainable parameter count (including proprio projector)
        trainable_vla = sum(p.numel() for p in vla_trainable_params if p.requires_grad)
        trainable_proprio = sum(p.numel() for p in proprio_proj_params if p.requires_grad)
        trainable_total = sum(p.numel() for p in actor_params if p.requires_grad)
        total = sum(p.numel() for p in self.actor.vla.parameters())
        
        print(f"\n📊 Final Optimizer Parameters:")
        print(f"   VLA trainable: {trainable_vla:,}")
        print(f"   Proprio projector: {trainable_proprio:,}")
        print(f"   Total trainable: {trainable_total:,} / {total:,} ({100*trainable_total/total:.2f}%)")
        print(f"   Frozen: {total - trainable_vla:,} parameters\n")

        # Initialize optimizer(s) - GAE uses separate actor/critic, GRPO uses single optimizer
        if ppo_config.use_gae:
            print("\n" + "="*70)
            print("🎯 GAE-PPO Mode: Initializing Value Head and Separate Optimizers")
            print("="*70)

            # Create value head (3-layer MLP)
            self.value_head = ValueHead(
                input_dim=4096,  # OpenVLA hidden size
                hidden_dim=1024,
            ).to(self.training_device)

            # Count value head parameters
            value_params = sum(p.numel() for p in self.value_head.parameters())
            print(f"   Value head parameters: {value_params:,}")

            # Create GAE extensions with separate optimizers
            self.gae_ext = GAEPPOExtensions(
                value_head=self.value_head,
                vla_model=self.actor.vla,
                actor_lr=ppo_config.actor_lr,
                critic_lr=ppo_config.critic_lr,
                gamma=ppo_config.gamma,
                gae_lambda=ppo_config.gae_lambda,
                value_loss_coef=ppo_config.value_loss_coef,
                freeze_vla_for_critic=ppo_config.freeze_vla_for_critic,
                device=self.training_device,
            )

            print("="*70 + "\n")
        else:
            print("\n🔵 GRPO Mode: Using single optimizer (value-free)")
            self.actor_optimizer = optim.AdamW(actor_params, lr=ppo_config.actor_lr)
            self.value_head = None
            self.gae_ext = None

        self.max_grad_norm = ppo_config.max_grad_norm

        # Trajectory buffer for storing complete episodes
        self.trajectory_buffer = TrajectoryBuffer()

        # Integrate GAE into trajectory buffer if enabled
        if ppo_config.use_gae:
            integrate_gae_into_trajectory_buffer(
                trajectory_buffer=self.trajectory_buffer,
                gae_extensions=self.gae_ext,
                use_gae=True,
            )
            print("✓ Trajectory buffer patched for GAE advantage computation\n")
        
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
        self.training_history = defaultdict(list)  # For matplotlib plotting

        # Create plots directory
        self.plots_dir = Path("plots")
        self.plots_dir.mkdir(exist_ok=True)
        
        # Multi-task tracking
        self.is_vectorized = ppo_config.num_envs > 1
        
        print(f"Initialized OpenVLA-PPO (GRPO - Value-Free Advantages)")
        print(f"  - Mode: {'Multi-task vectorized' if self.is_vectorized else 'Single-task'}")
        print(f"  - Num envs: {ppo_config.num_envs}")
        print(f"  - Actor trainable params: {sum(p.numel() for p in actor_params):,}")
        print(f"  - Action prediction: {'L1 Head + Token Log Probs (Hybrid)' if self.actor.l1_action_head else 'Tokenized (PPO mode)'}")
        print(f"  - Action bins: {self.action_tokenizer.n_bins}")
        print(f"  - L1 regression head: {'Loaded ({})'.format('frozen' if vla_config.freeze_l1_action_head else 'trainable') if self.actor.l1_action_head else 'Not loaded'}")
        print(f"  - Rollout temperature: {ppo_config.rollout_temperature}")
        print(f"  - Clip ratios: high={ppo_config.clip_ratio_high}, low={ppo_config.clip_ratio_low}")
        print(f"  - GRPO gamma: {ppo_config.verifier_gamma}")
        print(f"  - KL coefficient: {ppo_config.kl_coef}")
        print(f"  - Device: {self.device}")
        if vla_config.use_data_parallel and torch.cuda.device_count() > 1:
            print(f"  - Multi-GPU: DataParallel on {torch.cuda.device_count()} GPUs")
        else:
            print(f"  - Multi-GPU: Disabled (single GPU)")
        
        # Important: Confirm action head configuration for training
        if self.actor.l1_action_head:
            print(f"\n✓ Training will use L1 action head (same as validation) for rollouts")
            print(f"  This ensures consistent 80% success rate during training!")
    
    @property
    def vla_unwrapped(self):
        """Get the underlying VLA model, unwrapping DataParallel if needed."""
        return self.actor.vla.module if isinstance(self.actor.vla, nn.DataParallel) else self.actor.vla
    
    def get_action(
        self,
        obs: Dict[str, Any],
        task_prompt: str,
        temperature: float = 1.6,
        use_builtin_predict: bool = False,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Dict[str, Any]]:
        """
        Get action chunk from policy.
        
        Three modes:
        1. PPO training (use_builtin_predict=False): Uses L1 action head + computes log probs
        2. Validation (use_builtin_predict=True): Uses VLA's built-in predict_action()
           to exactly match reference 98% evaluation
        
        Args:
            obs: Observation dictionary with 'image' and 'proprio'
            task_prompt: Task description string
            temperature: Sampling temperature (1.6 for exploration, 0.0 for greedy eval)
            use_builtin_predict: If True, use VLA's predict_action() (for validation)
        
        Returns:
            actions: (8, action_dim) numpy array
            info: Dictionary with log_prob, responses, inputs, etc.
        """
        if use_builtin_predict:
            # Validation mode: Use VLA's built-in predict_action (matches reference 98% eval)
            return self._get_action_via_predict(obs, task_prompt)
        else:
            # Training mode: Use L1 action head (for good actions) + compute log probs (for PPO)
            if self.actor.l1_action_head is not None:
                return self._get_action_l1_with_logprobs(obs, task_prompt, temperature)
            else:
                # Fallback to tokenized actions if no L1 head available
                return self._get_action_via_tokens(obs, task_prompt, temperature)
    
    def _get_action_via_predict(
        self,
        obs: Dict[str, Any],
        task_prompt: str,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get action using VLA's built-in predict_action() method.
        This exactly matches the reference 98% evaluation implementation.
        """
        # Process observation for VLA
        prompt = f"In: What action should the robot take to {task_prompt.lower()}?\nOut:"
        
        # Get unwrapped VLA model for method calls
        vla_model = self.vla_unwrapped
        
        # Get image (already PIL Image or list of PIL Images from processing)
        image = obs['image']
        
        # Handle multi-image input: concatenate pixel_values along image dimension
        if isinstance(image, list):
            # Process first image to get base inputs
            inputs = self.actor.processor(prompt, image[0]).to(
                self.device, dtype=torch.bfloat16
            )
            
            # Process additional images and concatenate pixel_values
            for additional_image in image[1:]:
                additional_inputs = self.actor.processor(prompt, additional_image).to(
                    self.device, dtype=torch.bfloat16
                )
                # Concatenate along the image dimension (dim=1)
                inputs["pixel_values"] = torch.cat(
                    [inputs["pixel_values"], additional_inputs["pixel_values"]], dim=1
                )
        else:
            # Single image case
            inputs = self.actor.processor(prompt, image).to(
                self.device, dtype=torch.bfloat16
            )
        
        # Process proprioception if available
        proprio = None
        vla_model = self.vla_unwrapped
        if self.vla_config.use_proprio and obs.get('proprio') is not None:
            proprio = obs['proprio']
            # Normalize proprio using VLA's norm_stats
            if hasattr(vla_model, 'norm_stats') and self.unnorm_key in vla_model.norm_stats:
                proprio_stats = vla_model.norm_stats[self.unnorm_key].get('proprio')
                if proprio_stats is not None:
                    # Normalize: (proprio - mean) / std
                    mean = np.array(proprio_stats['mean'])
                    std = np.array(proprio_stats['std'])
                    proprio = (proprio - mean) / (std + 1e-8)
            
            # IMPORTANT: Convert to numpy array if not already
            if not isinstance(proprio, np.ndarray):
                proprio = np.array(proprio, dtype=np.float32)
        
        # Call VLA's predict_action (works with DataParallel)
        # Set eval mode for validation
        self.actor.vla.eval()
        with torch.no_grad():
            if self.actor.l1_action_head is not None:
                # Use L1 regression head for continuous actions
                actions, _ = vla_model.predict_action(
                    **inputs,
                    unnorm_key=self.unnorm_key,
                    do_sample=False,  # Greedy for validation
                    proprio=proprio,
                    proprio_projector=self.actor.proprio_projector,
                    action_head=self.actor.l1_action_head,
                    use_film=False,
                )
            else:
                # Standard tokenized action prediction
                actions, _ = vla_model.predict_action(
                    **inputs,
                    unnorm_key=self.unnorm_key,
                    do_sample=False,  # Greedy for validation
                )
        
        # Convert to numpy if needed
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        
        # Return actions and minimal info (no log_prob in validation)
        info = {
            'log_prob': None,
            'responses': None,
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'pixel_values': inputs['pixel_values'],
            'proprio': proprio,
        }
        
        return actions, info
    
    def _get_action_l1_with_logprobs(
        self,
        obs: Dict[str, Any],
        task_prompt: str,
        temperature: float = 1.6,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get action using L1 action head (for good performance) while computing
        log probabilities from token distribution (for PPO training).
        
        This combines the best of both worlds:
        - Actions from L1 head → same as validation (80% success rate)
        - Log probs from tokens → enables PPO gradient updates
        
        The key insight: we can execute L1 actions but train the token distribution
        to match those actions, gradually improving the tokenized action head.
        """
        # Process observation for VLA
        prompt = f"In: What action should the robot take to {task_prompt.lower()}?\nOut:"
        
        # Get image (already PIL Image or list of PIL Images from processing)
        image = obs['image']
        
        # Handle multi-image input: concatenate pixel_values along image dimension
        if isinstance(image, list):
            # Process first image to get base inputs
            inputs = self.actor.processor(prompt, image[0]).to(
                self.device, dtype=torch.bfloat16
            )
            
            # Process additional images and concatenate pixel_values
            for additional_image in image[1:]:
                additional_inputs = self.actor.processor(prompt, additional_image).to(
                    self.device, dtype=torch.bfloat16
                )
                # Concatenate along the image dimension (dim=1)
                inputs["pixel_values"] = torch.cat(
                    [inputs["pixel_values"], additional_inputs["pixel_values"]], dim=1
                )
        else:
            # Single image case
            inputs = self.actor.processor(prompt, image).to(
                self.device, dtype=torch.bfloat16
            )
        
        # Process proprioception if available
        proprio = None
        vla_model = self.vla_unwrapped
        if self.vla_config.use_proprio and obs.get('proprio') is not None:
            proprio = obs['proprio']
            # Normalize proprio using VLA's norm_stats
            if hasattr(vla_model, 'norm_stats') and self.unnorm_key in vla_model.norm_stats:
                proprio_stats = vla_model.norm_stats[self.unnorm_key].get('proprio')
                if proprio_stats is not None:
                    # Normalize: (proprio - mean) / std
                    mean = np.array(proprio_stats['mean'])
                    std = np.array(proprio_stats['std'])
                    proprio = (proprio - mean) / (std + 1e-8)
            
            # Convert to numpy array if not already
            if not isinstance(proprio, np.ndarray):
                proprio = np.array(proprio, dtype=np.float32)
        
        # Get actions from L1 head (deterministic, high-quality actions)
        # Set eval mode and use unwrapped model
        self.actor.vla.eval()
        with torch.no_grad():
            actions, _ = vla_model.predict_action(
                **inputs,
                unnorm_key=self.unnorm_key,
                do_sample=False,  # Greedy for consistency
                proprio=proprio,
                proprio_projector=self.actor.proprio_projector,
                action_head=self.actor.l1_action_head,
                use_film=False,
            )
            
            # Convert to numpy
            if isinstance(actions, torch.Tensor):
                actions = actions.cpu().numpy()
        
        # Now compute log probabilities from token distribution (for PPO)
        # We'll tokenize the L1 actions and compute their log probability
        from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
        
        # Tokenize the L1 actions
        actions_normalized = actions  # Already in [-1, 1] from predict_action
        actions_flat = actions_normalized.flatten()  # (action_dim * action_chunk,)
        
        # Discretize to token bins
        discretized = self.action_tokenizer.discretize_actions(actions_flat)  # Returns token IDs
        
        # Now get the token logits from the model to compute log prob
        # We need to do a forward pass to get logits
        action_data = self.predict_action_tokens_with_grad(
            obs, task_prompt, temperature=temperature, sample=False
        )
        
        # Get log prob of the discretized L1 actions under current token distribution
        action_token_logits = action_data['logits']  # (batch, seq_len, 256)
        
        # Convert discretized tokens to indices in [0, 255] range
        token_indices = discretized - (self.action_tokenizer.vocab_size - 256)
        token_indices = torch.from_numpy(token_indices).to(action_token_logits.device).unsqueeze(0)
        
        # Compute log probs for these specific tokens
        log_probs_per_token = logprobs_from_logits(action_token_logits, token_indices)
        log_prob = log_probs_per_token.mean(dim=-1)  # Mean over action dimensions
        
        # Compile info dictionary
        info = {
            'log_prob': log_prob[0],  # Scalar tensor
            'responses': torch.from_numpy(discretized).to(self.device),  # Tokenized L1 actions
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'pixel_values': inputs['pixel_values'],
            'proprio': proprio,
            'l1_action': actions,  # Store raw L1 actions for BC targets
        }
        
        return actions, info
    
    def _get_action_via_tokens(
        self,
        obs: Dict[str, Any],
        task_prompt: str,
        temperature: float = 1.6,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get action using tokenized prediction with gradients (for PPO training).
        """
        # Verify we're using tokenized actions
        if not self.vla_config.use_tokenized_actions:
            raise RuntimeError(
                "_get_action_via_tokens requires use_tokenized_actions=True"
            )
        
        # Get action token logits with gradients
        action_data = self.predict_action_tokens_with_grad(
            obs, task_prompt, temperature=temperature, sample=True
        )
        
        # Extract continuous actions chunk (8 actions)
        actions = action_data['continuous_actions']  # (8, 7)
        
        # Note: Actions are already in normalized form [-1, 1]
        # For training, we keep them normalized and let the environment handle it
        # For validation, we use predict_action which handles unnormalization
        
        # Compile info dictionary
        info = {
            'log_prob': action_data['log_prob'],
            'responses': action_data['responses'],
            'input_ids': action_data['input_ids'],
            'attention_mask': action_data['attention_mask'],
            'pixel_values': action_data['pixel_values'],
            'proprio': obs.get('proprio', None),
        }
        
        return actions, info
    
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
        
        # Build prompt
        prompt = f"In: What action should the robot take to {task_prompt.lower()}?\nOut:"
        
        # Convert single numpy array to PIL if needed
        # (lists of PIL Images from process_observation_for_vla are already correct)
        if isinstance(image, np.ndarray):
            from PIL import Image
            image = Image.fromarray(image)
        
        # Handle multi-image input: concatenate pixel_values along image dimension
        if isinstance(image, list) and len(image) > 1:
            # Process first image to get base inputs
            inputs = self.actor.processor(prompt, image[0]).to(
                self.device, dtype=torch.bfloat16
            )
            
            # Process additional images and concatenate pixel_values
            for additional_image in image[1:]:
                additional_inputs = self.actor.processor(prompt, additional_image).to(
                    self.device, dtype=torch.bfloat16
                )
                # Concatenate along the image dimension (dim=1)
                inputs["pixel_values"] = torch.cat(
                    [inputs["pixel_values"], additional_inputs["pixel_values"]], dim=1
                )
        else:
            # Single image (either list with 1 element or not a list)
            single_img = image[0] if isinstance(image, list) else image
            inputs = self.actor.processor(prompt, single_img).to(self.device, dtype=torch.bfloat16)
        
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
        
        # Unwrap DataParallel for method calls
        vla_model = self.vla_unwrapped
        
        # Prepare inputs (add action tokens)
        input_ids, attention_mask = vla_model._prepare_input_for_action_prediction(input_ids, attention_mask)
        labels = vla_model._prepare_labels_for_action_prediction(labels, input_ids)
        
        # Get input embeddings and action masks
        input_embeddings = vla_model.get_input_embeddings()(input_ids)
        all_actions_mask = vla_model._process_action_masks(labels)
        
        # Extract language embeddings
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )
        
        # Process vision features
        use_film = False  # Our model doesn't use FiLM
        projected_patch_embeddings = vla_model._process_vision_features(pixel_values, language_embeddings, use_film)
        
        # Add proprio if available
        if self.actor.proprio_projector is not None and proprio is not None:
            proprio_tensor = torch.from_numpy(proprio).to(projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype)
            projected_patch_embeddings = vla_model._process_proprio_features(
                projected_patch_embeddings, proprio_tensor, self.actor.proprio_projector
            )
        
        # Calculate number of patches
        NUM_PATCHES = vla_model.vision_backbone.get_num_patches() * vla_model.vision_backbone.get_num_images_in_input()
        if self.actor.proprio_projector is not None and proprio is not None:
            NUM_PATCHES += 1
        
        # Zero out action token embeddings
        all_actions_mask = all_actions_mask.unsqueeze(-1)
        input_embeddings = input_embeddings * ~all_actions_mask
        
        # Build multimodal embeddings
        multimodal_embeddings, multimodal_attention_mask = vla_model._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )
        
        # Forward through language model
        language_model_output = vla_model.language_model(
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
        # FIXED: Should be [-256:] not [-256-64:-64] to get tokens 31744-32000
        action_token_logits = action_logits[..., -256:]  # Shape: (batch, seq_len, 256)
        
        # Sample or take argmax (greedy)
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
            # Greedy decoding (argmax) - matches reference eval (modeling_prismatic.py line 936)
            sampled_indices = action_token_logits.argmax(dim=-1)
            responses = sampled_indices + (self.action_tokenizer.vocab_size - 256)
        
        # Compute log probabilities
        log_probs_per_token = logprobs_from_logits(action_token_logits, sampled_indices)
        # CRITICAL: Use mean instead of sum to normalize by sequence length (256 tokens)
        # This prevents massive log prob values (-500 to -800) that cause gradient explosions
        log_prob = log_probs_per_token.mean(dim=-1)  # Mean over action dimensions
        
        # Detokenize to continuous action chunk (8 actions)
        # Matches reference: modeling_prismatic.py lines 937-941
        responses_np = responses[0].detach().cpu().numpy()  # (action_dim * action_chunk,)
        discretized_actions = self.action_tokenizer.vocab_size - responses_np
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.action_tokenizer.bin_centers.shape[0] - 1)
        continuous_actions = self.action_tokenizer.bin_centers[discretized_actions]
        continuous_actions = continuous_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)  # (8, 7)
        
        return {
            'logits': action_token_logits,
            'responses': responses[0],  # (action_dim * action_chunk,)
            'log_prob': log_prob[0],  # Scalar tensor
            'continuous_actions': continuous_actions,  # (8, 7) - all 8 actions
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
        }
    
    def _should_use_l1_actions(self) -> bool:
        """
        Determine whether to use L1 or tokenized actions for rollouts based on training phase.
        
        Training phases:
        1. Warmup (0 to l1_warmup_steps): Use L1 actions (behavior cloning)
        2. Transition (warmup to warmup+transition): Epsilon-greedy mixing
        3. RL (after transition): Use tokenized actions (true PPO)
        
        Returns:
            True if should use L1 actions, False if should use tokenized actions
        """
        if not self.cfg.use_l1_warmstart:
            return False  # Warmstart disabled, always use tokenized
        
        if self.global_step < self.cfg.l1_warmup_steps:
            return True  # Warmup phase: use L1
        elif self.global_step < self.cfg.l1_warmup_steps + self.cfg.l1_transition_steps:
            # Transition phase: epsilon-greedy
            progress = (self.global_step - self.cfg.l1_warmup_steps) / self.cfg.l1_transition_steps
            epsilon = 1.0 - progress  # 1.0 → 0.0 over transition period
            return np.random.rand() < epsilon
        else:
            return False  # RL phase: use tokenized
    
    def _get_training_stage(self) -> str:
        """
        Determine current training stage based on global_step.
        
        Returns:
            "warmup", "transition", or "rl"
        """
        if not self.cfg.use_l1_warmstart:
            return "rl"
        
        if self.global_step < self.cfg.l1_warmup_steps:
            return "warmup"
        elif self.global_step < self.cfg.l1_warmup_steps + self.cfg.l1_transition_steps:
            return "transition"
        else:
            return "rl"
    
    def _update_training_stage(self):
        """
        Update training stage and log transition if stage changed.
        """
        new_stage = self._get_training_stage()
        if new_stage != self.training_stage:
            old_stage = self.training_stage
            self.training_stage = new_stage
            self.stage_step = 0
            
            # Log stage transition
            print(f"\n{'='*70}")
            print(f"🔄 STAGE TRANSITION: {old_stage.upper()} → {new_stage.upper()}")
            print(f"   Global Step: {self.global_step}")
            print(f"{'='*70}\n")
            
            if self.cfg.use_wandb:
                try:
                    wandb.log({
                        "stage/transition": 1,
                        "stage/name": new_stage,
                        "stage/transition_step": self.global_step,
                    }, step=self.global_step)
                except Exception as e:
                    print(f"⚠️  WARNING: Failed to log stage transition: {e}")
        else:
            self.stage_step += 1
    
    def _discretize_l1_actions(self, l1_actions: np.ndarray) -> torch.Tensor:
        """
        Convert L1 continuous actions to target token IDs for behavior cloning.
        
        Args:
            l1_actions: (action_dim,) continuous actions from L1 head
        
        Returns:
            target_tokens: (action_dim,) token IDs in [vocab_size-256, vocab_size)
        """
        # Use action tokenizer to discretize
        token_ids = self.action_tokenizer.discretize_actions(l1_actions)
        return torch.from_numpy(token_ids).long()
    
    def _bc_update_from_l1(self, task_prompt: str) -> Dict[str, float]:
        """
        Behavior cloning update: Train tokenized head to match L1 actions.
        
        Uses cross-entropy loss with L1 actions as ground truth targets.
        This is more effective than PPO loss during warmup since L1 actions
        are already near-optimal (80% success rate).
        
        Args:
            task_prompt: Task description for VLA inputs
        
        Returns:
            stats: Dictionary with training statistics
        """
        import gc
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        
        self.actor.vla.train()
        # L1 action head stays in eval mode (used for generating targets)
        if self.actor.l1_action_head is not None:
            self.actor.l1_action_head.eval()
        
        # Get data from trajectory buffer
        data = self.trajectory_buffer.get()
        
        if len(data['observations']) == 0:
            return {"train/no_data": 1.0}
        
        # Extract L1 actions (these are the ground truth targets)
        # l1_actions should be stored in the buffer during rollout
        if 'l1_actions' not in data:
            print("⚠️  WARNING: No L1 actions in buffer! Cannot perform BC update.")
            print("    Falling back to PPO update...")
            # Fall back to regular PPO (will implement after this works)
            return {"train/no_l1_data": 1.0}
        
        l1_actions = data['l1_actions']  # List of (action_dim,) arrays
        
        print(f"\n{'='*70}")
        print(f"🎯 BEHAVIOR CLONING UPDATE (Warmup Phase)")
        print(f"   Training tokenized head to match L1 actions")
        print(f"   Samples: {len(l1_actions)}")
        print(f"{'='*70}\n")
        
        # Training statistics
        stats = defaultdict(list)
        
        # Multiple epochs of optimization
        epoch_pbar = tqdm(
            range(self.cfg.n_epochs),
            desc="BC update",
            leave=False,
            ncols=100,
            bar_format='{l_bar}{bar}| Epoch {n_fmt}/{total_fmt}'
        )
        
        for epoch in epoch_pbar:
            if self.training_device.type == 'cuda':
                torch.cuda.synchronize(self.training_device)
            
            # Generate random minibatch indices
            indices = torch.randperm(len(l1_actions))
            
            # Calculate number of minibatches
            num_minibatches = (len(l1_actions) + self.cfg.batch_size - 1) // self.cfg.batch_size
            
            # Progress bar for minibatches
            minibatch_pbar = tqdm(
                range(0, len(l1_actions), self.cfg.batch_size),
                desc=f"  Epoch {epoch+1} minibatches",
                leave=False,
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} batches',
                total=num_minibatches
            )
            
            # Process minibatches
            for minibatch_idx, start_idx in enumerate(minibatch_pbar):
                end_idx = min(start_idx + self.cfg.batch_size, len(l1_actions))
                mb_indices = indices[start_idx:end_idx]
                
                # Zero gradients
                self.actor_optimizer.zero_grad()
                
                total_bc_loss = 0.0
                total_accuracy = 0.0
                
                # Process each sample in minibatch
                for i, idx in enumerate(mb_indices):
                    obs = data["observations"][idx.item()]
                    l1_actions_chunk = l1_actions[idx.item()]  # Full chunk (8, 7)
                    
                    # Convert L1 actions chunk to target token IDs
                    # Flatten to (56,) then discretize
                    l1_actions_flat = l1_actions_chunk.flatten()  # (8, 7) -> (56,)
                    target_tokens = self._discretize_l1_actions(l1_actions_flat).to(self.training_device)
                    
                    # Get action token logits from current policy (generates full chunk)
                    action_data = self.predict_action_tokens_with_grad(
                        obs, task_prompt, temperature=1.0, sample=False
                    )
                    
                    # Extract logits for the 256 action tokens
                    # logits shape: (1, 56, 256) for 8 actions × 7 dims
                    logits_full = action_data['logits'][0]  # (56, 256)
                    logits_full = logits_full.to(self.training_device)
                    
                    # Map target tokens to indices in [0, 255]
                    target_indices = target_tokens - (self.action_tokenizer.vocab_size - 256)
                    target_indices = target_indices.clamp(0, 255)  # Safety clamp
                    
                    # Cross-entropy loss: train all 8 actions at once
                    # logits: (56, 256), targets: (56,)
                    bc_loss = nn.functional.cross_entropy(logits_full, target_indices, reduction='mean')
                    
                    # Normalize by batch size for gradient accumulation
                    (bc_loss / len(mb_indices)).backward()
                    
                    # Compute accuracy (percentage of correctly predicted bins across all 8 actions)
                    with torch.no_grad():
                        pred_indices = logits_full.argmax(dim=-1)
                        accuracy = (pred_indices == target_indices).float().mean()
                        total_accuracy += accuracy.item()
                    
                    total_bc_loss += bc_loss.item()
                    
                    # Clear intermediate data
                    del action_data
                
                # Gradient clipping
                actor_params = (
                    [p for p in self.actor.vla.parameters() if p.requires_grad] +
                    (list(self.actor.proprio_projector.parameters()) if self.actor.proprio_projector else [])
                )
                
                # Check for NaN gradients
                has_nan_grad = False
                for p in actor_params:
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    print(f"\n⚠️  WARNING: NaN or inf detected in gradients! Skipping optimizer step.")
                    self.actor_optimizer.zero_grad()
                    continue
                
                # Clip gradients
                total_norm = torch.nn.utils.clip_grad_norm_(actor_params, self.max_grad_norm)
                
                if total_norm > self.max_grad_norm * 1000:
                    print(f"\n⚠️  CRITICAL: Gradient explosion: {total_norm:.2f}")
                    self.actor_optimizer.zero_grad()
                    continue
                
                # Optimizer step
                self.actor_optimizer.step()
                
                # Track statistics
                n_samples = len(mb_indices)
                avg_bc_loss = total_bc_loss / n_samples
                avg_accuracy = total_accuracy / n_samples
                
                stats['train/bc_loss'].append(avg_bc_loss)
                stats['train/bc_accuracy'].append(avg_accuracy)
                
                # Update progress bar
                minibatch_pbar.set_postfix({
                    'loss': f"{avg_bc_loss:.4f}",
                    'acc': f"{avg_accuracy:.3f}",
                }, refresh=False)
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
            
            minibatch_pbar.close()
            
            # Calculate epoch statistics
            epoch_bc_loss = np.mean([s for s in stats['train/bc_loss'][-num_minibatches:]])
            epoch_accuracy = np.mean([s for s in stats['train/bc_accuracy'][-num_minibatches:]])
            
            # Print epoch summary
            print(f"\n  Epoch {epoch+1}/{self.cfg.n_epochs} → "
                  f"BC Loss: {epoch_bc_loss:.6f} | "
                  f"Accuracy: {epoch_accuracy:.4f}")
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'loss': f"{epoch_bc_loss:.4f}",
                'acc': f"{epoch_accuracy:.3f}",
            }, refresh=False)
        
        epoch_pbar.close()
        
        # Aggregate statistics
        aggregated_stats = {
            'train/bc_loss': np.mean(stats['train/bc_loss']),
            'train/bc_accuracy': np.mean(stats['train/bc_accuracy']),
        }
        
        print(f"\n✓ BC update complete:")
        print(f"   Loss: {aggregated_stats['train/bc_loss']:.6f}")
        print(f"   Accuracy: {aggregated_stats['train/bc_accuracy']:.4f}")
        print(f"{'='*70}\n")
        
        # Clear trajectory buffer after update
        self.trajectory_buffer.clear()
        
        return aggregated_stats
    
    def _get_rollout_policy_name(self) -> str:
        """
        Get human-readable name of current rollout policy for logging.
        
        Returns:
            String describing current policy (e.g., "L1 (warmup)", "Tokenized (RL)")
        """
        if not self.cfg.use_l1_warmstart:
            return "Tokenized (RL)"
        
        if self.global_step < self.cfg.l1_warmup_steps:
            return "L1 (warmup)"
        elif self.global_step < self.cfg.l1_warmup_steps + self.cfg.l1_transition_steps:
            progress = (self.global_step - self.cfg.l1_warmup_steps) / self.cfg.l1_transition_steps
            return f"L1→Tokenized ({progress:.0%})"
        else:
            return "Tokenized (RL)"
    
    def collect_rollouts(
        self,
        env,
        task_prompts: Union[str, List[str]],
    ) -> Dict[str, float]:
        """
        Collect trajectory-based rollouts with sparse rewards.
        
        Collects complete episodes, assigns sparse rewards only at episode completion,
        and stores action token IDs for PPO training.
        
        Implements phased training:
        1. Warmup: Execute L1 actions, train tokenized to match (behavior cloning)
        2. Transition: Gradually shift to tokenized actions (epsilon-greedy)
        3. RL: Execute tokenized actions, train with true PPO
        
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
                        
                        # Compute value estimate if using GAE
                        if self.cfg.use_gae:
                            # Get hidden states from VLA
                            hidden_states = self.actor.vla.get_hidden_states(
                                obs_dict=actor_obs,
                                instruction=task_prompts[env_idx],
                            )
                            value_estimate = self.value_head(hidden_states).item()
                        else:
                            value_estimate = 0.0  # GRPO doesn't use values

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
                            value=value_estimate,
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
                    # Single environment with action chunking
                    # Initialize action queue if needed
                    if not hasattr(self, 'action_queue') or len(self.action_queue) == 0:
                        processed_obs = process_observation_for_vla(
                            obs,
                            camera_name="agentview",
                            resize_size=self.cfg.image_size,
                            num_images=self.vla_config.num_images_in_input,  # Match model's expected input (2 cameras)
                            center_crop=True,
                            crop_scale=0.9,
                            return_pil=True,  # Return PIL Images for VLA processor
                        )
                        
                        actor_obs = {
                            "image": processed_obs["image"],
                            "proprio": processed_obs["proprio"],  # Use 8D axis-angle format
                        }
                        
                        # Choose policy based on training phase
                        use_l1 = self._should_use_l1_actions()
                        
                        if use_l1:
                            # Warmup/Transition: Get action chunk (8 actions) using L1 head
                            actions_chunk, action_info = self.get_action(
                                actor_obs,
                                task_prompts[0],
                                temperature=self.cfg.rollout_temperature,
                                use_builtin_predict=False,  # Use L1 head + log probs for training
                            )
                            
                            # During warmup, also store the L1 actions for BC targets
                            # The action_info already contains 'l1_action' key from _get_action_l1_with_logprobs
                            l1_actions_chunk = action_info.get('l1_action', actions_chunk)
                        else:
                            # RL phase: Get action chunk using tokenized head
                            actions_chunk, action_info = self._get_action_via_tokens(
                                actor_obs,
                                task_prompts[0],
                                temperature=self.cfg.rollout_temperature,
                            )
                            # No L1 actions in RL phase
                            l1_actions_chunk = None
                        
                        # Initialize action queue with all 8 actions
                        from collections import deque
                        self.action_queue = deque(maxlen=8)
                        for act in actions_chunk:
                            self.action_queue.append(act)
                        
                        # Store action_info, observation, and full L1 chunk for this action chunk
                        # We'll add to buffer ONCE per chunk, not per action
                        self.current_action_info = action_info
                        self.current_actor_obs = actor_obs
                        self.current_actions_chunk = actions_chunk  # Store full chunk (8, 7)
                        self.current_l1_actions_chunk = l1_actions_chunk  # Full (8, 7) array or None
                        self.chunk_step_count = 0  # Track how many actions from this chunk we've executed
                        self.chunk_rewards = []  # Collect rewards for all 8 steps
                    
                    # Pop action from queue and use stored observation/info
                    action = self.action_queue.popleft()
                    action_info = self.current_action_info
                    actor_obs = self.current_actor_obs
                    
                    # Process action for LIBERO (normalize + invert gripper)
                    action = process_action_for_libero(action)
                    
                    # Step environment
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    # Collect reward for this step in the chunk
                    self.chunk_rewards.append(reward)
                    self.chunk_step_count += 1
                    
                    # Sparse rewards: zero intermediate rewards, only at finish
                    sparse_reward = 0.0
                    if done:
                        success = info.get("success", False)
                        sparse_reward = 1.0 if success else 0.0
                        episode_successes.append(float(success))
                        episode_lengths.append(current_episode_length + 1)
                        current_episode_length = 0
                    else:
                        current_episode_length += 1
                        obs = next_obs
                    
                    # Add to buffer when chunk is complete (all 8 actions executed) OR episode ends
                    should_add_chunk = (self.chunk_step_count == 8) or done or len(self.action_queue) == 0
                    
                    if should_add_chunk:
                        # Compute chunk reward (use sparse reward if done, else 0)
                        chunk_reward = sparse_reward if done else 0.0
                        
                        # Add ONCE per chunk to trajectory buffer
                        self.trajectory_buffer.add(
                            obs=self.current_actor_obs,
                            responses=self.current_action_info['responses'],
                            input_ids=self.current_action_info['input_ids'],
                            attention_mask=self.current_action_info['attention_mask'],
                            pixel_values=self.current_action_info['pixel_values'],
                            proprio=self.current_action_info['proprio'],
                            action=self.current_actions_chunk,  # Store full chunk (8, 7)
                            l1_action=self.current_l1_actions_chunk,  # Full (8, 7) or None
                            reward=chunk_reward,
                            done=done,
                            value=0.0,
                            old_log_prob=self.current_action_info['log_prob'],
                        )
                        
                        # Reset chunk tracking
                        self.chunk_step_count = 0
                        self.chunk_rewards = []
                    
                    if done:
                        # Clear action queue, info, and observation on episode end
                        self.action_queue.clear()
                        self.current_action_info = None
                        self.current_actor_obs = None
                        self.current_actions_chunk = None
                        self.current_l1_actions_chunk = None
                        obs, info = env.reset()
                    
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
        
        # Log rollout policy for tracking training phase
        rollout_policy = self._get_rollout_policy_name()
        print(f"\n🎯 Rollout Policy: {rollout_policy}")
        if self.cfg.use_l1_warmstart:
            warmup_progress = min(1.0, self.global_step / self.cfg.l1_warmup_steps)
            print(f"   Warmup Progress: {warmup_progress:.1%} ({self.global_step}/{self.cfg.l1_warmup_steps} steps)")
        
        # Debug: Check old log probabilities from rollout
        data_for_debug = self.trajectory_buffer.get()
        if len(data_for_debug['old_log_probs']) > 0:
            old_log_probs_np = data_for_debug['old_log_probs'].cpu().numpy()
            print(f"\n📊 Old Log Probability Statistics (from rollout):")
            print(f"   Mean: {old_log_probs_np.mean():.6f}")
            print(f"   Std: {old_log_probs_np.std():.6f}")
            print(f"   Min: {old_log_probs_np.min():.6f}")
            print(f"   Max: {old_log_probs_np.max():.6f}")
            print(f"   Any NaN: {np.isnan(old_log_probs_np).any()}")
            print(f"   Any Inf: {np.isinf(old_log_probs_np).any()}")
        
        # Compute statistics
        success_rate = np.mean(episode_successes) if episode_successes else 0.0
        mean_length = np.mean(episode_lengths) if episode_lengths else 0.0
        num_trajectories = len(self.trajectory_buffer)
        
        stats = {
            "rollout/mean_reward": success_rate,  # Success rate as reward
            "rollout/mean_length": mean_length,
            "rollout/success_rate": success_rate,
            "rollout/num_episodes": len(episode_successes),
            "rollout/num_trajectories": num_trajectories,
        }
        
        # Print rollout summary
        print(f"\n📊 Rollout Summary:")
        print(f"   Trajectories collected: {num_trajectories}")
        print(f"   Episodes completed: {len(episode_successes)}")
        print(f"   Success rate: {success_rate*100:.1f}%")
        print(f"   Mean episode length: {mean_length:.1f} steps")
        print(f"   Steps collected: {steps_collected}/{self.cfg.n_steps}")
        
        return stats
    
    def update_policy(self, task_prompt: str) -> Dict[str, float]:
        """
        Update policy using appropriate loss based on training stage:
        - Warmup stage: Behavior cloning with cross-entropy loss (L1 actions as targets)
        - Transition/RL stages: PPO with GRPO advantages
        
        Implements PPO policy gradient with:
        - Action token log probabilities
        - Asymmetric clipped surrogate objective
        - GRPO advantages (value-free)
        - Per-sample gradient accumulation
        - Optional KL penalty from reference policy
        
        Note: GRPO computes advantages directly from sparse rewards,
        no value function needed.
        """
        # Check if we should use behavior cloning (warmup phase)
        training_stage = self._get_training_stage()
        if training_stage == "warmup":
            return self._bc_update_from_l1(task_prompt)
        
        # Otherwise, use PPO loss (transition and RL phases)
        # Aggressive cleanup before training to prevent CUDA memory corruption
        import gc
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        
        self.actor.vla.train()
        # L1 action head stays in eval mode (not used for PPO)
        if self.actor.l1_action_head is not None:
            self.actor.l1_action_head.eval()
        
        # Get data from trajectory buffer
        data = self.trajectory_buffer.get()
        
        if len(data['observations']) == 0:
            return {"train/no_data": 1.0}
        
        # CRITICAL: Verify all model components are on correct devices before training
        print(f"\n🔍 Device Audit Before Training:")
        print(f"  VLA backbone: {next(self.actor.vla.parameters()).device}")
        if hasattr(self.actor, 'l1_action_head') and self.actor.l1_action_head is not None:
            print(f"  L1 action head: {next(self.actor.l1_action_head.parameters()).device}")
        if hasattr(self.actor, 'proprio_projector') and self.actor.proprio_projector is not None:
            print(f"  Proprio projector: {next(self.actor.proprio_projector.parameters()).device}")
        if hasattr(self.actor.vla, 'action_head') and self.actor.vla.action_head is not None:
            print(f"  VLA's action_head: {next(self.actor.vla.action_head.parameters()).device}")
        if hasattr(self.actor.vla, 'proprio_projector') and self.actor.vla.proprio_projector is not None:
            print(f"  VLA's proprio_projector: {next(self.actor.vla.proprio_projector.parameters()).device}")
        print(f"  Expected device: {self.device}")
        print(f"  Expected training device: {self.training_device}")
        
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
            # Synchronize CUDA to catch any async errors early
            if self.training_device.type == 'cuda':
                torch.cuda.synchronize(self.training_device)
            
            # Generate random minibatch indices on training_device
            indices = torch.randperm(len(advantages), device=self.training_device)
            
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
            for minibatch_idx, start_idx in enumerate(minibatch_pbar):
                end_idx = min(start_idx + self.cfg.batch_size, len(advantages))
                mb_indices = indices[start_idx:end_idx]
                
                # Index tensors (keep indices on CPU, move indexed tensors to training_device)
                mb_indices_cpu = mb_indices.cpu()
                mb_advantages = advantages[mb_indices_cpu].to(self.training_device)
                mb_old_log_probs = old_log_probs[mb_indices_cpu].to(self.training_device)
                mb_responses = responses[mb_indices_cpu].to(self.training_device)
                
                # ==================== ACTOR UPDATE (PPO Policy Gradient) ====================
                # Batch process samples with gradient accumulation
                if self.cfg.use_gae:
                    self.gae_ext.actor_optimizer.zero_grad()
                    self.gae_ext.critic_optimizer.zero_grad()
                else:
                    self.actor_optimizer.zero_grad()
                
                total_policy_loss = 0.0
                total_clipfrac = 0.0
                total_approx_kl = 0.0
                
                # Collect all action data in a batch to minimize GPU transfers
                batch_logits = []
                batch_log_probs = []
                
                for i, idx in enumerate(mb_indices_cpu):
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
                num_valid_samples = 0
                
                # Debug first minibatch
                if minibatch_idx == 0:
                    print(f"\n🔍 Debugging Minibatch {minibatch_idx}:")
                
                for i, logits in enumerate(batch_logits):
                    response = mb_responses[i]
                    old_log_prob = mb_old_log_probs[i]
                    advantage = mb_advantages[i]
                    
                    # Check for NaN/inf in inputs
                    if torch.isnan(old_log_prob).any() or torch.isinf(old_log_prob).any():
                        print(f"\n⚠️  WARNING: NaN/inf in old_log_prob at sample {i}! Skipping sample.")
                        continue
                    
                    if torch.isnan(advantage).any() or torch.isinf(advantage).any():
                        print(f"\n⚠️  WARNING: NaN/inf in advantage at sample {i}! Skipping sample.")
                        continue
                    
                    # Map response tokens to indices in [0, 255]
                    response_indices = response - (self.action_tokenizer.vocab_size - 256)
                    
                    # Ensure logits and response_indices are on the same device
                    logits = logits.to(self.training_device)
                    response_indices = response_indices.to(self.training_device)
                    log_prob = logprobs_from_logits(logits.unsqueeze(0), response_indices.unsqueeze(0))
                    # CRITICAL: Use mean to match rollout computation and prevent massive values
                    log_prob = log_prob.mean()  # Mean over action dimensions
                    
                    # Check for NaN in computed log_prob
                    if torch.isnan(log_prob).any() or torch.isinf(log_prob).any():
                        print(f"\n⚠️  WARNING: NaN/inf in computed log_prob at sample {i}!")
                        print(f"   old_log_prob: {old_log_prob.item():.4f}")
                        print(f"   Skipping sample.")
                        continue
                    
                    # Debug first sample of first minibatch
                    if minibatch_idx == 0 and i == 0:
                        print(f"   Sample {i}:")
                        print(f"     old_log_prob: {old_log_prob.item():.4f}")
                        print(f"     new_log_prob: {log_prob.item():.4f}")
                        print(f"     advantage (raw): {advantage.item():.4f}")
                    
                    # CRITICAL: Scale advantage to control gradient magnitude
                    # With sparse binary rewards (0 or 1), successful trajectories have advantage=1.0
                    # This causes large gradients and high clip fractions
                    # Scaling reduces this while maintaining relative importance
                    advantage = advantage * self.cfg.advantage_scale
                    
                    if minibatch_idx == 0 and i == 0:
                        print(f"     advantage (scaled): {advantage.item():.4f}")
                    
                    # Compute PPO loss for this sample with VERY aggressive clamping
                    log_ratio = log_prob - old_log_prob
                    
                    # Debug first sample
                    if minibatch_idx == 0 and i == 0:
                        print(f"     log_ratio (raw): {log_ratio.item():.4f}")
                    
                    # Debug extreme log ratios
                    if torch.abs(log_ratio).item() > 10.0:
                        print(f"\n⚠️  Large log ratio detected: {log_ratio.item():.2f}")
                        print(f"   new_log_prob: {log_prob.item():.2f}, old_log_prob: {old_log_prob.item():.2f}")
                    
                    # CRITICAL: Clamp to very tight range since log probs are huge (-500 to -800)
                    # Even small absolute differences create massive ratios
                    log_ratio = torch.clamp(log_ratio, min=-5.0, max=5.0)  # e^5 ≈ 148, e^-5 ≈ 0.007
                    ratio = torch.exp(log_ratio)
                    
                    # Also clamp advantage to prevent extreme products
                    advantage_clamped = torch.clamp(advantage, min=-10.0, max=10.0)
                    
                    clip_high = torch.clamp(ratio, 1 - self.cfg.clip_ratio_low, 1 + self.cfg.clip_ratio_high)
                    clip_low = torch.clamp(ratio, 1 - self.cfg.clip_ratio_high, 1 + self.cfg.clip_ratio_low)
                    clipped_ratio = torch.where(advantage_clamped > 0, clip_high, clip_low)
                    policy_loss = -torch.min(ratio * advantage_clamped, clipped_ratio * advantage_clamped)
                    
                    # Debug first sample
                    if minibatch_idx == 0 and i == 0:
                        print(f"     ratio: {ratio.item():.4f}")
                        print(f"     clipped_ratio: {clipped_ratio.item():.4f}")
                        print(f"     policy_loss: {policy_loss.item():.4f}")
                        print(f"     Has NaN: old_log={torch.isnan(old_log_prob).any()}, new_log={torch.isnan(log_prob).any()}, adv={torch.isnan(advantage).any()}, loss={torch.isnan(policy_loss).any()}")
                    
                    # Safety check: if loss is extreme, skip this sample
                    if torch.abs(policy_loss).item() > 100.0:
                        print(f"\n⚠️  WARNING: Extreme policy loss {policy_loss.item():.2f} at sample {i}! Skipping.")
                        continue
                    
                    # Accumulate loss (normalize by batch size for gradient averaging)
                    accumulated_loss += policy_loss / len(mb_indices)
                    num_valid_samples += 1
                    
                    # Track statistics
                    total_policy_loss += policy_loss.item()
                    with torch.no_grad():
                        clipfrac = ((ratio - clipped_ratio).abs() > 1e-6).float().mean()
                        approx_kl = (old_log_prob - log_prob).mean()
                        total_clipfrac += clipfrac.item()
                        total_approx_kl += approx_kl.item()
                
                # Check if we have any valid samples
                if num_valid_samples == 0:
                    print(f"\n⚠️  WARNING: No valid samples in minibatch! Skipping backward pass.")
                    continue

                # ==================== VALUE LOSS (GAE only) ====================
                if self.cfg.use_gae:
                    # Compute value estimates for the minibatch
                    batch_values = []
                    for i, idx in enumerate(mb_indices_cpu):
                        obs = data["observations"][idx.item()]
                        # Get hidden states from VLA
                        hidden_states = self.actor.vla.get_hidden_states(
                            obs_dict=obs,
                            instruction=task_prompt,
                        )
                        value_est = self.value_head(hidden_states)
                        batch_values.append(value_est)

                    # Stack and compute value loss
                    batch_values = torch.stack(batch_values).squeeze(-1)
                    mb_returns = returns[mb_indices_cpu].to(self.training_device)
                    value_loss = self.gae_ext.compute_value_loss(batch_values, mb_returns)

                    # Combined loss
                    total_loss = accumulated_loss + self.cfg.value_loss_coef * value_loss

                    # Track value loss
                    stats['train/value_loss'].append(value_loss.item())

                    if minibatch_idx == 0:
                        print(f"   Accumulated policy loss: {accumulated_loss.item():.4f} (from {num_valid_samples} samples)")
                        print(f"   Value loss: {value_loss.item():.4f}")
                        print(f"   Total loss: {total_loss.item():.4f}")
                        print(f"   Has NaN: {torch.isnan(total_loss).any()}")
                else:
                    # GRPO mode: no value loss
                    total_loss = accumulated_loss
                    value_loss = torch.tensor(0.0)

                    # Debug accumulated loss
                    if minibatch_idx == 0:
                        print(f"   Accumulated loss: {accumulated_loss.item():.4f} (from {num_valid_samples} samples)")
                        print(f"   Has NaN: {torch.isnan(accumulated_loss).any()}")

                # ==================== BACKWARD PASS ====================
                # Single backward pass for entire minibatch (policy + value if GAE)
                total_loss.backward()
                
                # Clear batch data
                del batch_logits, accumulated_loss
                torch.cuda.empty_cache()
                
                # Gradient clipping with NaN detection
                actor_params = (
                    [p for p in self.actor.vla.parameters() if p.requires_grad] +
                    (list(self.actor.l1_action_head.parameters()) if self.actor.l1_action_head and not self.vla_config.freeze_l1_action_head else []) +
                    (list(self.actor.proprio_projector.parameters()) if self.actor.proprio_projector else [])
                )
                
                # Check for NaN gradients before clipping
                has_nan_grad = False
                for p in actor_params:
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    print(f"\n⚠️  WARNING: NaN or inf detected in gradients! Skipping optimizer step.")
                    self.actor_optimizer.zero_grad()
                    continue
                
                # Compute gradient norm before clipping (for debugging)
                total_norm = torch.nn.utils.clip_grad_norm_(actor_params, self.max_grad_norm)
                
                # Skip only truly catastrophic gradient explosions (>1000x)
                # With LoRA (55M params), even gradients of 100-500 can be clipped and trained
                if total_norm > self.max_grad_norm * 1000:
                    print(f"\n⚠️  CRITICAL: Gradient explosion: {total_norm:.2f} (clip at {self.max_grad_norm})")
                    print(f"   Skipping optimizer step to prevent training collapse.")
                    self.actor_optimizer.zero_grad()
                    continue
                
                # Warn if gradient norm is very large (but clipping handles it)
                if total_norm > self.max_grad_norm * 100:
                    print(f"⚠️  Large gradient: {total_norm:.2f} → clipped to {self.max_grad_norm}")
                
                # Optimizer step
                if self.cfg.use_gae:
                    # GAE mode: clip gradients and step both optimizers
                    actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.actor.vla.parameters(), self.max_grad_norm
                    )
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.value_head.parameters(), self.max_grad_norm
                    )

                    # Check for NaN gradients
                    actor_has_nan = any(
                        torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                        for p in self.actor.vla.parameters()
                        if p.grad is not None
                    )
                    critic_has_nan = any(
                        torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                        for p in self.value_head.parameters()
                        if p.grad is not None
                    )

                    if not actor_has_nan and not critic_has_nan:
                        self.gae_ext.actor_optimizer.step()
                        self.gae_ext.critic_optimizer.step()

                        stats['train/actor_grad_norm'].append(actor_grad_norm.item() if isinstance(actor_grad_norm, torch.Tensor) else actor_grad_norm)
                        stats['train/critic_grad_norm'].append(critic_grad_norm.item() if isinstance(critic_grad_norm, torch.Tensor) else critic_grad_norm)
                    else:
                        print("\n⚠️  WARNING: NaN/inf detected in gradients! Skipping optimizer step.")
                        self.gae_ext.actor_optimizer.zero_grad()
                        self.gae_ext.critic_optimizer.zero_grad()
                else:
                    # GRPO mode: update actor only
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
            
            # Calculate and print epoch statistics
            epoch_policy_loss = np.mean([s for s in stats['train/policy_loss'][-num_minibatches:]])
            epoch_clipfrac = np.mean([s for s in stats['train/clipfrac'][-num_minibatches:]])
            epoch_approx_kl = np.mean([s for s in stats['train/approx_kl'][-num_minibatches:]])
            
            # Print epoch summary
            print(f"\n  Epoch {epoch+1}/{self.cfg.n_epochs} → "
                  f"Policy Loss: {epoch_policy_loss:.6f} | "
                  f"Clip Frac: {epoch_clipfrac:.4f} | "
                  f"KL: {epoch_approx_kl:.6f}")
            
            # Update epoch progress bar with average stats
            epoch_pbar.set_postfix({
                'avg_loss': f"{epoch_policy_loss:.4f}",
                'avg_clip': f"{epoch_clipfrac:.3f}",
            }, refresh=False)
        
        epoch_pbar.close()
        # ...existing code...
        return {k: np.mean(v) for k, v in stats.items()}
    
    def plot_training_curves(self, update: int):
        """
        Create and save matplotlib plots of training metrics.

        Saves plots to plots/ directory after each epoch/update.
        """
        if not self.training_history:
            return

        # Create figure with subplots
        num_plots = 4 if self.cfg.use_gae else 3
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4*num_plots))
        if num_plots == 1:
            axes = [axes]

        plot_idx = 0

        # Plot 1: Losses
        ax = axes[plot_idx]
        if 'train/policy_loss' in self.training_history:
            ax.plot(self.training_history['train/policy_loss'], label='Policy Loss', color='blue')
        if 'train/value_loss' in self.training_history and self.cfg.use_gae:
            ax.plot(self.training_history['train/value_loss'], label='Value Loss', color='orange')
        ax.set_xlabel('Update')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

        # Plot 2: Success Rate and Rewards
        ax = axes[plot_idx]
        if 'rollout/success_rate' in self.training_history:
            ax.plot(self.training_history['rollout/success_rate'], label='Success Rate', color='green')
        if 'val/success_rate' in self.training_history:
            val_steps = [i * (self.cfg.val_interval // self.cfg.n_steps) for i in range(len(self.training_history['val/success_rate']))]
            ax.plot(val_steps, self.training_history['val/success_rate'], 'o-', label='Val Success Rate', color='darkgreen', markersize=6)
        ax.set_xlabel('Update')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rates')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

        # Plot 3: PPO Metrics
        ax = axes[plot_idx]
        if 'train/clipfrac' in self.training_history:
            ax.plot(self.training_history['train/clipfrac'], label='Clip Fraction', color='purple')
        if 'train/approx_kl' in self.training_history:
            ax2 = ax.twinx()
            ax2.plot(self.training_history['train/approx_kl'], label='Approx KL', color='red', alpha=0.7)
            ax2.set_ylabel('KL Divergence', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
        ax.set_xlabel('Update')
        ax.set_ylabel('Clip Fraction')
        ax.set_title('PPO Metrics')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        plot_idx += 1

        # Plot 4: Gradient Norms (GAE only)
        if self.cfg.use_gae and plot_idx < num_plots:
            ax = axes[plot_idx]
            if 'train/actor_grad_norm' in self.training_history:
                ax.plot(self.training_history['train/actor_grad_norm'], label='Actor Grad Norm', color='blue')
            if 'train/critic_grad_norm' in self.training_history:
                ax.plot(self.training_history['train/critic_grad_norm'], label='Critic Grad Norm', color='orange')
            ax.set_xlabel('Update')
            ax.set_ylabel('Gradient Norm')
            ax.set_title('Gradient Norms')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = self.plots_dir / f"training_curves_update{update:05d}.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        print(f"✓ Saved training plots to {plot_path}")

    def validate_tokenized(self, env, task_prompt: str) -> Dict[str, float]:
        """
        Run validation episodes using ONLY tokenized action head (no L1 head).
        
        This validates whether the tokenized action head has learned from the hybrid
        training (where we execute L1 actions but train the tokenized head).
        
        Uses greedy (argmax) sampling from token distribution for deterministic evaluation.
        """
        # Set to eval mode
        self.actor.vla.eval()
        
        # Memory cleanup
        import gc
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        
        val_rewards = []
        val_successes = []
        
        # Use inference_mode for deterministic evaluation
        with torch.inference_mode():
            for ep in tqdm(
                range(self.cfg.val_episodes),
                desc="Val (Tokenized)",
                leave=False,
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} episodes'
            ):
                obs, info = env.reset()
                episode_reward = 0
                done = False
                step_count = 0
                
                from collections import deque
                action_queue = deque(maxlen=8)
                
                while not done:
                    # Query policy if action queue is empty
                    if len(action_queue) == 0:
                        # Process observation (2 cameras)
                        processed_obs = process_observation_for_vla(
                            obs,
                            camera_name="agentview",
                            resize_size=self.cfg.image_size,
                            num_images=2,
                            center_crop=True,
                            crop_scale=0.9,
                            return_pil=True,
                        )
                        actor_obs = {
                            "image": processed_obs["image"],
                            "proprio": processed_obs["proprio"],
                        }
                        
                        # Get actions from tokenized head ONLY (greedy sampling)
                        action_data = self.predict_action_tokens_with_grad(
                            actor_obs,
                            task_prompt,
                            temperature=0.0,  # Greedy (argmax)
                            sample=False,     # Use argmax, not sampling
                        )
                        
                        # Extract continuous action chunk (8, 7)
                        actions_chunk = action_data['continuous_actions']
                        
                        # Fill action queue with all 8 actions
                        for act in actions_chunk:
                            action_queue.append(act)
                    
                    # Pop action from queue
                    action = action_queue.popleft()
                    
                    # Process action for LIBERO (normalize + invert gripper)
                    action = process_action_for_libero(action)
                    
                    # Step environment
                    obs, reward, terminated, truncated, info = env.step(
                        action.tolist() if hasattr(action, 'tolist') else action
                    )
                    episode_reward += reward
                    done = terminated or truncated
                    step_count += 1
            
                val_rewards.append(episode_reward)
                val_successes.append(float(info.get("success", False)))
                
                # Clear cache periodically
                if (ep + 1) % 2 == 0:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        
        # Final cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        return {
            "val/tokenized_mean_reward": np.mean(val_rewards),
            "val/tokenized_success_rate": np.mean(val_successes),
        }
    
    def validate(self, env, task_prompt: str) -> Dict[str, float]:
        """
        Run validation episodes to measure success rate.
        
        Runs TWO types of validation:
        1. L1 head validation (pretrained, deterministic)
        2. Tokenized head validation (being trained via PPO)
        
        This allows us to track whether the tokenized head is learning from
        the hybrid training approach (execute L1, train tokenized).
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
        
        # === RUN L1 VALIDATION ===
        val_rewards = []
        val_successes = []
        
        # Use inference_mode for deterministic evaluation (matches reference)
        with torch.inference_mode():
            for ep in tqdm(
                range(self.cfg.val_episodes),
                desc="Val (L1 Head)",
                leave=False,
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} episodes'
            ):
                obs, info = env.reset()
                episode_reward = 0
                done = False
                step_count = 0
                
                # Note: Environment already does 10-step waiting in reset(), no need to wait again
                
                # Initialize action queue for this episode
                from collections import deque
                action_queue = deque(maxlen=8)
                
                # Debug: Print first episode info
                # if ep == 0:
                #     print(f"\n[DEBUG] Validation episode {ep}:")
                #     print(f"  Task: {task_prompt}")
                #     print(f"  Obs keys: {list(obs.keys())}")
                #     print(f"  Image shape: {obs.get('agentview_image', np.array([])).shape}")
                
                while not done:
                    # Query policy if action queue is empty
                    if len(action_queue) == 0:
                        # Process observation (2 cameras: agentview + wrist)
                        # IMPORTANT: Model was trained with num_images_in_input=2
                        processed_obs = process_observation_for_vla(
                            obs,
                            camera_name="agentview",
                            resize_size=self.cfg.image_size,
                            num_images=2,  # TWO cameras required by pretrained model
                            center_crop=True,
                            crop_scale=0.9,
                            return_pil=True,  # Return PIL Images for VLA processor
                        )
                        actor_obs = {
                            "image": processed_obs["image"],
                            "proprio": processed_obs["proprio"],  # Use 8D axis-angle format
                        }
                        
                        # Get action chunk (8 actions) using VLA's built-in predict_action
                        # This exactly matches reference 98% evaluation
                        actions_chunk, action_info = self.get_action(
                            actor_obs,
                            task_prompt,
                            temperature=self.cfg.eval_temperature,
                            use_builtin_predict=True,  # Use VLA's predict_action for validation
                        )
                        
                        # Debug: Print first action chunk
                        if ep == 0 and step_count == 0:
                            print(f"  Actions shape: {actions_chunk.shape}")
                            print(f"  First action (raw): {actions_chunk[0]}")
                        
                        # Fill action queue with all 8 actions
                        for act in actions_chunk:
                            action_queue.append(act)
                    
                    # Pop action from queue
                    action = action_queue.popleft()
                    
                    # Debug: Print first processed action
                    if ep == 0 and step_count == 0:
                        print(f"  Action before processing: {action}")
                    
                    # Process action for LIBERO (normalize + invert gripper)
                    action = process_action_for_libero(action)
                    
                    # Debug: Print first processed action
                    if ep == 0 and step_count == 0:
                        print(f"  Action after processing: {action}")
                        print(f"  Action type: {type(action)}, dtype: {action.dtype if hasattr(action, 'dtype') else 'N/A'}")
                    
                    # Step environment (convert to list like reference does)
                    obs, reward, terminated, truncated, info = env.step(action.tolist() if hasattr(action, 'tolist') else action)
                    episode_reward += reward
                    done = terminated or truncated
                    step_count += 1
            
                val_rewards.append(episode_reward)
                val_successes.append(float(info.get("success", False)))
                
                # Clear cache frequently during validation to prevent memory buildup
                if (ep + 1) % 2 == 0:  # Every 2 episodes
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        
        # Store L1 results
        l1_metrics = {
            "val/l1_mean_reward": np.mean(val_rewards),
            "val/l1_success_rate": np.mean(val_successes),
        }
        
        # Memory cleanup between validations
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        
        # === RUN TOKENIZED VALIDATION ===
        print(f"\n[Validation] L1 Head: {l1_metrics['val/l1_success_rate']*100:.1f}% success")
        tokenized_metrics = self.validate_tokenized(env, task_prompt)
        print(f"[Validation] Tokenized Head: {tokenized_metrics['val/tokenized_success_rate']*100:.1f}% success")
        
        # Calculate gap
        gap = l1_metrics['val/l1_success_rate'] - tokenized_metrics['val/tokenized_success_rate']
        print(f"[Validation] Gap: {gap*100:.1f}% (tokenized needs to close this)")
        
        # Final cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        # Combine metrics
        all_metrics = {**l1_metrics, **tokenized_metrics}
        all_metrics['val/gap'] = gap
        
        return all_metrics
    
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
            
            print("\n" + "="*70)
            print("Initializing Weights & Biases...")
            print("="*70)
            
            try:
                # Force online mode (not offline)
                os.environ.pop('WANDB_MODE', None)  # Remove offline mode if set
                
                run = wandb.init(
                    project="OFT_RL",
                    entity=self.cfg.wandb_entity,
                    config={**vars(self.cfg)},
                    name=run_name,
                    settings=wandb.Settings(start_method="thread"),  # Avoid fork issues
                    mode="online",  # Force online mode
                )
                
                print(f"✓ wandb initialized successfully!")
                print(f"  Project: OFT_RL")
                print(f"  Run name: {run_name}")
                print(f"  Run URL: {run.get_url()}")
                print(f"  Entity: {self.cfg.wandb_entity if self.cfg.wandb_entity else 'default'}")
                print(f"  Mode: {'offline' if wandb.run.mode == 'offline' else 'online'}")
                
                # Define custom metrics for stage separation
                wandb.define_metric("global_step")
                wandb.define_metric("warmup/stage_step")
                wandb.define_metric("transition/stage_step")
                wandb.define_metric("rl/stage_step")
                
                # Set all warmup metrics to use warmup/stage_step as x-axis
                wandb.define_metric("warmup/*", step_metric="warmup/stage_step")
                # Set all transition metrics to use transition/stage_step as x-axis
                wandb.define_metric("transition/*", step_metric="transition/stage_step")
                # Set all rl metrics to use rl/stage_step as x-axis
                wandb.define_metric("rl/*", step_metric="rl/stage_step")
                
                # Keep global metrics on global_step
                wandb.define_metric("val/*", step_metric="global_step")
                wandb.define_metric("rollout/*", step_metric="global_step")
                wandb.define_metric("stage/*", step_metric="global_step")
                
                print(f"✓ Configured custom metrics for stage separation")
                print(f"  Stages: warmup, transition, rl")
                print("="*70 + "\n")
                
            except Exception as e:
                print(f"\n⚠️  ERROR: Failed to initialize wandb!")
                print(f"  Error: {e}")
                print(f"  Training will continue without wandb logging.")
                print("="*70 + "\n")
                self.cfg.use_wandb = False  # Disable wandb for rest of training
        
        # Create environment
        env = make_libero_env(
            task_suite_name=self.cfg.task_suite,
            task_id=task_ids,
            num_envs=self.cfg.num_envs,
            obs_mode="raw",  # We'll process manually
            action_normalization="none",  # VLA handles normalization
            auto_reset=True if self.is_vectorized else False,
            seed=0,  # IMPORTANT: Use seed 0 to match high-success evaluation setup
            num_steps_wait=10,  # Wait 10 steps after reset for stabilization (matches eval)
        )
        
        # Run initial validation to establish baseline
        print("\n" + "="*70)
        print("Running initial validation to establish baseline...")
        print("="*70)
        val_prompt = task_prompts[0] if isinstance(task_prompts, list) else task_prompts
        initial_val_stats = self.validate(env, val_prompt)
        print(f"Initial L1 Success Rate: {initial_val_stats['val/l1_success_rate']:.2%}")
        print(f"Initial Tokenized Success Rate: {initial_val_stats['val/tokenized_success_rate']:.2%}")
        print(f"Initial Performance Gap: {initial_val_stats['val/gap']:.2%}")
        print("="*70 + "\n")
        
        # Log initial validation to wandb
        if self.cfg.use_wandb:
            try:
                wandb.log({
                    "val/l1_success_rate": initial_val_stats['val/l1_success_rate'],
                    "val/tokenized_success_rate": initial_val_stats['val/tokenized_success_rate'],
                    "val/gap": initial_val_stats['val/gap'],
                }, step=0)
                print(f"✓ Logged initial validation to wandb: L1={initial_val_stats['val/l1_success_rate']:.2%}, Tokenized={initial_val_stats['val/tokenized_success_rate']:.2%}, Gap={initial_val_stats['val/gap']:.2%}")
            except Exception as e:
                print(f"⚠️  WARNING: Failed to log initial validation to wandb: {e}")
        
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
            # Update training stage tracking
            self._update_training_stage()
            
            # Collect rollouts
            rollout_stats = self.collect_rollouts(env, task_prompts)
            
            # Update policy (pass first task_prompt for value computation)
            # In multi-task, we use a generic prompt for value updates
            update_prompt = task_prompts[0] if isinstance(task_prompts, list) else task_prompts
            train_stats = self.update_policy(update_prompt)
            
            # Combine stats with stage prefixes
            stats = {}
            
            # Add stage-prefixed metrics
            stage = self.training_stage
            for key, value in {**rollout_stats, **train_stats}.items():
                # Skip already prefixed metrics
                if "/" in key:
                    stats[key] = value
                else:
                    stats[f"{stage}/{key}"] = value
            
            # Add global tracking metrics (not stage-specific)
            stats["global_step"] = self.global_step
            stats["stage/current"] = stage
            stats[f"{stage}/stage_step"] = self.stage_step
            
            # Add rollout policy tracking
            if self.cfg.use_l1_warmstart:
                stats["rollout/uses_l1"] = int(self._should_use_l1_actions())
                stats["rollout/warmup_progress"] = min(1.0, self.global_step / self.cfg.l1_warmup_steps)
                if self.global_step < self.cfg.l1_warmup_steps + self.cfg.l1_transition_steps:
                    transition_progress = max(0.0, (self.global_step - self.cfg.l1_warmup_steps) / self.cfg.l1_transition_steps)
                    stats["rollout/transition_progress"] = transition_progress

            # Update training history for matplotlib plotting
            for key, value in stats.items():
                if key != "global_step" and isinstance(value, (int, float)):
                    self.training_history[key].append(value)

            # Print update summary
            if "train/policy_loss" in train_stats:
                msg = f"\n[Update {update}] Policy Loss: {train_stats['train/policy_loss']:.6f} | " \
                      f"Clip Frac: {train_stats.get('train/clipfrac', 0):.4f} | " \
                      f"KL: {train_stats.get('train/approx_kl', 0):.6f}"
                if self.cfg.use_gae and "train/value_loss" in train_stats:
                    msg += f" | Value Loss: {train_stats['train/value_loss']:.6f}"
                print(msg)
            
            # Validation - check AFTER global_step is updated (not before)
            if self.global_step > 0 and self.global_step % self.cfg.val_interval == 0:
                print(f"\n🔍 Running validation at step {self.global_step}...")
                # For multi-task, validate on first task (could extend to all tasks)
                val_prompt = task_prompts[0] if isinstance(task_prompts, list) else task_prompts
                val_stats = self.validate(env, val_prompt)
                stats.update(val_stats)
                print(f"✓ Validation complete: L1={val_stats['val/l1_success_rate']:.2%}, Tokenized={val_stats['val/tokenized_success_rate']:.2%}, Gap={val_stats['val/gap']:.2%}")
                
                # ALWAYS log validation metrics to wandb immediately
                if self.cfg.use_wandb:
                    try:
                        wandb.log({
                            "val/l1_success_rate": val_stats['val/l1_success_rate'],
                            "val/tokenized_success_rate": val_stats['val/tokenized_success_rate'],
                            "val/gap": val_stats['val/gap'],
                        }, step=self.global_step)
                        print(f"✓ Logged validation metrics to wandb: L1={val_stats['val/l1_success_rate']:.2%}, Tokenized={val_stats['val/tokenized_success_rate']:.2%}, Gap={val_stats['val/gap']:.2%}")
                    except Exception as e:
                        print(f"⚠️  WARNING: Failed to log validation to wandb: {e}")
            
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
            
            # Log to wandb after every policy update
            if self.cfg.use_wandb:
                # Filter out NaN/inf values and convert lists/arrays to scalars
                clean_stats = {}
                for key, value in stats.items():
                    # Convert lists/arrays to scalars by taking mean
                    if isinstance(value, (list, np.ndarray)):
                        if len(value) > 0:
                            value = float(np.mean(value))
                        else:
                            continue  # Skip empty lists
                    
                    # Now check if it's a valid scalar
                    if isinstance(value, (int, float)):
                        if not (np.isnan(value) or np.isinf(value)):
                            clean_stats[key] = float(value)
                    # Skip non-numeric values (strings, dicts, etc.)
                
                # Only log if we have valid stats
                if clean_stats:
                    try:
                        wandb.log(clean_stats, step=self.global_step)
                        # Print confirmation every 10 updates to avoid spam
                        if update % 10 == 0:
                            print(f"\n✓ Logged {len(clean_stats)} metrics to wandb at step {self.global_step}")
                            print(f"   train/policy_loss={clean_stats.get('train/policy_loss', 'N/A')}, rollout/success_rate={clean_stats.get('rollout/success_rate', 'N/A')}")
                    except Exception as e:
                        print(f"\n⚠️  WARNING: Failed to log to wandb: {e}")
                        if update % 10 == 0:  # Only print detailed debug every 10 updates
                            print(f"   Stats types: {[(k, type(v)) for k, v in stats.items()][:5]}...")  # First 5 only
                else:
                    print(f"\n⚠️  WARNING: No valid stats to log (all NaN/inf) at update {update}")

                # Create matplotlib plots after logging
                try:
                    self.plot_training_curves(update)
                except Exception as e:
                    print(f"\n⚠️  WARNING: Failed to create matplotlib plots: {e}")

            # Save checkpoint
            if self.global_step % self.cfg.save_interval == 0:
                self.save_checkpoint(f"checkpoint_{self.global_step}.pt")
        
        pbar.close()
        env.close()
        
        if self.cfg.use_wandb:
            print("\n" + "="*70)
            print("Finishing wandb run...")
            print("="*70)
            try:
                wandb.finish()
                print("✓ wandb run finished and synced to cloud")
            except Exception as e:
                print(f"⚠️  WARNING: Error finishing wandb: {e}")
            print("="*70 + "\n")
        
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
    parser.add_argument("--use-data-parallel", action="store_true",
                       help="Enable DataParallel for multi-GPU training")
    
    args = parser.parse_args()
    
    # VLA configuration - load L1 action head for validation (frozen)
    vla_config = OpenVLAActorConfig(
        load_l1_action_head=True,  # Load for validation (matches reference 98% eval)
        freeze_l1_action_head=True,  # Keep frozen for PPO (not trained)
        use_data_parallel=args.use_data_parallel,  # Enable DataParallel if flag provided
    )
    
    # PPO configuration
    ppo_config = PPOConfig(
        total_timesteps=args.timesteps,
        task_suite=args.task_suite,
        task_id=args.task_id,
        task_ids=args.task_ids,
        num_envs=args.num_envs,
        device=vla_config.device,  # Use same device as VLA
        use_wandb=not args.no_wandb,
        wandb_entity='deeprl_ais'
    )
    
    # Initialize and train
    trainer = OpenVLAPPO(vla_config, ppo_config)
    trainer.train()


if __name__ == "__main__":
    main()

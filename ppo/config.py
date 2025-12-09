"""
PPO Configuration for OpenVLA Training
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class PPOConfig:
    """
    PPO (Proximal Policy Optimization) Training Configuration.
    
    This config controls all aspects of PPO training for fine-tuning OpenVLA
    on LIBERO robotics tasks using reinforcement learning.
    """
    
    # ===========================================
    # Training Hyperparameters
    # ===========================================
    
    total_timesteps: int = 100000
    """Total number of environment steps to collect during training.
    Example: 100,000 steps = ~195 updates with n_steps=500, or ~781 updates with n_steps=128.
    Typical values: 50k-1M for single task, 500k-5M for multi-task.
    """
    
    n_steps: int = 512
    """Number of environment steps to collect before each policy update (rollout length).
    This is NOT the episode length - episodes can span multiple updates.
    - Higher values (500-2048): More stable gradients, slower updates, better for complex tasks
    - Lower values (128-256): Faster updates, more exploration, better for simple tasks
    Trade-off: Large n_steps needs more memory but provides better advantage estimates.
    
    Set to 512 to ensure at least 1-2 episodes complete per rollout for sparse reward learning.
    LIBERO episodes typically need 200-600 actions, so 128 was too short.
    """
    
    batch_size: int = 2
    """Minibatch size for SGD updates during policy optimization.
    
    CRITICAL: DataParallel REPLICATES the model on each GPU (doesn't split it).
    Only the batch is split across GPUs.
    
    Memory usage per GPU with DataParallel:
    - Model replica: ~18-20GB (FULL model on EACH GPU)
    - Batch forward: Split (batch_size/2 samples per GPU)
    - Gradients: ~2-3GB (synchronized across GPUs)
    - Gradient checkpointing overhead: Extra activations stored
    
    batch_size=2 (current, SAFEST):
    - Each GPU: Model (18GB) + 1 sample (~1GB) + gradients (2GB) = ~21GB ✓
    - Safe for 24GB GPUs with memory fragmentation
    - Minimal parallelism but stable
    - 512/2 = 256 minibatches/epoch
    
    batch_size=4 (CAUSES OOM):
    - Each GPU: Model (18GB) + 2 samples (~1.5GB) + gradients (2GB) = ~21.5GB
    - With fragmentation and checkpointing overhead → OOM ✗
    
    Note: DataParallel doesn't provide significant speedup with batch_size=2
    since each GPU only processes 1 sample. Consider disabling DataParallel
    if memory is tight, or switching to pipeline parallelism.
    """
    
    n_epochs: int = 10
    """Number of passes through the collected data during each policy update.
    Each update, we iterate over the n_steps buffer this many times.
    - Higher epochs (10-20): More learning per sample, risk of overfitting
    - Lower epochs (3-5): Less overfitting, may need more timesteps
    Standard PPO uses 3-10 epochs.
    """
    
    # ===========================================
    # PPO Algorithm Hyperparameters
    # ===========================================
    
    actor_lr: float = 5e-7
    """Learning rate for the actor (VLA policy network).
    Lower than standard RL due to fine-tuning a pretrained 7B model.
    - 3e-7: Ultra-conservative, for very stable training
    - 5e-7: Very conservative, good balance of stability and learning speed ✓
    - 1e-6: Conservative, but can cause high clip fractions with large advantages
    - 5e-6: Too aggressive for 7B LoRA fine-tuning
    - 1e-5: Moderate, risk of catastrophic forgetting
    - 1e-4 to 3e-4: Standard RL, use only if training from scratch
    
    Reduced from 1e-6 to 5e-7 based on training analysis:
    - With 1e-6: clip fraction 0.8-0.99 (too aggressive)
    - Target: clip fraction 0.2-0.4 (stable learning)
    
    Adam optimizer is used for both actor and critic.
    """
    
    critic_lr: float = 3e-4
    """Learning rate for the critic (value head network)
    Can be higher than actor since value head trains from scratch.
    - 1e-4 to 5e-4: Standard range for critic learning
    - 3e-4: Default value used in many RL implementations
    """
    
    clip_ratio_high: float = 0.28
    """PPO upper clip ratio for policy updates (from SimpleVLA-RL reference).
    Asymmetric clipping allows more aggressive positive updates.
    - 0.2 to 0.3: Standard range for upper bound
    Formula: L^CLIP = min(r(θ)A, clip(r(θ), 1-ε_low, 1+ε_high)A)
    """
    
    clip_ratio_low: float = 0.2
    """PPO lower clip ratio for policy updates.
    More conservative clipping on negative side.
    - 0.1 to 0.2: Standard range for lower bound
    """
    
    gamma: float = 0.99
    """Discount factor for future rewards (γ in RL literature).
    Determines how much the agent values future vs immediate rewards.
    - 0.99: Standard for long-horizon tasks (values rewards ~100 steps ahead)
    - 0.95-0.98: For shorter horizon tasks
    - 0.999: For very long episodes (rare)
    Used in GAE and return computation.
    """
    
    gae_lambda: float = 0.95
    """Lambda parameter for Generalized Advantage Estimation (GAE-λ).
    NOTE: Currently unused - GRPO is used instead for advantage estimation.
    Kept for backward compatibility and future option to switch to GAE.
    - 1.0: High variance, low bias (Monte Carlo)
    - 0.0: Low variance, high bias (TD)
    - 0.95: Standard balanced value
    Formula: A^GAE = Σ(γλ)^t δ_t where δ_t = r_t + γV(s_{t+1}) - V(s_t)
    """
    
    verifier_gamma: float = 1.0
    """Discount factor for GRPO advantage estimation.
    Used for sparse reward propagation from finish_step.
    - 1.0: No discounting (standard for GRPO)
    - 0.99: Light discounting if needed
    """
    
    advantage_scale: float = 0.5
    """Scaling factor for advantages to control gradient magnitude.
    
    With sparse binary rewards (0 or 1), successful trajectories get advantage=1.0.
    This can cause large gradients and high clip fractions. Scaling reduces this:
    
    - 1.0: No scaling (default GRPO, but can cause clip fraction >0.9)
    - 0.5: Half-scale advantages (reduces aggressive updates) ✓
    - 0.3: Strong scaling (very conservative)
    - 0.1: Extreme scaling (too slow learning)
    
    Formula: scaled_advantage = advantage * advantage_scale
    
    Effect on training:
    - advantage_scale=1.0 → clip fraction 0.8-0.99 (unstable)
    - advantage_scale=0.5 → clip fraction 0.3-0.5 (target range)
    
    This is especially important when:
    - Success rate is high (80-100%) → most advantages are 1.0
    - Learning rate is conservative (5e-7) but gradients still large
    - Clip fraction consistently >0.7
    """
    
    entropy_coef: float = 0.01
    """Coefficient for entropy bonus in the loss function.
    Encourages exploration by penalizing overly deterministic policies.
    - 0.0: No entropy bonus (deterministic policy)
    - 0.01: Standard value for continuous control
    - 0.001-0.1: Typical range
    Note: Currently not used since VLA is deterministic, but kept for future stochastic versions.
    """
    
    # === L1 Warmstart Configuration ===
    use_l1_warmstart: bool = True
    """Enable L1 warmstart before transitioning to tokenized RL.
    When enabled, training proceeds in phases:
    1. Warmup (0-l1_warmup_steps): Execute L1 actions, train tokenized to match (behavior cloning)
    2. Transition (warmup_steps to warmup_steps+transition_steps): Gradually shift to tokenized
    3. RL (after transition): Execute tokenized actions, train with true PPO
    
    This prevents the tokenized head from learning poorly initially and enables
    true reinforcement learning once the tokenized policy is competent.
    """
    
    l1_warmup_steps: int = 25000
    """Number of steps to use L1 actions for rollouts (behavior cloning phase).
    During warmup:
    - L1 head generates high-quality actions (frozen)
    - Environment executes L1 actions
    - Tokenized head is trained to match L1 (supervised learning)
    - Expected tokenized success rate: 0% → 40%
    
    Recommended values:
    - 25000: Standard warmup (25k steps ≈ 500-1000 episodes)
    - 50000: Extended warmup for complex tasks
    - 0: Skip warmup (not recommended, tokenized starts from scratch)
    """
    
    l1_transition_steps: int = 10000
    """Number of steps for gradual transition from L1 to tokenized actions.
    Uses epsilon-greedy policy:
    - Start of transition: 100% L1, 0% tokenized
    - Middle: 50% L1, 50% tokenized
    - End: 0% L1, 100% tokenized
    
    This prevents catastrophic forgetting and ensures smooth handoff.
    Expected tokenized success rate during transition: 40% → 50%
    
    Recommended values:
    - 5000: Standard transition
    - 10000: Slower transition for stability
    - 0: Immediate switch (may cause performance drop)
    """
    
    value_loss_coef: float = 0.5
    """Coefficient for value function loss in the total loss.
    Balances policy gradient loss and value function loss.
    - 0.5: Standard value from PPO paper
    - 0.25-1.0: Typical range
    - Higher: More emphasis on accurate value predictions
    Formula: L_total = L_policy - c1*L_value + c2*H(policy)
    """
    
    max_grad_norm: float = 1.0
    """Maximum gradient norm for gradient clipping.
    Prevents exploding gradients during training.
    - 0.05: Ultra-tight, prevents any gradient explosions (7B LoRA with sparse rewards)
    - 0.1: Very conservative, prevents massive gradient explosions
    - 0.3: Conservative, prevents NaN losses (recommended for 7B models)
    - 0.5: Standard for large models (7B parameters)
    - 1.0: Standard for smaller models
    Applied via torch.nn.utils.clip_grad_norm_().
    """
    
    rollout_temperature: float = 1.0
    """Temperature for action sampling during rollout collection.
    Higher temperature = more exploration.
    - 1.0: Standard softmax sampling (better balance)
    - 1.6: High exploration (original, but too random for learning)
    - 0.0: Greedy (argmax) selection
    
    Reduced from 1.6 to 1.0 to reduce action randomness and improve success rate.
    """
    
    eval_temperature: float = 0.0
    """Temperature for action sampling during evaluation.
    Use greedy decoding for deterministic validation.
    - 0.0: Greedy (argmax) - recommended for eval
    - 1.0: Stochastic sampling
    """
    
    kl_coef: float = 0.0
    """Coefficient for KL divergence penalty in policy loss.
    KL(π_new || π_ref) penalizes divergence from reference policy.
    - 0.0: No KL penalty (reference setting)
    - 0.01-0.1: Light penalty for stability
    """
    
    traj_split_num: int = 4
    """Number of chunks to split trajectory for gradient accumulation.
    Reduces memory by processing trajectory in smaller segments.
    - 4: Reference setting, good balance
    - 2: Less splitting, higher memory
    - 8: More splitting, lower memory
    """
    
    traj_mini_batch_size: int = 8
    """Mini-batch size for trajectory processing.
    Number of trajectories to process together.
    - 16: Reference setting
    - 8: Lower memory
    - 32: Higher memory, more stable
    """
    
    separate_rollout_training: bool = False
    """Whether to separate rollout and training workers.
    If True, use separate GPU for rollout (faster inference via vLLM).
    If False, run both on same GPU (simpler, single-GPU mode).
    - False: Default, simpler setup
    - True: Advanced, requires multi-GPU setup
    """
    
    # ===========================================
    # Validation
    # ===========================================
    
    val_interval: int = 1024
    """Run validation every N environment steps (not updates).
    Validation runs val_episodes full episodes to measure success rate.
    - 1000-5000: Frequent validation, useful for debugging
    - 10000-50000: Less frequent, reduces overhead
    Set to 0 to disable validation during training.
    """
    
    val_episodes: int = 10
    """Number of episodes to run during each validation phase.
    More episodes give better success rate estimates but take longer.
    - 5-10: Quick validation
    - 20-50: More reliable metrics
    - 100: Comprehensive evaluation (slow)
    """
    
    # ===========================================
    # Logging and Checkpointing
    # ===========================================
    
    checkpoint_dir: str = "/home/abhi/Documents/Deep-RL/OpenVLA-OFT-RL/checkpoints"
    """Directory to save model checkpoints.
    
    Checkpoints include:
    - Periodic: checkpoint_stage_{stage}_update_{update}_step_{step}.pt
    - Best: best_model_stage_{stage}_success_{rate}.pt
    - Final: final_model_step_{step}.pt
    
    Default: "checkpoints" (relative to working directory)
    Can be absolute path: "/path/to/checkpoints"
    """
    
    save_interval: int = 5
    """Save checkpoint every N policy updates.
    
    Checkpoints are saved with naming pattern:
    checkpoint_stage_{stage}_update_{update}_step_{step}.pt
    
    Additionally, the best model (highest tokenized success rate) is saved as:
    best_model_stage_{stage}_success_{rate}.pt
    
    The final model at training completion is saved as:
    final_model_step_{step}.pt
    
    Default: 5 (save every 5 policy updates)
    """
    
    use_wandb: bool = True
    """Enable Weights & Biases logging for experiment tracking.
    Logs metrics like reward, success rate, value loss to wandb dashboard.
    Set to False to disable (useful for quick tests).
    """
    
    wandb_entity: Optional[str] = None
    """Weights & Biases entity (username or team name).
    - None: Uses your default wandb entity
    - 'username': Logs to personal account
    - 'team-name': Logs to team workspace
    Set this if you want to log to a specific entity.
    """
    
    # ===========================================
    # Environment Configuration
    # ===========================================
    
    task_suite: str = "libero_spatial"
    """LIBERO task suite to use for training.
    Available suites:
    - 'libero_spatial': 10 spatial reasoning tasks (pick and place)
    - 'libero_object': 10 object manipulation tasks
    - 'libero_goal': 10 goal-oriented tasks
    - 'libero_10': 10 diverse tasks
    Each suite has 10 tasks indexed 0-9.
    """
    
    task_ids: List[int] = None
    """List of task IDs for multi-task training (e.g., [0, 1, 2, 3]).
    If provided, trains on multiple tasks simultaneously with vectorized envs.
    - None: Single-task mode (uses task_id instead)
    - [0]: Single task in list form
    - [0, 1, 2, 3]: Multi-task training on 4 tasks
    Must match num_envs in length.
    """
    
    task_id: int = 0
    """Single task ID to train on (0-9 for each suite).
    Only used if task_ids is None (single-task mode).
    Each task has different objects, goals, and difficulty.
    See LIBERO documentation for task descriptions.
    """
    
    num_envs: int = 1
    """Number of parallel environments to run.
    In single-task mode: Must be 1
    In multi-task mode: Must equal len(task_ids)
    - 1: Single environment (slower, simpler)
    - 4-8: Good parallelization for multi-task
    - 16+: High parallelization (requires GPU memory)
    Each env runs independently and steps are collected in parallel.
    """
    
    obs_mode: str = "image_state"
    """Observation mode for LIBERO environments.
    - 'image_state': RGB image + robot proprioception (required for VLA)
    Other modes not supported for OpenVLA.
    """
    
    image_size: Tuple[int, int] = (224, 224)
    """Input image size (height, width) for the VLA model.
    OpenVLA is trained on 224x224 images.
    Do not change unless retraining vision backbone.
    """
    
    # ===========================================
    # Device Configuration
    # ===========================================
    
    device: str = "cuda:1"
    """Primary device for model and rollouts.
    - 'cuda:0': First GPU
    - 'cuda:1': Second GPU (default)
    - 'cpu': CPU only (very slow, not recommended)
    Must have CUDA-capable GPU with 16GB+ VRAM for 7B model.
    """
    
    training_device: str = "cuda:1"
    """Device for gradient computation during training updates.
    For single-GPU setup, should match 'device'.
    """
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Multi-task validation
        if self.task_ids is not None:
            if self.num_envs != len(self.task_ids):
                raise ValueError(
                    f"num_envs ({self.num_envs}) must equal len(task_ids) ({len(self.task_ids)}) "
                    f"for multi-task training"
                )
        else:
            # Single-task mode
            if self.num_envs != 1:
                raise ValueError(
                    f"num_envs must be 1 for single-task mode (got {self.num_envs}). "
                    f"Use --task-ids for multi-task training."
                )
        
        # Batch size validation
        if self.batch_size > self.n_steps:
            raise ValueError(
                f"batch_size ({self.batch_size}) cannot be larger than n_steps ({self.n_steps})"
            )
        
        # Device validation
        if self.training_device != self.device:
            print(f"⚠️  Warning: training_device ({self.training_device}) differs from device ({self.device})")
            print(f"   This is advanced dual-GPU mode - ensure tensors are moved correctly!")

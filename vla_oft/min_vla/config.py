# min_openvla/config.py
from dataclasses import dataclass

@dataclass
class OpenVLAActorConfig:
    """
    Minimal config to load OpenVLA-OFT 7B Libero-Spatial from local directory or HuggingFace Hub
    """
    # Model path: can be HuggingFace repo ID or local path (relative to vla-oft folder)
    # pretrained_checkpoint: str = "moojink/openvla-7b-oft-finetuned-libero-spatial"  # HuggingFace Hub
    pretrained_checkpoint: str = "openvla-7b-oft-finetuned-libero-spatial" # local path
    
    # Set to True to use local model, False to use HuggingFace Hub
    # If True and local directory exists, uses local; otherwise falls back to HF
    use_local: bool = True

    # GPU configuration
    use_multi_gpu: bool = True  # Enable to split model across GPUs
    gpu_id: int = 1  # Primary GPU for VLA backbone (vision + language model)
    secondary_gpu_id: int = 0  # Secondary GPU for training components (action head, value head, gradients)
    
    """
    Multi-GPU Memory Distribution Strategy:
    
    Single-GPU (use_multi_gpu=False) - Requires >32GB:
    - GPU 0: Everything (~32GB during training)
    
    Multi-GPU (use_multi_gpu=True) - Works with 2x24GB:
    - GPU 0 (~16-18GB):
      * VLA backbone (vision encoder + LLM): ~14GB
      * Forward pass activations: ~2-4GB
      
    - GPU 1 (~14-16GB):
      * Action head + Proprio projector: ~100MB
      * Value head (critic): ~5MB
      * Gradients + optimizer states: ~12-14GB
      * Training activations: ~2GB
    
    During training:
    1. Forward pass on GPU 0 (VLA with LoRA adapters)
    2. Transfer hidden states to GPU 1
    3. Action head forward + backward on GPU 1 (gradients only here)
    4. Value head forward + backward on GPU 1
    5. Optimizer updates on GPU 1 only
    
    Benefits:
    - GPU 0 stays light (backbone + LoRA, forward-only during inference)
    - GPU 1 handles all training overhead
    - Clean separation: inference vs training
    - LoRA gradients stored on GPU 0 but optimizer state on GPU 1
    
    IMPORTANT: With LoRA enabled, both GPUs will have some gradients:
    - GPU 0: LoRA adapter gradients (~2-4GB)
    - GPU 1: Action head + Value head gradients (~12-14GB)
    Total gradient memory is still manageable on 2x24GB.
    """
    
    # ===========================================
    # Device Properties (Auto-Generated)
    # ===========================================
    
    @property
    def device(self) -> str:
        """Primary device for VLA backbone (frozen during training).
        Returns: 'cuda:0' by default
        """
        return f"cuda:{self.gpu_id}" if self.gpu_id >= 0 else "cpu"
    
    @property
    def action_head_device(self) -> str:
        """Device for action head (trainable component).
        Single-GPU: Same as VLA backbone
        Multi-GPU: GPU 1 (secondary_gpu_id) for training
        """
        if self.use_multi_gpu:
            return f"cuda:{self.secondary_gpu_id}" if self.secondary_gpu_id >= 0 else "cpu"
        return self.device
    
    @property
    def proprio_projector_device(self) -> str:
        """Device for proprio projector.
        Single-GPU: Same as VLA backbone
        Multi-GPU: GPU 1 (keeps projector with action head)
        """
        if self.use_multi_gpu:
            return f"cuda:{self.secondary_gpu_id}" if self.secondary_gpu_id >= 0 else "cpu"
        return self.device
    
    @property
    def value_head_device(self) -> str:
        """Device for value head (critic, trainable for PPO).
        Always on same device as action head for efficient training.
        """
        return self.action_head_device
    
    @property
    def training_device(self) -> str:
        """Device where gradients and optimizer states live.
        Multi-GPU: GPU 1 (all trainable components + gradients)
        Single-GPU: Same as main device
        """
        return self.action_head_device

    # ===========================================
    # Training Configuration
    # ===========================================
    
    freeze_vla_backbone: bool = False
    """Freeze VLA backbone (vision + language model) during training.
    
    IMPORTANT: With LoRA enabled, gradients exist on BOTH GPUs:
    
    Multi-GPU with LoRA (use_multi_gpu=True, use_lora=True):
    - GPU 0 (~18-20GB):
      * VLA backbone: ~14GB
      * LoRA adapter parameters: ~200MB
      * LoRA gradients: ~2-4GB
      * Forward pass activations: ~2GB
      
    - GPU 1 (~16-18GB):
      * Action head: ~100MB
      * Value head: ~5MB  
      * Action head gradients: ~100MB
      * Value head gradients: ~5MB
      * Optimizer states: ~12-14GB (AdamW stores 2x params)
      * Training activations: ~2GB
    
    Total trainable parameters with LoRA:
    - LoRA adapters: ~110M params (1.45% of 7.6B)
    - Action head: ~167M params
    - Proprio projector: ~1.6M params
    - Value head: ~4.7M params
    Total: ~278M trainable parameters
    
    Gradient distribution:
    - LoRA adapters: Gradients stored on GPU 0 (but optimizer state can be on GPU 1)
    - Action/Value heads: Gradients on GPU 1
    
    - False (Recommended with LoRA for 2x24GB):
      * Use LoRA adapters to fine-tune language model efficiently
      * Only ~1-2% of model parameters trainable
      * Full model adaptation with minimal memory overhead
      
    - True (Fallback if LoRA fails):
      * Only train action head + value head
      * VLA backbone completely frozen (no gradients)
      * Memory: ~16GB on GPU 0, ~16GB on GPU 1
      * Faster but less expressive
    
    With LoRA enabled (use_lora=True), you can keep this False to get
    full model fine-tuning benefits with 2x24GB GPUs.
    """
    
    # ===========================================
    # LoRA Configuration
    # ===========================================
    
    use_lora: bool = True
    """Enable LoRA (Low-Rank Adaptation) for efficient full-model fine-tuning.
    
    LoRA allows fine-tuning the entire 7B language model with minimal memory:
    - Adds small trainable adapter weights to attention layers
    - Only 1-2% of model parameters trainable (~100-200M params)
    - Memory overhead: ~2-4GB for adapters + gradients
    - Performance: Nearly same as full fine-tuning
    
    Memory with LoRA (2x24GB GPUs):
    - GPU 0: VLA backbone (14GB) + activations (4GB) = ~18GB
    - GPU 1: Action head + Value head + LoRA adapters + gradients = ~20GB
    
    - True: Use LoRA adapters (recommended for 2x24GB)
    - False: Either freeze backbone or do full fine-tuning (requires >40GB)
    
    Requires: pip install peft
    """
    
    lora_rank: int = 32
    """LoRA rank (r) - controls adapter size and expressiveness.
    
    Higher rank = more parameters = better adaptation but more memory:
    - r=8: ~50M params, ~1GB memory, faster training
    - r=16: ~100M params, ~2GB memory, good balance
    - r=32: ~200M params, ~4GB memory, best quality (recommended)
    - r=64: ~400M params, ~8GB memory, overkill for most tasks
    
    Recommendation: 32 for robotics tasks (good quality/memory trade-off)
    """
    
    lora_alpha: int = 16
    """LoRA scaling factor (α) - controls adapter strength.
    
    Following OpenVLA finetune.py: α is capped at min(rank, 16)
    This means:
    - For rank ≤ 16: α = rank (standard 1x scaling)
    - For rank > 16: α = 16 (reduced scaling for stability)
    
    The effective learning rate for LoRA weights is: α / r
    - rank=32, α=16: scaling = 0.5x (stable, proven config)
    - rank=16, α=16: scaling = 1.0x (standard)
    - rank=8, α=8: scaling = 1.0x (standard)
    
    This value is automatically capped in code, so setting higher
    values here won't increase it beyond min(rank, 16).
    
    Recommendation: Keep at 16 (matches OpenVLA training)
    """
    
    lora_dropout: float = 0.0
    """LoRA dropout rate for regularization.
    
    Prevents overfitting by randomly dropping adapter weights:
    - 0.0: No dropout (faster, standard for OpenVLA)
    - 0.05: Light dropout
    - 0.1: Standard dropout
    
    OpenVLA finetune.py uses 0.0 by default.
    
    Recommendation: 0.0 (matches proven OpenVLA config)
    """
    
    lora_target_modules: str = "all-linear"
    """Which modules to apply LoRA adapters to.
    
    Following OpenVLA finetune.py approach:
    - "all-linear": Apply to all linear layers (recommended)
    
    This applies LoRA to:
    - All attention layers (Q, K, V, O projections)
    - All FFN layers (gate, up, down projections)
    - Vision encoder layers
    - Language model layers
    
    Recommendation: "all-linear" (matches OpenVLA training)
    """

    # ===========================================
    # Model Quantization
    # ===========================================
    
    # ===========================================
    # Model Quantization
    # ===========================================
    
    load_in_4bit: bool = False
    """Enable 4-bit quantization to reduce memory usage (quality/speed trade-off).
    
    4-bit quantized (load_in_4bit=True):
    - Memory: ~2GB VRAM (7x reduction)
    - Inference: ~116ms per action (8.6 Hz)
    - Quality: Slight degradation in accuracy
    - Use case: Low-memory systems, evaluation only
    
    Full precision bfloat16 (load_in_4bit=False):
    - Memory: ~14GB VRAM
    - Inference: ~53ms per action (18.8 Hz with Flash Attention)
    - Quality: Full model accuracy
    - Use case: Training, high-performance inference
    
    Recommendation: False for training (need gradients), True for memory-constrained eval.
    Note: Quantization uses bitsandbytes library, requires compatible CUDA.
    """

    # ===========================================
    # Proprioception (Robot State) Configuration
    # ===========================================
    
    use_proprio: bool = True
    """Enable proprioceptive (robot state) input to the model.
    
    LIBERO tasks require proprio information for spatial reasoning.
    Proprio is projected into the language model's embedding space and
    concatenated with visual features before feeding to the LLM.
    
    - True: Use robot state (required for LIBERO-trained models)
    - False: Vision-only mode (not compatible with LIBERO checkpoints)
    
    Keep True for all LIBERO tasks.
    """
    
    proprio_dim: int = 8
    """Expected proprioception dimension for the model.
    
    LIBERO models are trained with 8D proprio:
    - 3D: End-effector position (x, y, z)
    - 4D: End-effector orientation (quaternion w, x, y, z)
    - 1D: Gripper state (open/close, normalized)
    
    If environment observations have different dimensions:
    - More than 8D: Automatically clipped to first 8 dimensions
    - Less than 8D: Raises error (cannot pad safely)
    
    Note: LIBERO observations are actually 9D (2 gripper joints), but models
    expect 8D, so we clip to match training data format.
    
    Do not change unless retraining with different proprio format.
    """

    # ===========================================
    # Vision Input Configuration
    # ===========================================
    
    num_images_in_input: int = 2
    """Number of RGB images to use as input.
    
    OpenVLA processes RGB camera views for action prediction.
    - 1: Single camera (agentview only) - used for training to save memory
    - 2: Multi-camera (agentview + wrist camera) - REQUIRED for pretrained model
    
    Each image is resized to 224x224 and processed by the vision backbone
    (SigLIP-400M) to extract 256 patch embeddings.
    """

    # ===========================================
    # Action Head Configuration
    # ===========================================
    
    load_l1_action_head: bool = False
    """Load the L1 regression action head from checkpoint.
    
    The L1 regression head is an MLP that predicts continuous actions directly
    from hidden states: action = MLP(hidden_states) → continuous actions.
    
    This was used during supervised pre-training with L1 loss but is NOT needed
    for PPO training which uses tokenized actions.
    
    Training modes:
    - PPO (tokenized actions): Set to False (saves 668MB memory)
    - Supervised learning (L1 loss): Set to True
    - Inference after PPO: Set to False (use tokenized actions)
    - Inference with original checkpoint: Set to True
    
    Memory impact: ~167M parameters (~668MB)
    
    Default: False (optimized for PPO training)
    """
    
    freeze_l1_action_head: bool = True
    """Freeze L1 regression action head if loaded.
    
    If load_l1_action_head=True, this controls whether the head is trainable.
    
    - True: Freeze L1 head (read-only, for comparison or fallback inference)
    - False: Train L1 head (for continued supervised learning)
    
    Note: For PPO training, the L1 head is never used even if loaded,
    so this setting only matters if you're doing hybrid training.
    
    Default: True (freeze if loaded)
    """
    
    use_tokenized_actions: bool = True
    """Use tokenized action prediction (for PPO/RL training).
    
    Action prediction modes:
    - True: Extract action token logits from LM vocab, sample/argmax, detokenize
           Required for PPO training. Works for both training and inference.
    - False: Use L1 regression head for direct continuous action prediction
             Legacy mode from supervised pre-training.
    
    For PPO training: Must be True
    For inference after PPO: Should be True (same mode as training)
    For inference with original supervised checkpoint: Can use either
    
    Default: True (PPO mode)
    """
    
    finetuned_on_discrete_actions: bool = False
    """Whether the checkpoint was fine-tuned on discrete action tokens.
    
    LIBERO models use continuous actions, not discrete.
    - False: Continuous actions (LIBERO spatial/object/goal)
    - True: Discrete action bins (not used in LIBERO)
    
    This affects action head loading and action tokenizer behavior.
    Keep False for all LIBERO checkpoints.
    """

    # ===========================================
    # Inference Behavior
    # ===========================================
    
    deterministic_eval: bool = True
    """Use deterministic policy during evaluation (no sampling).
    
    - True: Always select mode/mean action (deterministic)
    - False: Sample from action distribution (stochastic)
    
    For PPO training with deterministic VLA:
    - Keep True (VLA outputs single action, no distribution)
    
    For stochastic policies (future work):
    - Set False to enable exploration via sampling
    """
    
    # ===========================================
    # Performance Optimizations
    # ===========================================
    
    use_flash_attention: bool = True
    """Enable Flash Attention 2 for faster and more memory-efficient attention.
    
    Flash Attention 2 optimizations:
    - 2-4x faster attention computation
    - 50% less memory usage during forward/backward
    - Exact attention (not approximate)
    
    Requirements:
    - flash-attn package installed (pip install flash-attn)
    - CUDA 11.8+ and compatible GPU (Ampere/Ada/Hopper)
    - PyTorch 2.0+
    
    Current setup: flash-attn 2.7.3 + torch 2.2.2 + CUDA 12.1
    
    Performance with Flash Attention 2:
    - Inference: ~53ms per action (18.8 Hz)
    - Without: ~80-100ms per action (10-12 Hz)
    
    Recommendation: True if flash-attn is installed, False otherwise.
    """

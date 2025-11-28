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
    use_multi_gpu: bool = False  # If False, all components on single GPU specified by gpu_id
    gpu_id: int = 0  # Primary GPU ID (used when use_multi_gpu=False, or for VLA model if True)
    
    # Multi-GPU setup (only used if use_multi_gpu=True)
    # VLA model goes on gpu_id, action_head and proprio_projector go on secondary_gpu_id
    secondary_gpu_id: int = 0  # GPU for action head and proprio projector in multi-GPU mode
    
    # Legacy device strings (auto-generated from gpu_id settings)
    @property
    def device(self) -> str:
        return f"cuda:{self.gpu_id}" if self.gpu_id >= 0 else "cpu"
    
    @property
    def action_head_device(self) -> str:
        if self.use_multi_gpu:
            return f"cuda:{self.secondary_gpu_id}" if self.secondary_gpu_id >= 0 else "cpu"
        return self.device
    
    @property
    def proprio_projector_device(self) -> str:
        if self.use_multi_gpu:
            return f"cuda:{self.secondary_gpu_id}" if self.secondary_gpu_id >= 0 else "cpu"
        return self.device

    # Quantization - trades memory for speed
    # 4-bit: ~2GB memory, ~116ms inference (8.6 Hz)
    # bfloat16: ~14GB memory, ~53ms inference (18.8 Hz with Flash Attention)
    load_in_4bit: bool = False  # Enable for low memory (slower inference)
    # load_in_8bit: bool = False  # Middle ground (not tested yet)

    # Libero tasks use proprio; keep True
    use_proprio: bool = True

    # We'll use standard OpenVLA resize/crop behavior
    num_images_in_input: int = 1  # 1 RGB image only

    # Don't use FiLM for the actor (FiLM is for critic in VLA-RL)
    # use_film: bool = False

    # For action head: we use L1 regression, not diffusion
    use_l1_regression: bool = True
    finetuned_on_discrete_actions: bool = False
    # use_diffusion: bool = False

    # Some PPO-friendly options
    deterministic_eval: bool = True
    
    # Flash Attention 2 (requires compatible flash-attn package)
    # Installed: flash-attn 2.7.3 (compatible with torch 2.2.2 + CUDA 12.1)
    use_flash_attention: bool = True  # Set to True to use flash_attention_2 (faster, lower memory)

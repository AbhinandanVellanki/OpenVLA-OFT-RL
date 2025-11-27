# min_openvla/config.py
from dataclasses import dataclass

@dataclass
class OpenVLAActorConfig:
    """
    Minimal config to load OpenVLA-OFT 7B Libero-Spatial from local directory.
    """
    # Local model directory (relative to vla-oft folder)
    pretrained_checkpoint: str = "openvla-7b-finetuned-libero-spatial" # using local checkpoint

    # Device configuration for multi-GPU setup
    device: str = "cuda:1"  # Main device for VLA model (~14GB)
    action_head_device: str = "cuda:0"  # Device for action head (~134MB)
    proprio_projector_device: str = "cuda:0"  # Device for proprio projector (~66MB)

    # Quantization (removing for now)
    # load_in_8bit: bool = False
    # load_in_4bit: bool = False

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
    
    # Flash Attention 2 (commented - requires flash-attn package)
    # use_flash_attention: bool = False  # Set to True to use flash_attention_2 (faster, lower memory)

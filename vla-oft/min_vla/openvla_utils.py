"""Utils for loading and using OpenVLA-OFT models."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

# add current directory to path so imports work
current_dir = Path(__file__).resolve().parent.parent
import sys
sys.path.append(str(current_dir))

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.projectors import ProprioProjector
from prismatic.vla.constants import ACTION_DIM

DEVICE = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
OPENVLA_IMAGE_SIZE = 224

def model_is_on_hf_hub(model_path: str) -> bool:
    """Check if model path points to HuggingFace Hub."""
    try:
        HfApi().model_info(model_path)
        return True
    except Exception:
        return False


def update_auto_map(pretrained_checkpoint: str) -> None:
    """Update AutoMap configuration in checkpoint config.json."""
    if not os.path.isdir(pretrained_checkpoint):
        return

    config_path = os.path.join(pretrained_checkpoint, "config.json")
    if not os.path.exists(config_path):
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    config["auto_map"] = {
        "AutoConfig": "configuration_prismatic.OpenVLAConfig",
        "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def find_checkpoint_file(pretrained_checkpoint: str, file_pattern: str) -> str:
    """Find checkpoint file matching pattern."""
    assert os.path.isdir(pretrained_checkpoint), f"Checkpoint path must be a directory: {pretrained_checkpoint}"

    checkpoint_files = [
        os.path.join(pretrained_checkpoint, f)
        for f in os.listdir(pretrained_checkpoint)
        if file_pattern in f and "checkpoint" in f
    ]

    assert len(checkpoint_files) == 1, (
        f"Expected exactly 1 {file_pattern} checkpoint but found {len(checkpoint_files)}"
    )

    return checkpoint_files[0]


def load_component_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load state dict and remove DDP prefix if present."""
    state_dict = torch.load(checkpoint_path, weights_only=True)
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def get_vla(cfg: Any) -> torch.nn.Module:
    """Load and initialize VLA model from checkpoint."""
    print("Loading pretrained VLA policy...")

    if not model_is_on_hf_hub(cfg.pretrained_checkpoint):
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        update_auto_map(cfg.pretrained_checkpoint)

    # Attention implementation selection
    # Default to "eager" to avoid _supports_sdpa compatibility issues
    attn_implementation = "eager"
    
    # Flash Attention 2 support (commented - uncomment if flash-attn is installed)
    # if hasattr(cfg, 'use_flash_attention') and cfg.use_flash_attention:
    #     try:
    #         import flash_attn
    #         attn_implementation = "flash_attention_2"
    #         print("Using Flash Attention 2 for faster inference")
    #     except ImportError:
    #         print("Warning: flash-attn not installed, falling back to eager attention")
    #         attn_implementation = "eager"
    
    # Alternative: SDPA (Scaled Dot Product Attention) - may have compatibility issues
    # attn_implementation = "sdpa"  # Uncomment to try SDPA (requires compatible transformers version)

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )

    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    vla.eval()

    if not (hasattr(cfg, 'load_in_8bit') and cfg.load_in_8bit) and \
       not (hasattr(cfg, 'load_in_4bit') and cfg.load_in_4bit):
        vla = vla.to(DEVICE)

    _load_dataset_stats(vla, cfg.pretrained_checkpoint)
    return vla


def _load_dataset_stats(vla: torch.nn.Module, checkpoint_path: str) -> None:
    """Load dataset statistics for action normalization."""
    if model_is_on_hf_hub(checkpoint_path):
        dataset_statistics_path = hf_hub_download(
            repo_id=checkpoint_path,
            filename="dataset_statistics.json",
        )
    else:
        dataset_statistics_path = os.path.join(checkpoint_path, "dataset_statistics.json")

    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            vla.norm_stats = json.load(f)
    else:
        print("WARNING: No dataset_statistics.json found. May cause errors when calling predict_action().")


def get_processor(cfg: Any) -> AutoProcessor:
    """Get VLA model's processor.
    This is the processor that will be used to preprocess the images before passing them to the VLA model.
    It is a PrismaticProcessor that is registered with the AutoProcessor class.
    """
    return AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)


def get_proprio_projector(cfg: Any, llm_dim: int, proprio_dim: int, device: Optional[torch.device] = None) -> ProprioProjector:
    """Get proprioception projector."""
    target_device = device if device is not None else DEVICE
    proprio_projector = ProprioProjector(llm_dim=llm_dim, proprio_dim=proprio_dim).to(target_device)
    proprio_projector = proprio_projector.to(torch.bfloat16).eval()

    if model_is_on_hf_hub(cfg.pretrained_checkpoint):
        model_path_to_proprio_projector_name = {
            "moojink/openvla-7b-oft-finetuned-libero-spatial": "proprio_projector--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-object": "proprio_projector--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-goal": "proprio_projector--50000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-10": "proprio_projector--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10": "proprio_projector--300000_checkpoint.pt",
        }
        if cfg.pretrained_checkpoint not in model_path_to_proprio_projector_name:
            raise ValueError("Unsupported HF Hub pretrained checkpoint!")
        proprio_projector_path = hf_hub_download(
            repo_id=cfg.pretrained_checkpoint,
            filename=model_path_to_proprio_projector_name[cfg.pretrained_checkpoint]
        )
        state_dict = load_component_state_dict(proprio_projector_path)
        proprio_projector.load_state_dict(state_dict)
    else:
        checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "proprio_projector")
        state_dict = load_component_state_dict(checkpoint_path)
        proprio_projector.load_state_dict(state_dict)

    return proprio_projector


def get_action_head(cfg: Any, llm_dim: int, device: Optional[torch.device] = None) -> L1RegressionActionHead:
    """Get L1 regression action head."""
    target_device = device if device is not None else DEVICE
    action_head = L1RegressionActionHead(input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM)
    action_head = action_head.to(torch.bfloat16).to(target_device).eval()

    if model_is_on_hf_hub(cfg.pretrained_checkpoint):
        model_path_to_action_head_name = {
            "moojink/openvla-7b-oft-finetuned-libero-spatial": "action_head--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-object": "action_head--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-goal": "action_head--50000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-10": "action_head--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10": "action_head--300000_checkpoint.pt",
        }
        if cfg.pretrained_checkpoint not in model_path_to_action_head_name:
            raise ValueError("Unsupported HF Hub pretrained checkpoint!")
        action_head_path = hf_hub_download(
            repo_id=cfg.pretrained_checkpoint,
            filename=model_path_to_action_head_name[cfg.pretrained_checkpoint]
        )
        state_dict = load_component_state_dict(action_head_path)
        action_head.load_state_dict(state_dict)
    else:
        checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "action_head")
        state_dict = load_component_state_dict(checkpoint_path)
        action_head.load_state_dict(state_dict)

    return action_head


def check_image_format(image: Any) -> None:
    """Validate input image format."""
    assert isinstance(image, np.ndarray), "Image must be numpy array"
    assert len(image.shape) == 3 and image.shape[-1] == 3, "Image must be (H, W, 3)"
    assert image.dtype == np.uint8, "Image must be uint8"


def prepare_images_for_vla(images: List[np.ndarray], cfg: Any) -> List[Image.Image]:
    """Prepare images for VLA input."""
    processed_images = []
    for image in images:
        check_image_format(image)
        if image.shape != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE, 3):
            from PIL import Image as PILImage
            pil_image = PILImage.fromarray(image).convert("RGB")
            pil_image = pil_image.resize((OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE), Image.Resampling.LANCZOS)
            processed_images.append(pil_image)
        else:
            processed_images.append(Image.fromarray(image).convert("RGB"))
    return processed_images


def get_vla_action(
    cfg: Any,
    vla: torch.nn.Module,
    processor: Any,
    obs: Dict[str, Any],
    task_label: str,
    action_head: Optional[torch.nn.Module] = None,
    proprio_projector: Optional[torch.nn.Module] = None,
) -> List[np.ndarray]:
    """Generate action predictions with VLA policy."""
    with torch.inference_mode():
        # Collect input images
        all_images = [obs["full_image"]]
        if cfg.num_images_in_input > 1:
            all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])

        # Process images
        all_images = prepare_images_for_vla(all_images, cfg)
        primary_image = all_images.pop(0)

        # Build prompt
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

        # Process primary image
        inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)

        # Process additional wrist images if any
        if all_images:
            all_wrist_inputs = [
                processor(prompt, image_wrist).to(DEVICE, dtype=torch.bfloat16) for image_wrist in all_images
            ]
            primary_pixel_values = inputs["pixel_values"]
            all_wrist_pixel_values = [wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs]
            inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

        # Process proprioception if used
        proprio = None
        if cfg.use_proprio:
            proprio = obs["state"]
            proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
            from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE
            from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType
            
            if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
                mask = proprio_norm_stats.get("mask", np.ones_like(proprio_norm_stats["min"], dtype=bool))
                proprio_high, proprio_low = np.array(proprio_norm_stats["max"]), np.array(proprio_norm_stats["min"])
            elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
                mask = proprio_norm_stats.get("mask", np.ones_like(proprio_norm_stats["q01"], dtype=bool))
                proprio_high, proprio_low = np.array(proprio_norm_stats["q99"]), np.array(proprio_norm_stats["q01"])
            else:
                raise ValueError("Unsupported normalization type!")

            proprio = np.clip(
                np.where(
                    mask,
                    2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
                    proprio,
                ),
                a_min=-1.0,
                a_max=1.0,
            )

        # Generate action
        if action_head is None:
            action, _ = vla.predict_action(**inputs, unnorm_key=cfg.unnorm_key, do_sample=False)
        else:
            action, _ = vla.predict_action(
                **inputs,
                unnorm_key=cfg.unnorm_key,
                do_sample=False,
                proprio=proprio,
                proprio_projector=proprio_projector,
                action_head=action_head,
            )

    return [action[i] for i in range(len(action))]

# min_openvla/actor.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from transformers import AutoProcessor

from .config import OpenVLAActorConfig
from . import openvla_utils  # your copied file


@dataclass
class OpenVLAActorState:
    vla: torch.nn.Module
    processor: AutoProcessor
    proprio_projector: Optional[torch.nn.Module]
    action_head: Optional[torch.nn.Module]
    device: torch.device


class OpenVLAActor:
    """
    PPO-facing wrapper around OpenVLA-OFT 7B Libero-Spatial checkpoint.

    Exposes a simple API:
        action, info = actor.forward(obs, task_prompt)
    where `obs` is a dict containing:
        - "image": HxWx3 uint8 numpy array or PIL Image
        - "proprio": (D,) numpy array of robot state (optional)
    """

    def __init__(self, cfg: OpenVLAActorConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # 1) Load VLA model from HF (this pulls modeling_prismatic, etc.)
        self.vla = openvla_utils.get_vla(cfg) # return vla model in eval mode

        # 2) Load HF processor (handles tokenization + image preprocessing)
        self.processor = openvla_utils.get_processor(cfg)

        # 3) Load proprio projector (if using proprio)
        self.proprio_projector = None
        if cfg.use_proprio:
            llm_dim = self.vla.llm_dim   # defined in modeling_prismatic.OpenVLAForActionPrediction
            # Libero spatial proprio dim is 8 (see constants or blog), but you can also infer
            proprio_dim = 8 # 7 joints + gripper
            self.proprio_projector = openvla_utils.get_proprio_projector(
                cfg, llm_dim=llm_dim, proprio_dim=proprio_dim
            )

        # 4) Load continuous action head
        self.action_head = None
        if cfg.use_l1_regression and not cfg.finetuned_on_discrete_actions:
            llm_dim = self.vla.llm_dim
            self.action_head = openvla_utils.get_action_head(cfg, llm_dim=llm_dim)

        self.state = OpenVLAActorState(
            vla=self.vla,
            processor=self.processor,
            proprio_projector=self.proprio_projector,
            action_head=self.action_head,
            device=self.device,
        )

    # ------------------------------------------------------------------ #
    # Public API: use this inside your PPO agent
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward(
        self,
        obs: Dict[str, Any],
        task_prompt: str,
        unnorm_key: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute a continuous action given observation + language instruction.

        Args:
            obs:
                {
                    "image": HxWx3 uint8 np.ndarray or PIL.Image.Image,
                    "proprio": (D,) np.ndarray (optional)
                }
            task_prompt: language instruction for the Libero task
            unnorm_key: if None, uses the only dataset stats in the model

        Returns:
            action: (ACTION_DIM,) np.ndarray in *unnormalized* robot space
            info: dict containing other useful stuff (raw hidden states, etc.)
        """
        image = obs["image"]
        proprio = obs.get("proprio", None)

        # Normalize image to PIL
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # 1) Use processor to get tokens + pixel_values
        #    For Libero they used the pattern: "IN: <instruction>\nOUT:"
        prompt = f"IN: {task_prompt}\nOUT:"
        proc_out = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        input_ids = proc_out["input_ids"].to(self.device)
        attention_mask = proc_out["attention_mask"].to(self.device)
        pixel_values = proc_out["pixel_values"].to(self.device, dtype=self.vla.dtype)  # Match model dtype (bfloat16)

        # 2) Call VLA's predict_action API (defined in modeling_prismatic)
        actions, action_hidden_states = self.vla.predict_action(
            input_ids=input_ids,
            unnorm_key=unnorm_key,
            proprio=proprio,
            proprio_projector=self.proprio_projector,
            action_head=self.action_head,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            use_film=False,  # FiLM is for critic in VLA-RL, not for actor
        )

        # actions is numpy array shape (NUM_ACTIONS_CHUNK, ACTION_DIM)
        # For Libero we typically only use the first action in the chunk
        action = actions[0] 

        info = {
            "raw_actions_chunk": actions,
            "action_hidden_states": action_hidden_states,
        }
        return action, info

    # Optional: function that returns torch tensor for PPO code
    @torch.no_grad()
    def act_torch(
        self,
        obs: Dict[str, Any],
        task_prompt: str,
        unnorm_key: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        action_np, info = self.forward(obs, task_prompt, unnorm_key)
        action = torch.from_numpy(action_np).to(self.device).float()
        return action, info

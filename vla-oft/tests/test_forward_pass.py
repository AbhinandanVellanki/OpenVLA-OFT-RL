# tests/test_forward_pass.py
import os
import sys
from pathlib import Path

# Add parent directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import torch

from min_vla.config import OpenVLAActorConfig
from min_vla.actor import OpenVLAActor


# Skip tests if no GPU or not enough memory could be a concern.
# You can relax this depending on your environment.
GPU_REQUIRED_ENV = os.environ.get("OPENVLA_TEST_GPU_REQUIRED", "1") == "1"


@pytest.fixture(scope="module")
def actor():
    """Shared actor fixture to avoid loading model multiple times."""
    cfg = OpenVLAActorConfig(device="cuda" if torch.cuda.is_available() else "cpu")
    actor_instance = OpenVLAActor(cfg)
    yield actor_instance
    # Cleanup: clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        del actor_instance


@pytest.mark.skipif(
    GPU_REQUIRED_ENV and not torch.cuda.is_available(),
    reason="GPU required for OpenVLA-OFT smoke test",
)
def test_actor_loads_and_moves_to_device():
    print("\n" + "="*60)
    print("TEST: test_actor_loads_and_moves_to_device")
    print("="*60)
    
    print("\n[1/5] Creating config...")
    cfg = OpenVLAActorConfig(device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"     Device: {cfg.device}")
    print(f"     Model: {cfg.pretrained_checkpoint}")
    
    print("\n[2/5] Initializing OpenVLAActor (this may take a while)...")
    actor = OpenVLAActor(cfg)
    print("     ✓ Actor initialized")

    print("\n[3/5] Checking components...")
    assert actor.vla is not None
    print("     ✓ VLA model loaded")
    
    assert actor.processor is not None
    print("     ✓ Processor loaded")
    
    if cfg.use_proprio:
        assert actor.proprio_projector is not None
        print("     ✓ Proprio projector loaded")
    
    assert actor.action_head is not None
    print("     ✓ Action head loaded")
    
    print("\n[4/5] Checking device placement...")
    device_type = next(actor.vla.parameters()).device.type
    assert device_type == cfg.device
    print(f"     ✓ Model is on {device_type} (expected {cfg.device})")
    
    print("\n[5/5] All checks passed! ✓")
    print("="*60 + "\n")
    
    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        del actor


@pytest.mark.skipif(
    GPU_REQUIRED_ENV and not torch.cuda.is_available(),
    reason="GPU required for OpenVLA-OFT smoke test",
)
def test_forward_on_dummy_observation(actor):
    print("\n" + "="*60)
    print("TEST: test_forward_on_dummy_observation")
    print("="*60)
    
    print("\n[1/4] Using shared actor...")
    assert actor is not None, "Actor fixture failed to provide actor instance"
    assert hasattr(actor, 'forward'), "Actor missing forward method"
    print("     ✓ Actor ready")

    print("\n[2/4] Preparing dummy observation...")
    dummy_image = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    dummy_proprio = np.zeros(8, dtype=np.float32)
    obs = {
        "image": dummy_image,
        "proprio": dummy_proprio,
    }
    task_prompt = "pick up the red cube and place it on the left platform."
    print(f"     ✓ Image shape: {dummy_image.shape}")
    print(f"     ✓ Proprio shape: {dummy_proprio.shape}")
    print(f"     ✓ Task prompt: {task_prompt[:50]}...")

    print("\n[3/4] Running forward pass...")
    action, info = actor.forward(obs, task_prompt)
    print("     ✓ Forward pass completed")

    print("\n[4/4] Validating outputs...")
    assert isinstance(action, np.ndarray)
    print(f"     ✓ Action is numpy array")
    
    assert action.shape == (7,)
    print(f"     ✓ Action shape: {action.shape} (expected (7,))")
    
    assert "raw_actions_chunk" in info
    assert "action_hidden_states" in info
    print(f"     ✓ Info dict contains required keys")
    print(f"     ✓ Action values: {action}")
    print("="*60 + "\n")


@pytest.mark.skipif(
    GPU_REQUIRED_ENV and not torch.cuda.is_available(),
    reason="GPU required for OpenVLA-OFT smoke test",
)
def test_deterministic_eval_mode(actor):
    print("\n" + "="*60)
    print("TEST: test_deterministic_eval_mode")
    print("="*60)
    
    print("\n[1/5] Using shared actor...")
    print("     ✓ Actor ready")

    print("\n[2/5] Preparing dummy observation...")
    dummy_image = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    dummy_proprio = np.zeros(8, dtype=np.float32)
    obs = {"image": dummy_image, "proprio": dummy_proprio}
    task_prompt = "pick up the red cube and place it on the left platform."
    print("     ✓ Observation prepared")

    print("\n[3/5] Setting model to eval mode...")
    actor.vla.eval()
    print("     ✓ Model in eval mode")

    print("\n[4/5] Running first forward pass...")
    action1, _ = actor.forward(obs, task_prompt)
    print(f"     ✓ First action: {action1}")
    
    print("\n[5/5] Running second forward pass (should be identical)...")
    action2, _ = actor.forward(obs, task_prompt)
    print(f"     ✓ Second action: {action2}")
    
    print("\n     Checking determinism...")
    assert np.allclose(action1, action2, atol=1e-6)
    max_diff = np.abs(action1 - action2).max()
    print(f"     ✓ Actions match (max difference: {max_diff:.2e})")
    print("="*60 + "\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s flag shows print statements

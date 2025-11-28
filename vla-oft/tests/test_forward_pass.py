# tests/test_forward_pass.py
import os
import sys
from pathlib import Path

# Add current directory to path so imports work
current_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(current_dir))

import pytest
import numpy as np
import torch
import time

from min_vla.config import OpenVLAActorConfig
from min_vla.actor import OpenVLAActor

# Skip tests if no GPU or not enough memory could be a concern.
# You can relax this depending on your environment.
GPU_REQUIRED_ENV = os.environ.get("OPENVLA_TEST_GPU_REQUIRED", "1") == "1"


@pytest.fixture(scope="module")
def actor():
    """Shared actor fixture to avoid loading model multiple times."""
    cfg = OpenVLAActorConfig(
        use_multi_gpu=True if torch.cuda.device_count() >= 2 else False,
        gpu_id=1 if torch.cuda.device_count() >= 2 else 0,
        secondary_gpu_id=0
    )
    actor_instance = OpenVLAActor(cfg)
    yield actor_instance
    # Cleanup: clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        del actor_instance


def print_gpu_memory():
    """Helper to print current GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"     GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


@pytest.mark.skipif(
    GPU_REQUIRED_ENV and not torch.cuda.is_available(),
    reason="GPU required for OpenVLA-OFT smoke test",
)
def test_actor_loads_and_moves_to_device():
    print("\n" + "="*60)
    print("TEST: test_actor_loads_and_moves_to_device")
    print("="*60)
    
    print("\n[1/7] Creating config...")
    cfg = OpenVLAActorConfig(
        use_multi_gpu=True if torch.cuda.device_count() >= 2 else False,
        gpu_id=1 if torch.cuda.device_count() >= 2 else 0,
        secondary_gpu_id=0
    )
    print(f"     Use Multi-GPU: {cfg.use_multi_gpu}")
    print(f"     GPU ID: {cfg.gpu_id}")
    if cfg.use_multi_gpu:
        print(f"     Secondary GPU ID: {cfg.secondary_gpu_id}")
    print(f"     VLA Device: {cfg.device}")
    print(f"     Action Head Device: {cfg.action_head_device}")
    print(f"     Proprio Device: {cfg.proprio_projector_device}")
    print(f"     Model: {cfg.pretrained_checkpoint}")
    
    print("\n[2/7] Initial GPU memory:")
    print_gpu_memory()
    
    print("\n[3/7] Initializing OpenVLAActor (this may take a while)...")
    actor = OpenVLAActor(cfg)
    print("     ✓ Actor initialized")
    
    print("\n[4/7] GPU memory after loading:")
    print_gpu_memory()

    print("\n[5/7] Checking components...")
    assert actor.vla is not None
    print("     ✓ VLA model loaded")
    
    assert actor.processor is not None
    print("     ✓ Processor loaded")
    
    if cfg.use_proprio:
        assert actor.proprio_projector is not None
        print("     ✓ Proprio projector loaded")
    
    assert actor.action_head is not None
    print("     ✓ Action head loaded")
    
    print("\n[6/7] Checking device placement...")
    vla_device = next(actor.vla.parameters()).device
    print(f"     ✓ VLA model is on {vla_device}")
    
    action_head_device = next(actor.action_head.parameters()).device
    print(f"     ✓ Action head is on {action_head_device}")
    
    if cfg.use_proprio:
        proprio_device = next(actor.proprio_projector.parameters()).device
        print(f"     ✓ Proprio projector is on {proprio_device}")
    
    print("\n[7/7] All checks passed! ✓")
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
    """Test basic forward pass - most memory efficient version"""
    print("\n" + "="*60)
    print("TEST: test_forward_on_dummy_observation (Memory Efficient)")
    print("="*60)
    
    print("\n[1/5] Using shared actor...")
    assert actor is not None, "Actor fixture failed to provide actor instance"
    assert hasattr(actor, 'forward'), "Actor missing forward method"
    print("     ✓ Actor ready")
    
    print("\n[2/5] GPU memory before inference:")
    print_gpu_memory()

    print("\n[3/5] Preparing dummy observation...")
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

    print("\n[4/5] Running forward pass with timing...")
    
    # Warm up
    with torch.inference_mode():
        _ = actor.forward(obs, task_prompt)
    if torch.cuda.is_available():
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
    
    # Timed forward pass
    start_time = time.time()
    with torch.inference_mode():  # More memory efficient than torch.no_grad()
        action, info = actor.forward(obs, task_prompt)
    
    # Synchronize to get accurate timing
    if torch.cuda.is_available():
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
    
    inference_time = time.time() - start_time
    inference_hz = 1.0 / inference_time if inference_time > 0 else 0
    
    print("     ✓ Forward pass completed")
    print(f"     ⏱️  Inference time: {inference_time*1000:.2f} ms ({inference_hz:.1f} Hz)")
    if inference_hz >= 100:
        print(f"     ✓ Achieves 100Hz target!")
    else:
        print(f"     ⚠️  Below 100Hz target (expected ~10ms, got {inference_time*1000:.2f}ms)")
    
    print("\n     GPU memory during/after inference:")
    print_gpu_memory()
    
    if torch.cuda.is_available():
        peak_gpu0 = torch.cuda.max_memory_allocated(0) / 1024**3
        peak_gpu1 = torch.cuda.max_memory_allocated(1) / 1024**3
        print(f"\n     Peak memory usage:")
        print(f"     GPU 0: {peak_gpu0:.2f} GB")
        print(f"     GPU 1: {peak_gpu1:.2f} GB")

    print("\n[5/5] Validating outputs...")
    assert isinstance(action, np.ndarray)
    print(f"     ✓ Action is numpy array")
    
    assert action.shape == (7,)
    print(f"     ✓ Action shape: {action.shape} (expected (7,))")
    
    # We no longer require hidden states in the info dict for basic testing
    assert "raw_actions_chunk" in info
    print(f"     ✓ Info dict contains raw_actions_chunk")
    print(f"     ✓ Action values: {action}")
    
    # Check if action values are reasonable (not NaN or Inf)
    assert not np.any(np.isnan(action)), "Action contains NaN values!"
    assert not np.any(np.isinf(action)), "Action contains Inf values!"
    print(f"     ✓ Action values are valid (no NaN/Inf)")
    
    print("="*60 + "\n")


@pytest.mark.skipif(
    GPU_REQUIRED_ENV and not torch.cuda.is_available(),
    reason="GPU required for OpenVLA-OFT smoke test",
)
def test_deterministic_eval_mode(actor):
    """Test that model produces deterministic outputs in eval mode"""
    print("\n" + "="*60)
    print("TEST: test_deterministic_eval_mode (Memory Efficient)")
    print("="*60)
    
    print("\n[1/6] Using shared actor...")
    print("     ✓ Actor ready")

    print("\n[2/6] Preparing dummy observation...")
    dummy_image = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    dummy_proprio = np.zeros(8, dtype=np.float32)
    obs = {"image": dummy_image, "proprio": dummy_proprio}
    task_prompt = "pick up the red cube and place it on the left platform."
    print("     ✓ Observation prepared")

    print("\n[3/6] Setting model to eval mode...")
    actor.vla.eval()
    print("     ✓ Model in eval mode")

    print("\n[4/6] Running first forward pass...")
    with torch.inference_mode():
        action1, _ = actor.forward(obs, task_prompt)
    print(f"     ✓ First action: {action1}")
    
    print("\n[5/6] Running second forward pass (should be identical)...")
    with torch.inference_mode():
        action2, _ = actor.forward(obs, task_prompt)
    print(f"     ✓ Second action: {action2}")
    
    print("\n[6/6] Checking determinism...")
    assert np.allclose(action1, action2, atol=1e-6)
    max_diff = np.abs(action1 - action2).max()
    print(f"     ✓ Actions match (max difference: {max_diff:.2e})")
    
    print("\n     Final GPU memory:")
    print_gpu_memory()
    
    print("="*60 + "\n")


@pytest.mark.skipif(
    GPU_REQUIRED_ENV and not torch.cuda.is_available(),
    reason="GPU required for OpenVLA-OFT smoke test",
)
def test_multiple_forward_passes_memory_stable(actor):
    """Test that memory doesn't grow with multiple forward passes"""
    print("\n" + "="*60)
    print("TEST: test_multiple_forward_passes_memory_stable")
    print("="*60)
    
    print("\n[1/4] Preparing dummy observation...")
    dummy_image = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    dummy_proprio = np.zeros(8, dtype=np.float32)
    obs = {"image": dummy_image, "proprio": dummy_proprio}
    task_prompt = "pick up the red cube and place it on the left platform."
    
    print("\n[2/4] Running 5 forward passes to warm up...")
    with torch.inference_mode():
        for i in range(5):
            action, _ = actor.forward(obs, task_prompt)
            if torch.cuda.is_available():
                torch.cuda.synchronize(0)
                torch.cuda.synchronize(1)
    
    print("\n[3/4] Measuring memory after warmup...")
    print_gpu_memory()
    
    if torch.cuda.is_available():
        baseline_gpu0 = torch.cuda.memory_allocated(0) / 1024**3
        baseline_gpu1 = torch.cuda.memory_allocated(1) / 1024**3
    
    print("\n[4/4] Running 10 more passes to check for memory leaks and benchmark speed...")
    inference_times = []
    with torch.inference_mode():
        for i in range(10):
            start = time.time()
            action, _ = actor.forward(obs, task_prompt)
            if torch.cuda.is_available():
                torch.cuda.synchronize(0)
                torch.cuda.synchronize(1)
            inference_times.append(time.time() - start)
    
    # Calculate statistics
    avg_time = np.mean(inference_times) * 1000  # ms
    std_time = np.std(inference_times) * 1000   # ms
    min_time = np.min(inference_times) * 1000   # ms
    max_time = np.max(inference_times) * 1000   # ms
    avg_hz = 1000.0 / avg_time
    
    print("\n     Inference timing statistics (10 runs):")
    print(f"     Average: {avg_time:.2f} ± {std_time:.2f} ms ({avg_hz:.1f} Hz)")
    print(f"     Min: {min_time:.2f} ms ({1000.0/min_time:.1f} Hz)")
    print(f"     Max: {max_time:.2f} ms ({1000.0/max_time:.1f} Hz)")
    
    if avg_hz >= 100:
        print(f"     ✓ Achieves 100Hz target! (avg {avg_hz:.1f} Hz)")
    else:
        print(f"     ⚠️  Below 100Hz target: {avg_hz:.1f} Hz (expected ≥100 Hz)")
    
    print("\n     Final memory:")
    print_gpu_memory()
    
    if torch.cuda.is_available():
        final_gpu0 = torch.cuda.memory_allocated(0) / 1024**3
        final_gpu1 = torch.cuda.memory_allocated(1) / 1024**3
        
        growth_gpu0 = final_gpu0 - baseline_gpu0
        growth_gpu1 = final_gpu1 - baseline_gpu1
        
        print(f"\n     Memory growth after 10 passes:")
        print(f"     GPU 0: {growth_gpu0:.3f} GB")
        print(f"     GPU 1: {growth_gpu1:.3f} GB")
        
        # Allow small growth for caching, but not linear growth
        assert abs(growth_gpu0) < 0.5, f"GPU 0 memory grew by {growth_gpu0:.2f} GB - possible leak!"
        assert abs(growth_gpu1) < 0.5, f"GPU 1 memory grew by {growth_gpu1:.2f} GB - possible leak!"
        print(f"     ✓ No significant memory leaks detected")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s flag shows print statements

"""
Standalone GAE tests (no pytest dependency)

Run directly with: python test_gae_standalone.py
"""

import numpy as np
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ppo.gae import compute_gae, compute_gae_batch, compute_gae_torch


def test_output_shapes():
    """Test that output shapes match input shapes."""
    print("Testing output shapes...")
    rewards = np.array([0.0, 0.0, 0.0, 1.0])
    values = np.array([0.5, 0.6, 0.7, 0.8])
    dones = np.array([False, False, False, True])

    advantages, returns = compute_gae(rewards, values, dones)

    assert advantages.shape == rewards.shape, f"Shape mismatch: {advantages.shape} != {rewards.shape}"
    assert returns.shape == rewards.shape, f"Shape mismatch: {returns.shape} != {rewards.shape}"
    print("✓ Output shapes correct")


def test_simple_trajectory():
    """Test GAE on a simple successful trajectory."""
    print("Testing simple trajectory...")
    rewards = np.array([0.0, 0.0, 0.0, 1.0])
    values = np.array([0.5, 0.6, 0.7, 0.8])
    dones = np.array([False, False, False, True])

    advantages, returns = compute_gae(
        rewards, values, dones, gamma=0.99, gae_lambda=0.95, normalize_advantages=False
    )

    # Verify returns are computed correctly: returns = advantages + values
    np.testing.assert_allclose(returns, advantages + values, rtol=1e-5)

    # Check that advantages are non-zero
    assert np.any(advantages != 0.0), "Advantages should not all be zero"

    # Final timestep should have positive advantage (got reward 1.0)
    assert advantages[-1] > 0.0, "Final advantage should be positive"
    print("✓ Simple trajectory test passed")


def test_normalization():
    """Test advantage normalization."""
    print("Testing normalization...")
    rewards = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    values = np.array([0.5, 0.6, 0.7, 0.8, 0.4, 0.5, 0.6, 0.7])
    dones = np.array([False, False, False, True, False, False, False, True])

    advantages, _ = compute_gae(
        rewards, values, dones, normalize_advantages=True
    )

    # After normalization, mean should be ~0 and std should be ~1
    mean = np.mean(advantages)
    std = np.std(advantages)
    assert abs(mean) < 1e-5, f"Mean should be ~0, got {mean}"
    assert abs(std - 1.0) < 1e-5, f"Std should be ~1, got {std}"
    print("✓ Normalization test passed")


def test_done_resets_gae():
    """Test that done=True resets GAE accumulation."""
    print("Testing episode boundaries...")
    # Two episodes concatenated
    rewards = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    values = np.array([0.5, 0.6, 0.7, 0.4, 0.5, 0.6])
    dones = np.array([False, False, True, False, False, True])

    advantages, _ = compute_gae(
        rewards, values, dones, gamma=0.99, gae_lambda=0.95, normalize_advantages=False
    )

    assert advantages.shape == (6,), f"Shape mismatch: {advantages.shape}"
    assert not np.all(advantages == 0.0), "Advantages should not all be zero"
    print("✓ Episode boundary test passed")


def test_all_zero_rewards():
    """Test trajectory with all zero rewards."""
    print("Testing all zero rewards...")
    rewards = np.zeros(4)
    values = np.array([0.5, 0.6, 0.7, 0.8])
    dones = np.array([False, False, False, True])

    advantages, returns = compute_gae(
        rewards, values, dones, normalize_advantages=False
    )

    # All rewards zero -> advantages should be negative (values overestimated)
    assert np.all(advantages <= 0.0), "Advantages should be non-positive for zero rewards"
    print("✓ All zero rewards test passed")


def test_nan_handling():
    """Test that NaN values are handled safely."""
    print("Testing NaN handling...")
    rewards = np.array([0.0, 0.0, np.nan, 1.0])
    values = np.array([0.5, 0.6, 0.7, 0.8])
    dones = np.array([False, False, False, True])

    advantages, returns = compute_gae(rewards, values, dones)

    # Verify no NaN in output
    assert not np.any(np.isnan(advantages)), "Advantages should not contain NaN"
    assert not np.any(np.isnan(returns)), "Returns should not contain NaN"
    print("✓ NaN handling test passed")


def test_batch_processing():
    """Test compute_gae_batch on multiple trajectories."""
    print("Testing batch processing...")
    # 2 trajectories of length 4
    rewards = np.array([
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    values = np.array([
        [0.5, 0.6, 0.7, 0.8],
        [0.5, 0.6, 0.7, 0.8],
    ])
    dones = np.array([
        [False, False, False, True],
        [False, False, False, True],
    ])

    advantages, returns = compute_gae_batch(
        rewards, values, dones, normalize_advantages=False
    )

    assert advantages.shape == (2, 4), f"Shape mismatch: {advantages.shape}"
    assert returns.shape == (2, 4), f"Shape mismatch: {returns.shape}"

    # First trajectory should have positive advantages
    assert np.any(advantages[0] > 0.0), "First trajectory should have positive advantages"
    # Second trajectory should have negative advantages (no reward)
    assert np.all(advantages[1] <= 0.0), "Second trajectory should have non-positive advantages"
    print("✓ Batch processing test passed")


def test_torch_cpu():
    """Test torch version on CPU."""
    print("Testing PyTorch CPU version...")
    rewards = torch.tensor([0.0, 0.0, 0.0, 1.0])
    values = torch.tensor([0.5, 0.6, 0.7, 0.8])
    dones = torch.tensor([False, False, False, True])

    advantages, returns = compute_gae_torch(
        rewards, values, dones, normalize_advantages=False
    )

    assert advantages.shape == torch.Size([4]), f"Shape mismatch: {advantages.shape}"
    assert returns.shape == torch.Size([4]), f"Shape mismatch: {returns.shape}"
    assert advantages.device == torch.device('cpu'), f"Device mismatch: {advantages.device}"
    print("✓ PyTorch CPU test passed")


def test_torch_matches_numpy():
    """Test that torch version produces same results as numpy."""
    print("Testing PyTorch vs NumPy consistency...")
    np_rewards = np.array([0.0, 0.0, 0.0, 1.0])
    np_values = np.array([0.5, 0.6, 0.7, 0.8])
    np_dones = np.array([False, False, False, True])

    torch_rewards = torch.tensor(np_rewards)
    torch_values = torch.tensor(np_values)
    torch_dones = torch.tensor(np_dones)

    np_adv, np_ret = compute_gae(
        np_rewards, np_values, np_dones, normalize_advantages=False
    )
    torch_adv, torch_ret = compute_gae_torch(
        torch_rewards, torch_values, torch_dones, normalize_advantages=False
    )

    np.testing.assert_allclose(np_adv, torch_adv.cpu().numpy(), rtol=1e-5)
    np.testing.assert_allclose(np_ret, torch_ret.cpu().numpy(), rtol=1e-5)
    print("✓ PyTorch vs NumPy consistency test passed")


def test_returns_decomposition():
    """Test that returns = advantages + values."""
    print("Testing returns decomposition...")
    rewards = np.array([0.1, 0.2, 0.3, 1.0])
    values = np.array([0.5, 0.6, 0.7, 0.8])
    dones = np.array([False, False, False, True])

    advantages, returns = compute_gae(
        rewards, values, dones, normalize_advantages=False
    )

    # Mathematical property: R_t = A_t + V(s_t)
    np.testing.assert_allclose(returns, advantages + values, rtol=1e-5)
    print("✓ Returns decomposition test passed")


def run_all_tests():
    """Run all tests and report results."""
    print("="*60)
    print("Running GAE Unit Tests")
    print("="*60)

    tests = [
        ("Output Shapes", test_output_shapes),
        ("Simple Trajectory", test_simple_trajectory),
        ("Normalization", test_normalization),
        ("Episode Boundaries", test_done_resets_gae),
        ("All Zero Rewards", test_all_zero_rewards),
        ("NaN Handling", test_nan_handling),
        ("Batch Processing", test_batch_processing),
        ("PyTorch CPU", test_torch_cpu),
        ("PyTorch vs NumPy", test_torch_matches_numpy),
        ("Returns Decomposition", test_returns_decomposition),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"✗ {name} failed: {e}")

    print("="*60)
    print(f"Test Results: {passed}/{len(tests)} passed")
    print("="*60)

    if failed > 0:
        print("\nFailed tests:")
        for name, error in errors:
            print(f"  - {name}: {error}")
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()

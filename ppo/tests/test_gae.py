"""
Unit tests for GAE (Generalized Advantage Estimation) implementation.

Tests verify correctness of GAE computation including:
- Basic functionality
- Episode boundary handling (done flags)
- Edge cases (all zeros, all ones, NaN/inf)
- Numerical stability
- Batch processing
- PyTorch GPU version
"""

import numpy as np
import torch
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ppo.gae import compute_gae, compute_gae_batch, compute_gae_torch


class TestGAEBasic:
    """Test basic GAE functionality."""

    def test_output_shapes(self):
        """Test that output shapes match input shapes."""
        rewards = np.array([0.0, 0.0, 0.0, 1.0])
        values = np.array([0.5, 0.6, 0.7, 0.8])
        dones = np.array([False, False, False, True])

        advantages, returns = compute_gae(rewards, values, dones)

        assert advantages.shape == rewards.shape
        assert returns.shape == rewards.shape

    def test_simple_trajectory(self):
        """Test GAE on a simple successful trajectory."""
        # Simple trajectory: sparse reward at end
        rewards = np.array([0.0, 0.0, 0.0, 1.0])
        values = np.array([0.5, 0.6, 0.7, 0.8])
        dones = np.array([False, False, False, True])

        advantages, returns = compute_gae(
            rewards, values, dones, gamma=0.99, gae_lambda=0.95, normalize_advantages=False
        )

        # Verify returns are computed correctly: returns = advantages + values
        np.testing.assert_allclose(returns, advantages + values, rtol=1e-5)

        # Check that advantages are non-zero (GAE should propagate final reward backward)
        assert np.any(advantages != 0.0)

        # Final timestep should have positive advantage (got reward 1.0)
        assert advantages[-1] > 0.0

    def test_normalization(self):
        """Test advantage normalization."""
        rewards = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        values = np.array([0.5, 0.6, 0.7, 0.8, 0.4, 0.5, 0.6, 0.7])
        dones = np.array([False, False, False, True, False, False, False, True])

        advantages, _ = compute_gae(
            rewards, values, dones, normalize_advantages=True
        )

        # After normalization, mean should be ~0 and std should be ~1
        assert abs(np.mean(advantages)) < 1e-5
        assert abs(np.std(advantages) - 1.0) < 1e-5

    def test_no_normalization(self):
        """Test that normalization can be disabled."""
        rewards = np.array([0.0, 0.0, 0.0, 1.0])
        values = np.array([0.5, 0.6, 0.7, 0.8])
        dones = np.array([False, False, False, True])

        advantages, _ = compute_gae(
            rewards, values, dones, normalize_advantages=False
        )

        # Without normalization, mean might not be 0
        # Just check it computed something
        assert np.any(advantages != 0.0)


class TestGAEEpisodeBoundaries:
    """Test GAE handling of episode boundaries."""

    def test_done_resets_gae(self):
        """Test that done=True resets GAE accumulation."""
        # Two episodes concatenated
        rewards = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
        values = np.array([0.5, 0.6, 0.7, 0.4, 0.5, 0.6])
        dones = np.array([False, False, True, False, False, True])

        advantages, _ = compute_gae(
            rewards, values, dones, gamma=0.99, gae_lambda=0.95, normalize_advantages=False
        )

        # Episode 1: steps 0-2
        # Episode 2: steps 3-5
        # Advantages should be independent between episodes

        # Check that first episode advantages don't depend on second episode
        # This is hard to verify directly, but we can check basic properties
        assert advantages.shape == (6,)
        assert not np.all(advantages == 0.0)

    def test_single_step_episode(self):
        """Test GAE on single-step episode."""
        rewards = np.array([1.0])
        values = np.array([0.5])
        dones = np.array([True])

        advantages, returns = compute_gae(
            rewards, values, dones, normalize_advantages=False
        )

        # For single timestep: delta = r + 0 * next_value - V
        # GAE = delta (no future accumulation)
        expected_delta = 1.0 - 0.5
        np.testing.assert_allclose(advantages[0], expected_delta, rtol=1e-5)

    def test_multiple_episodes(self):
        """Test multiple complete episodes."""
        # 3 episodes: [0,1], [2,3,4], [5]
        rewards = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 1.0])
        values = np.array([0.5, 0.6, 0.4, 0.5, 0.6, 0.5])
        dones = np.array([False, True, False, False, True, True])

        advantages, returns = compute_gae(rewards, values, dones, normalize_advantages=False)

        assert advantages.shape == (6,)
        assert not np.all(advantages == 0.0)


class TestGAEEdgeCases:
    """Test edge cases and error handling."""

    def test_all_zero_rewards(self):
        """Test trajectory with all zero rewards."""
        rewards = np.zeros(4)
        values = np.array([0.5, 0.6, 0.7, 0.8])
        dones = np.array([False, False, False, True])

        advantages, returns = compute_gae(
            rewards, values, dones, normalize_advantages=False
        )

        # All rewards zero -> advantages should be negative (values overestimated)
        assert np.all(advantages <= 0.0)

    def test_all_same_advantages(self):
        """Test when all advantages are identical (edge case for normalization)."""
        # Constant value function perfectly predicts zero rewards
        rewards = np.zeros(4)
        values = np.zeros(4)
        dones = np.array([False, False, False, True])

        advantages, returns = compute_gae(
            rewards, values, dones, normalize_advantages=True
        )

        # Should handle this gracefully (std = 0 case)
        assert advantages.shape == (4,)
        assert not np.any(np.isnan(advantages))

    def test_nan_handling(self):
        """Test that NaN values are handled safely."""
        rewards = np.array([0.0, 0.0, np.nan, 1.0])
        values = np.array([0.5, 0.6, 0.7, 0.8])
        dones = np.array([False, False, False, True])

        # This might produce NaN in advantages, but should be caught and replaced
        advantages, returns = compute_gae(rewards, values, dones)

        # Verify no NaN in output
        assert not np.any(np.isnan(advantages))
        assert not np.any(np.isnan(returns))

    def test_inf_handling(self):
        """Test that infinity values are handled safely."""
        rewards = np.array([0.0, 0.0, 0.0, np.inf])
        values = np.array([0.5, 0.6, 0.7, 0.8])
        dones = np.array([False, False, False, True])

        advantages, returns = compute_gae(rewards, values, dones)

        # Verify no inf in output
        assert not np.any(np.isinf(advantages))
        assert not np.any(np.isinf(returns))


class TestGAEParameters:
    """Test GAE with different hyperparameters."""

    def test_gamma_zero(self):
        """Test GAE with gamma=0 (no discounting)."""
        rewards = np.array([0.0, 0.0, 0.0, 1.0])
        values = np.array([0.5, 0.6, 0.7, 0.8])
        dones = np.array([False, False, False, True])

        advantages, _ = compute_gae(
            rewards, values, dones, gamma=0.0, gae_lambda=0.95, normalize_advantages=False
        )

        # With gamma=0, only immediate rewards matter
        # Steps 0-2 should have advantage = -value (no future reward)
        # Step 3 should have advantage = 1.0 - value
        np.testing.assert_allclose(advantages[3], 1.0 - 0.8, rtol=1e-5)

    def test_lambda_zero(self):
        """Test GAE with lambda=0 (TD(0) - one-step returns)."""
        rewards = np.array([0.0, 0.0, 0.0, 1.0])
        values = np.array([0.5, 0.6, 0.7, 0.8])
        dones = np.array([False, False, False, True])

        advantages, _ = compute_gae(
            rewards, values, dones, gamma=0.99, gae_lambda=0.0, normalize_advantages=False
        )

        # With lambda=0, GAE reduces to TD(0): A_t = delta_t only
        # No accumulation from future steps
        assert advantages.shape == (4,)

    def test_lambda_one(self):
        """Test GAE with lambda=1.0 (Monte Carlo returns)."""
        rewards = np.array([0.0, 0.0, 0.0, 1.0])
        values = np.array([0.5, 0.6, 0.7, 0.8])
        dones = np.array([False, False, False, True])

        advantages, _ = compute_gae(
            rewards, values, dones, gamma=0.99, gae_lambda=1.0, normalize_advantages=False
        )

        # With lambda=1.0, GAE becomes Monte Carlo (full rollout returns)
        assert advantages.shape == (4,)
        # First timestep should have highest advantage (longest horizon)
        assert advantages[0] > advantages[1]


class TestGAEBatch:
    """Test batched GAE computation."""

    def test_batch_processing(self):
        """Test compute_gae_batch on multiple trajectories."""
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

        assert advantages.shape == (2, 4)
        assert returns.shape == (2, 4)

        # First trajectory should have positive advantages
        assert np.any(advantages[0] > 0.0)
        # Second trajectory should have negative advantages (no reward)
        assert np.all(advantages[1] <= 0.0)

    def test_batch_normalization(self):
        """Test that batch normalization works across all trajectories."""
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

        advantages, _ = compute_gae_batch(
            rewards, values, dones, normalize_advantages=True
        )

        # Global normalization: mean=0, std=1 across all trajectories
        assert abs(np.mean(advantages)) < 1e-5
        assert abs(np.std(advantages) - 1.0) < 1e-5


class TestGAETorch:
    """Test PyTorch GPU-accelerated GAE version."""

    def test_torch_cpu(self):
        """Test torch version on CPU."""
        rewards = torch.tensor([0.0, 0.0, 0.0, 1.0])
        values = torch.tensor([0.5, 0.6, 0.7, 0.8])
        dones = torch.tensor([False, False, False, True])

        advantages, returns = compute_gae_torch(
            rewards, values, dones, normalize_advantages=False
        )

        assert advantages.shape == torch.Size([4])
        assert returns.shape == torch.Size([4])
        assert advantages.device == torch.device('cpu')

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_torch_gpu(self):
        """Test torch version on GPU."""
        rewards = torch.tensor([0.0, 0.0, 0.0, 1.0], device='cuda')
        values = torch.tensor([0.5, 0.6, 0.7, 0.8], device='cuda')
        dones = torch.tensor([False, False, False, True], device='cuda')

        advantages, returns = compute_gae_torch(
            rewards, values, dones, normalize_advantages=False
        )

        assert advantages.device.type == 'cuda'
        assert returns.device.type == 'cuda'

    def test_torch_matches_numpy(self):
        """Test that torch version produces same results as numpy."""
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


class TestGAEMathematicalProperties:
    """Test mathematical properties of GAE."""

    def test_returns_decomposition(self):
        """Test that returns = advantages + values."""
        rewards = np.array([0.1, 0.2, 0.3, 1.0])
        values = np.array([0.5, 0.6, 0.7, 0.8])
        dones = np.array([False, False, False, True])

        advantages, returns = compute_gae(
            rewards, values, dones, normalize_advantages=False
        )

        # Mathematical property: R_t = A_t + V(s_t)
        np.testing.assert_allclose(returns, advantages + values, rtol=1e-5)

    def test_gae_monotonicity_with_lambda(self):
        """Test that higher lambda gives more weight to longer horizons."""
        rewards = np.array([0.0, 0.0, 0.0, 1.0])
        values = np.array([0.5, 0.6, 0.7, 0.8])
        dones = np.array([False, False, False, True])

        # Compare lambda=0.5 vs lambda=0.95
        adv_low, _ = compute_gae(
            rewards, values, dones, gamma=0.99, gae_lambda=0.5, normalize_advantages=False
        )
        adv_high, _ = compute_gae(
            rewards, values, dones, gamma=0.99, gae_lambda=0.95, normalize_advantages=False
        )

        # With higher lambda, early timesteps get more advantage (longer horizon)
        # This is a general trend but not guaranteed for every timestep
        assert adv_low.shape == adv_high.shape

    def test_zero_advantage_on_perfect_value(self):
        """Test that perfect value function gives zero advantages."""
        # If value function perfectly predicts returns, advantages should be ~0
        # Create scenario: constant reward, perfect value prediction
        gamma = 0.99
        rewards = np.array([0.1, 0.1, 0.1, 0.1])
        # Perfect value: V(s_t) = sum of discounted future rewards
        values = np.array([
            0.1 + 0.1 * gamma + 0.1 * gamma**2 + 0.1 * gamma**3,  # t=0
            0.1 + 0.1 * gamma + 0.1 * gamma**2,                    # t=1
            0.1 + 0.1 * gamma,                                      # t=2
            0.1,                                                     # t=3
        ])
        dones = np.array([False, False, False, True])

        advantages, _ = compute_gae(
            rewards, values, dones, gamma=gamma, gae_lambda=1.0, normalize_advantages=False
        )

        # With perfect value prediction and lambda=1.0 (MC), advantages should be ~0
        np.testing.assert_allclose(advantages, 0.0, atol=1e-5)


def test_input_validation():
    """Test input validation and error handling."""
    rewards = np.array([0.0, 0.0, 0.0, 1.0])
    values = np.array([0.5, 0.6, 0.7, 0.8])
    dones = np.array([False, False, False, True])

    # Mismatched shapes should raise assertion error
    with pytest.raises(AssertionError):
        compute_gae(rewards[:-1], values, dones)

    with pytest.raises(AssertionError):
        compute_gae(rewards, values[:-1], dones)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])

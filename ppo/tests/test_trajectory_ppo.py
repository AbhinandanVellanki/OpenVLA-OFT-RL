"""
Test script for trajectory-based PPO implementation.

Tests basic functionality of action tokenization, trajectory buffer, and PPO algorithms.
"""

import numpy as np
import torch
from pathlib import Path
import sys
import os

# Add paths
vla_oft_path = Path(__file__).parent / "vla-oft"
sys.path.insert(0, str(vla_oft_path))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../vla_oft')))

from min_vla.action_tokenizer import ActionTokenizer
from ppo.trajectory_buffer import TrajectoryBuffer
from ppo.core_algos import logprobs_from_logits, compute_policy_loss, apply_mask_with_grad_control


def test_action_tokenizer():
    """Test action tokenization round-trip."""
    print("\n" + "="*70)
    print("Testing ActionTokenizer")
    print("="*70)
    
    tokenizer = ActionTokenizer(vocab_size=32000, n_bins=256)
    print(f"✓ Initialized: {tokenizer}")
    
    # Test single action
    action = np.array([0.5, -0.3, 0.8, -1.0, 1.0, 0.0, 0.2])
    tokens = tokenizer.discretize_actions(action)
    reconstructed = tokenizer.detokenize_actions(tokens)
    
    print(f"\nOriginal action:      {action}")
    print(f"Token IDs:            {tokens}")
    print(f"Reconstructed action: {reconstructed}")
    print(f"Reconstruction error: {np.abs(action - reconstructed).mean():.6f}")
    
    assert tokens.min() >= 31744 and tokens.max() < 32000, "Token IDs out of range!"
    assert np.abs(action - reconstructed).mean() < 0.01, "Reconstruction error too large!"
    
    # Test batch
    batch_actions = np.random.uniform(-1, 1, size=(10, 7))
    batch_tokens = tokenizer.discretize_actions(batch_actions)
    batch_reconstructed = tokenizer.detokenize_actions(batch_tokens)
    batch_error = np.abs(batch_actions - batch_reconstructed).mean()
    
    print(f"\nBatch reconstruction error: {batch_error:.6f}")
    assert batch_error < 0.01, "Batch reconstruction error too large!"
    
    print("✅ ActionTokenizer tests passed!")


def test_trajectory_buffer():
    """Test trajectory buffer storage and retrieval."""
    print("\n" + "="*70)
    print("Testing TrajectoryBuffer")
    print("="*70)
    
    buffer = TrajectoryBuffer()
    print("✓ Initialized TrajectoryBuffer")

    # Simulate 3 trajectories with random lengths, some ending with partial chunks
    rng = np.random.default_rng(42)
    trajectory_lengths = [17, 8, 13]  # All less than 24, so some partial chunks
    device = torch.device("cpu")

    for traj_idx, traj_len in enumerate(trajectory_lengths):
        print(f"\nSimulating stochastic trajectory {traj_idx+1} (length={traj_len})...")
        step = 0
        while step < traj_len:
            # Determine chunk size: full (8) or partial (remaining)
            chunk_size = min(8, traj_len - step)
            # Dummy data for chunk
            for i in range(chunk_size):
                obs = {"image": np.zeros((224, 224, 3), dtype=np.uint8), "proprio": np.zeros(8)}
                responses = torch.randint(31744, 32000, (7,))  # 7 dims per action
                input_ids = torch.randint(0, 32000, (1, 50))
                attention_mask = torch.ones((1, 50))
                pixel_values = torch.randn((1, 3, 224, 224))
                proprio = np.zeros(8)
                action = np.random.uniform(-1, 1, 7)
                reward = 1.0 if (step + i) == traj_len - 1 else 0.0
                done = ((step + i) == traj_len - 1)
                value = 0.5
                log_prob = torch.tensor(-2.0)
                buffer.add(
                    obs=obs,
                    responses=responses,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    proprio=proprio,
                    action=action,
                    l1_action=None,  # Stochastic rollout
                    reward=reward,
                    done=done,
                    value=value,
                    old_log_prob=log_prob,
                )
            step += chunk_size

    print(f"\n✓ Added {len(buffer)} stochastic trajectories with variable chunk sizes")

    # Test trajectory mask generation for each trajectory
    for idx, traj in enumerate(buffer.trajectories):
        mask = buffer.generate_traj_mask(traj['traj_len'], traj['finish_step'], device)
        assert mask.sum() == traj['traj_len'], f"Mask wrong size for trajectory {idx}: {mask.sum()}"
    print("✓ Trajectory masks generated correctly for all trajectories")

    # Compute advantages
    buffer.compute_advantages(gamma=0.99, verifier_gamma=1.0)
    print("✓ GRPO advantages computed")

    # Get data
    data = buffer.get()

    print(f"\nBuffer statistics:")
    print(f"  - Total steps: {len(data['observations'])}")
    print(f"  - Actions shape: {data['actions'].shape}")
    print(f"  - Advantages shape: {data['advantages'].shape}")
    print(f"  - Rewards (non-zero): {(data['rewards'] != 0).sum()}/{len(data['rewards'])}")

    # Check that total steps matches sum of trajectory lengths
    expected_steps = sum(trajectory_lengths)
    assert len(data['observations']) == expected_steps, f"Wrong total steps! Expected {expected_steps}, got {len(data['observations'])}"
    assert (data['rewards'] != 0).sum() == len(trajectory_lengths), f"Wrong number of non-zero rewards!"

    # Check that actions shape matches expected (variable chunks)
    assert data['actions'].shape[0] == expected_steps, f"Actions shape mismatch!"

    print("✅ Stochastic TrajectoryBuffer variable chunk tests passed!")


def test_ppo_algorithms():
    """Test PPO algorithm functions."""
    print("\n" + "="*70)
    print("Testing PPO Algorithms")
    print("="*70)
    
    # Test logprobs_from_logits
    batch_size, seq_len, vocab_size = 4, 10, 256
    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    log_probs = logprobs_from_logits(logits, token_ids)
    
    print(f"✓ logprobs_from_logits: input {logits.shape} → output {log_probs.shape}")
    assert log_probs.shape == (batch_size, seq_len), "Wrong log_probs shape!"
    assert torch.all(log_probs <= 0), "Log probs should be negative!"
    assert log_probs.requires_grad, "Log probs should inherit gradients from logits!"
    
    # Test compute_policy_loss
    old_log_probs = torch.randn(batch_size, seq_len) - 2.0  # Negative log probs (from buffer, no grad)
    new_log_probs = torch.randn(batch_size, seq_len, requires_grad=True) - 2.0  # From policy, needs grad
    advantages = torch.randn(batch_size, seq_len)  # Computed values, no grad
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    policy_loss, clipfrac, approx_kl = compute_policy_loss(
        old_log_probs, new_log_probs, advantages, mask,
        clip_ratio_high=0.28, clip_ratio_low=0.2
    )
    
    print(f"✓ compute_policy_loss:")
    print(f"  - Policy loss: {policy_loss.item():.4f}")
    print(f"  - Clip fraction: {clipfrac.item():.4f}")
    print(f"  - Approx KL: {approx_kl.item():.4f}")
    
    assert policy_loss.requires_grad, "Policy loss should have gradients!"
    assert 0 <= clipfrac <= 1, "Clip fraction out of range!"
    
    # Test apply_mask_with_grad_control
    tensor = torch.randn(batch_size, seq_len, 10, requires_grad=True)
    mask = torch.rand(batch_size, seq_len) > 0.5
    
    masked = apply_mask_with_grad_control(tensor, mask)
    loss = masked.sum()
    loss.backward()
    
    print(f"✓ apply_mask_with_grad_control: gradients preserved")
    assert tensor.grad is not None, "Gradients should flow through mask!"
    
    print("✅ PPO algorithm tests passed!")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("TRAJECTORY-BASED PPO IMPLEMENTATION TESTS")
    print("="*70)
    
    try:
        test_action_tokenizer()
        test_trajectory_buffer()
        test_ppo_algorithms()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nImplementation is ready for integration testing.")
        print("Next steps:")
        print("  1. Test trajectory collection with real environment")
        print("  2. Test policy update with real VLA model")
        print("  3. Run small-scale training (100 steps)")
        print("  4. Monitor for memory issues and NaN losses")
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TESTS FAILED!")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

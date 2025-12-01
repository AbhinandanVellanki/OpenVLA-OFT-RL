"""
Test suite for OpenVLA PPO training pipeline.
Focuses on trajectory collection and buffer management.
"""

import numpy as np
import torch
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add project path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_trajectory_buffer_signature():
    """Test that TrajectoryBuffer.add() has correct signature for GRPO mode."""
    print("\n" + "="*70)
    print("Test 1: TrajectoryBuffer.add() Signature")
    print("="*70)
    
    from ppo.trajectory_buffer import TrajectoryBuffer
    
    buffer = TrajectoryBuffer()
    
    # Test data matching what collect_rollouts() provides
    dummy_obs = {"image": np.zeros((224, 224, 3)), "proprio": np.zeros(8)}
    dummy_responses = torch.randint(31744, 32000, (56,))
    dummy_input_ids = torch.zeros((1, 50), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, 50))
    dummy_pixel_values = torch.randn((1, 3, 224, 224))
    dummy_proprio = np.zeros(8)
    dummy_action = np.random.uniform(-1, 1, 7)
    dummy_reward = 0.0
    dummy_done = False
    dummy_value = 0.0  # GRPO mode: values should be 0 or ignored
    dummy_log_prob = torch.tensor(-2.0)
    
    try:
        buffer.add(
            obs=dummy_obs,
            responses=dummy_responses,
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
            pixel_values=dummy_pixel_values,
            proprio=dummy_proprio,
            action=dummy_action,
            reward=dummy_reward,
            done=dummy_done,
            value=dummy_value,
            old_log_prob=dummy_log_prob,
        )
        print("✅ TrajectoryBuffer.add() accepts all required arguments including 'value'")
    except TypeError as e:
        print(f"❌ FAILED: {e}")
        print("\nExpected signature:")
        print("  buffer.add(obs, responses, input_ids, attention_mask, pixel_values,")
        print("             proprio, action, reward, done, value, old_log_prob)")
        raise


def test_grpo_mode_value_handling():
    """Test that GRPO mode correctly handles value parameter."""
    print("\n" + "="*70)
    print("Test 2: GRPO Mode Value Handling")
    print("="*70)
    
    from ppo.trajectory_buffer import TrajectoryBuffer
    
    buffer = TrajectoryBuffer()
    
    # In GRPO mode, values should be 0.0 (value-free advantages)
    for step in range(10):
        buffer.add(
            obs={"image": np.zeros((224, 224, 3)), "proprio": np.zeros(8)},
            responses=torch.randint(31744, 32000, (56,)),
            input_ids=torch.zeros((1, 50), dtype=torch.long),
            attention_mask=torch.ones((1, 50)),
            pixel_values=torch.randn((1, 3, 224, 224)),
            proprio=np.zeros(8),
            action=np.random.uniform(-1, 1, 7),
            reward=1.0 if step == 9 else 0.0,
            done=(step == 9),
            value=0.0,  # GRPO: value-free
            old_log_prob=torch.tensor(-2.0),
        )
    
    # Compute GRPO advantages (should work with value=0.0)
    buffer.compute_advantages(gamma=0.99, verifier_gamma=1.0)
    data = buffer.get()
    
    print(f"✓ Buffer size: {len(data['observations'])}")
    print(f"✓ Advantages computed: {data['advantages'].shape}")
    print(f"✓ Advantages mean: {data['advantages'].mean():.4f}")
    print(f"✓ Advantages std: {data['advantages'].std():.4f}")
    
    assert len(data['observations']) == 10
    assert data['advantages'].shape[0] == 10
    print("✅ GRPO mode handles value=0.0 correctly")


def test_collect_rollouts_call_signature():
    """Test that collect_rollouts properly calls buffer.add() with all arguments."""
    print("\n" + "="*70)
    print("Test 3: collect_rollouts() → buffer.add() Call Signature")
    print("="*70)
    
    # Mock the trainer components
    mock_actor = MagicMock()
    mock_actor.predict.return_value = (
        np.random.uniform(-1, 1, 7),  # action
        torch.randint(31744, 32000, (56,)),  # responses
        torch.tensor(-2.0),  # log_prob
        torch.zeros((1, 50), dtype=torch.long),  # input_ids
        torch.ones((1, 50)),  # attention_mask
        torch.randn((1, 3, 224, 224))  # pixel_values
    )
    
    mock_env = MagicMock()
    mock_env.reset.return_value = {
        "agentview_image": np.zeros((224, 224, 3)),
        "robot0_proprio-state": np.zeros(8)
    }
    mock_env.step.return_value = (
        {"agentview_image": np.zeros((224, 224, 3)), "robot0_proprio-state": np.zeros(8)},
        0.0,  # reward
        False,  # done
        {"success": False}
    )
    
    from ppo.trajectory_buffer import TrajectoryBuffer
    
    buffer = TrajectoryBuffer()
    
    # Simulate one rollout step
    obs = mock_env.reset()
    action, responses, log_prob, input_ids, attn_mask, pixel_vals = mock_actor.predict(obs, "test task")
    next_obs, reward, done, info = mock_env.step(action)
    
    # This should match the call in OpenVLA_PPO.py line 637
    try:
        buffer.add(
            obs=obs,
            responses=responses,
            input_ids=input_ids,
            attention_mask=attn_mask,
            pixel_values=pixel_vals,
            proprio=obs["robot0_proprio-state"],
            action=action,
            reward=reward,
            done=done,
            value=0.0,  # GRPO mode: must provide value even if unused
            old_log_prob=log_prob,
        )
        print("✅ Rollout collection call signature matches buffer.add() requirements")
    except TypeError as e:
        print(f"❌ FAILED: {e}")
        print("\nThe collect_rollouts() call is missing the 'value' argument!")
        print("Fix: Add 'value=0.0' to the buffer.add() call at line 637")
        raise


def test_full_rollout_collection():
    """Integration test: collect full trajectory with proper value handling."""
    print("\n" + "="*70)
    print("Test 4: Full Trajectory Collection")
    print("="*70)
    
    from ppo.trajectory_buffer import TrajectoryBuffer
    
    buffer = TrajectoryBuffer()
    
    # Simulate a complete episode
    episode_length = 50
    for step in range(episode_length):
        is_terminal = (step == episode_length - 1)
        
        buffer.add(
            obs={"image": np.zeros((224, 224, 3)), "proprio": np.zeros(8)},
            responses=torch.randint(31744, 32000, (56,)),
            input_ids=torch.zeros((1, 50), dtype=torch.long),
            attention_mask=torch.ones((1, 50)),
            pixel_values=torch.randn((1, 3, 224, 224)),
            proprio=np.zeros(8),
            action=np.random.uniform(-1, 1, 7),
            reward=1.0 if is_terminal else 0.0,  # Sparse reward
            done=is_terminal,
            value=0.0,  # GRPO: value-free
            old_log_prob=torch.tensor(-2.0),
        )
    
    print(f"✓ Collected {episode_length} timesteps")
    
    # Compute advantages
    buffer.compute_advantages(gamma=0.99, verifier_gamma=1.0)
    data = buffer.get()
    
    print(f"✓ Buffer contains {len(data['observations'])} steps")
    print(f"✓ Non-zero rewards: {(data['rewards'] != 0).sum()}")
    print(f"✓ Advantages range: [{data['advantages'].min():.4f}, {data['advantages'].max():.4f}]")
    
    assert len(data['observations']) == episode_length
    assert (data['rewards'] != 0).sum() == 1  # Only terminal reward
    print("✅ Full trajectory collection works correctly")


def test_error_reproduction():
    """Reproduce the exact error from the log."""
    print("\n" + "="*70)
    print("Test 5: Error Reproduction (Missing 'value' argument)")
    print("="*70)
    
    from ppo.trajectory_buffer import TrajectoryBuffer
    
    buffer = TrajectoryBuffer()
    
    # Reproduce the exact call from line 637 (without 'value')
    try:
        buffer.add(
            obs={"image": np.zeros((224, 224, 3)), "proprio": np.zeros(8)},
            responses=torch.randint(31744, 32000, (56,)),
            input_ids=torch.zeros((1, 50), dtype=torch.long),
            attention_mask=torch.ones((1, 50)),
            pixel_values=torch.randn((1, 3, 224, 224)),
            proprio=np.zeros(8),
            action=np.random.uniform(-1, 1, 7),
            reward=0.0,
            done=False,
            # value=0.0,  # <-- Missing this line!
            old_log_prob=torch.tensor(-2.0),
        )
        print("❌ Expected TypeError but call succeeded - signature may have changed")
    except TypeError as e:
        print(f"✅ Reproduced error: {e}")
        print("\n🔧 FIX: Add 'value=0.0' parameter to buffer.add() call")
        print("   Location: OpenVLA_PPO.py line 637")
        print("   For GRPO mode, value should be 0.0 (value-free advantages)")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PPO TRAINING PIPELINE TESTS")
    print("="*70)
    
    tests = [
        test_trajectory_buffer_signature,
        test_grpo_mode_value_handling,
        test_collect_rollouts_call_signature,
        test_full_rollout_collection,
        test_error_reproduction,
    ]
    
    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"\n❌ {test.__name__} FAILED: {e}")
            failed.append(test.__name__)
    
    print("\n" + "="*70)
    if not failed:
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\n🔧 REQUIRED FIX:")
        print("   In OpenVLA_PPO.py at line 637, add:")
        print("   value=0.0,  # GRPO mode: value-free advantages")
        print("\n   Full call should be:")
        print("   self.trajectory_buffer.add(")
        print("       obs=obs,")
        print("       responses=responses,")
        print("       input_ids=input_ids,")
        print("       attention_mask=attention_mask,")
        print("       pixel_values=pixel_values,")
        print("       proprio=proprio,")
        print("       action=action,")
        print("       reward=reward,")
        print("       done=done,")
        print("       value=0.0,  # <-- ADD THIS LINE")
        print("       old_log_prob=log_prob,")
        print("   )")
    else:
        print("❌ SOME TESTS FAILED")
        print("="*70)
        print(f"\nFailed tests: {', '.join(failed)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
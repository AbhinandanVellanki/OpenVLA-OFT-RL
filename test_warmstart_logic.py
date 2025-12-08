#!/usr/bin/env python3
"""
Test script to verify L1 warmstart → RL transition logic.
Simulates training progression and validates phase transitions.
"""

import numpy as np

class WarmstartTest:
    def __init__(self, l1_warmup_steps=25000, l1_transition_steps=5000):
        self.l1_warmup_steps = l1_warmup_steps
        self.l1_transition_steps = l1_transition_steps
        self.use_l1_warmstart = True
    
    def _should_use_l1_actions(self, global_step):
        """Determine whether to use L1 or tokenized actions."""
        if not self.use_l1_warmstart:
            return False
        
        if global_step < self.l1_warmup_steps:
            return True  # Warmup
        elif global_step < self.l1_warmup_steps + self.l1_transition_steps:
            # Transition: epsilon-greedy
            progress = (global_step - self.l1_warmup_steps) / self.l1_transition_steps
            epsilon = 1.0 - progress
            return np.random.rand() < epsilon
        else:
            return False  # RL phase
    
    def _get_rollout_policy_name(self, global_step):
        """Get human-readable policy name."""
        if not self.use_l1_warmstart:
            return "Tokenized (RL)"
        
        if global_step < self.l1_warmup_steps:
            return "L1 (warmup)"
        elif global_step < self.l1_warmup_steps + self.l1_transition_steps:
            progress = (global_step - self.l1_warmup_steps) / self.l1_transition_steps
            return f"L1→Tokenized ({progress:.0%})"
        else:
            return "Tokenized (RL)"
    
    def test_phase_transitions(self):
        """Test that phase transitions work correctly."""
        print("=" * 60)
        print("Testing L1 Warmstart → RL Transition Logic")
        print("=" * 60)
        
        # Test checkpoints
        test_steps = [
            0,          # Start of warmup
            12500,      # Mid-warmup
            24999,      # End of warmup
            25000,      # Start of transition
            27500,      # Mid-transition
            29999,      # End of transition
            30000,      # Start of RL
            50000,      # Mid-RL
            100000,     # Late RL
        ]
        
        print("\n📊 Phase Transitions:")
        print("-" * 60)
        for step in test_steps:
            policy_name = self._get_rollout_policy_name(step)
            uses_l1 = self._should_use_l1_actions(step)
            
            # Determine phase
            if step < self.l1_warmup_steps:
                phase = "WARMUP"
            elif step < self.l1_warmup_steps + self.l1_transition_steps:
                phase = "TRANSITION"
            else:
                phase = "RL"
            
            print(f"Step {step:6d} | {phase:11s} | {policy_name:20s} | Uses L1: {uses_l1}")
        
        # Test transition statistics
        print("\n📈 Transition Statistics:")
        print("-" * 60)
        
        # Sample 1000 steps during transition
        transition_start = self.l1_warmup_steps
        transition_samples = []
        for _ in range(1000):
            step = np.random.randint(transition_start, transition_start + self.l1_transition_steps)
            uses_l1 = self._should_use_l1_actions(step)
            transition_samples.append(int(uses_l1))
        
        l1_ratio = np.mean(transition_samples)
        print(f"Average L1 usage during transition: {l1_ratio:.1%}")
        print(f"Expected: ~50% (due to epsilon-greedy)")
        
        # Verify phase boundaries
        print("\n✅ Phase Boundary Verification:")
        print("-" * 60)
        
        # Warmup: should always use L1
        warmup_samples = [self._should_use_l1_actions(step) for step in range(0, self.l1_warmup_steps, 1000)]
        assert all(warmup_samples), "❌ FAIL: Not all warmup steps use L1"
        print("✓ Warmup phase: All steps use L1")
        
        # RL: should never use L1
        rl_start = self.l1_warmup_steps + self.l1_transition_steps
        rl_samples = [self._should_use_l1_actions(step) for step in range(rl_start, rl_start + 10000, 1000)]
        assert not any(rl_samples), "❌ FAIL: Some RL steps still use L1"
        print("✓ RL phase: No steps use L1")
        
        # Transition: should have mix
        transition_ratio = np.mean(transition_samples)
        assert 0.3 < transition_ratio < 0.7, f"❌ FAIL: Transition ratio {transition_ratio:.2%} not balanced"
        print(f"✓ Transition phase: Balanced L1/Tokenized mix ({transition_ratio:.1%})")
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)

if __name__ == "__main__":
    # Test default configuration
    tester = WarmstartTest(l1_warmup_steps=25000, l1_transition_steps=5000)
    tester.test_phase_transitions()
    
    print("\n\n")
    
    # Test extended warmup configuration
    print("Testing Extended Warmup Configuration:")
    tester_extended = WarmstartTest(l1_warmup_steps=50000, l1_transition_steps=10000)
    tester_extended.test_phase_transitions()

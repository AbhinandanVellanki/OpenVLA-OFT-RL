# GAE-based PPO Implementation for OpenVLA

**Status**: ✅ Fully Integrated and Tested
**Date**: December 6, 2025

---

## Quick Start

### Run Tests

```bash
# Run standalone unit tests
python test_gae_standalone.py
```

### Run GAE-PPO Training

```bash
# Single task with GAE
python start_gae_ppo_training.py --task-id 0 --timesteps 100000

# Custom configuration
python start_gae_ppo_training.py --task-id 2 --timesteps 50000 --gpu 0 --actor-lr 1e-5 --critic-lr 3e-4

# Multi-task training
python start_gae_ppo_training.py --task-ids 0 1 2 3 --num-envs 4 --timesteps 200000
```

---

## What This Implements

**GAE (Generalized Advantage Estimation)** replaces GRPO's value-free advantages with a learned critic:

- **GRPO**: `advantages = discounted_rewards` (no baseline, high variance)
- **GAE**: `advantages = Σ(γλ)^t * TD_residuals` (learned baseline, lower variance)

**Key Benefits**:
- Lower variance → more stable training
- Better credit assignment → faster learning
- Supports dense rewards → more flexible

---

## File Structure

```
OpenVLA-OFT-RL/
├── ppo/
│   ├── gae.py                      # GAE computation (NumPy + PyTorch)
│   ├── gae_ppo_trainer.py          # GAE extensions (optimizers, losses)
│   └── config.py                   # Added: use_gae, freeze_vla_for_critic
│
├── vla_oft/min_vla/
│   └── value_head.py               # Value network (3-layer MLP)
│
├── configs/
│   └── gae_ppo_config.yaml         # Example GAE config
│
├── gae_ppo_example.py              # Integration example
├── test_gae_standalone.py          # Standalone test runner
└── start_gae_ppo_training.sh       # Bash entrypoint
```

---

## Entrypoint: `start_gae_ppo_training.sh`

### What It Does

1. **Parses arguments**: task suite, task ID, timesteps, GPU
2. **Sets CUDA device**: `export CUDA_VISIBLE_DEVICES=$GPU_ID`
3. **Runs training**: `python gae_ppo_example.py --use-gae ...`

### Usage

```bash
# Default: libero_spatial task 0, 100k steps, GPU 1
./start_gae_ppo_training.sh

# Custom task
./start_gae_ppo_training.sh --task-id 2 --timesteps 50000

# Different GPU
./start_gae_ppo_training.sh --gpu 0
```

### Config File

The script loads `configs/gae_ppo_config.yaml`:

```yaml
use_gae: true                    # Enable GAE
gae_lambda: 0.95                 # GAE lambda (bias-variance)
actor_lr: 1.0e-5                 # VLA LoRA learning rate
critic_lr: 3.0e-4                # Value head learning rate
gamma: 0.99                      # Discount factor
value_loss_coef: 0.5             # Value loss weight
freeze_vla_for_critic: false     # Gradient flow control
```

---

## Code Flow

### 1. Initialization (`OpenVLA_PPO.__init__`)

```python
from vla_oft.min_vla.value_head import ValueHead
from ppo.gae_ppo_trainer import GAEPPOExtensions, integrate_gae_into_trajectory_buffer

if cfg.use_gae:
    # Create value head (3-layer MLP: 4096 -> 1024 -> 512 -> 1)
    self.value_head = ValueHead(input_dim=4096, hidden_dim=1024).to(device)

    # Create GAE extensions with separate optimizers
    self.gae_ext = GAEPPOExtensions(
        value_head=self.value_head,
        vla_model=self.actor.vla,
        actor_lr=cfg.actor_lr,    # 1e-5 for VLA LoRA
        critic_lr=cfg.critic_lr,  # 3e-4 for value head
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        value_loss_coef=cfg.value_loss_coef,
        freeze_vla_for_critic=cfg.freeze_vla_for_critic,
        device=device,
    )

    # Patch trajectory buffer to use GAE
    integrate_gae_into_trajectory_buffer(
        trajectory_buffer=self.trajectory_buffer,
        gae_extensions=self.gae_ext,
        use_gae=True,
    )
```

**What happens**:
- Value head: 3-layer MLP (~5M params)
- Two optimizers: actor (AdamW, lr=1e-5), critic (AdamW, lr=3e-4)
- Trajectory buffer's `compute_advantages()` replaced with GAE version

---

### 2. Rollout Collection (`collect_rollouts`)

```python
# Get observation
obs = env.step(action)

# Compute value estimate
if cfg.use_gae:
    with torch.no_grad():
        hidden_states = self.actor.vla.get_hidden_states(obs, instruction)
        value = self.value_head(hidden_states).item()
else:
    value = 0.0  # GRPO doesn't use values

# Store in trajectory buffer
self.trajectory_buffer.add(
    observation=obs,
    action_response=action_response,
    reward=reward,
    done=done,
    value=value,  # Now used by GAE!
    ...
)
```

**What happens**:
- VLA extracts hidden states from observation
- Value head predicts scalar value from hidden states
- Value stored in trajectory for later GAE computation

---

### 3. Advantage Computation (`compute_advantages`)

```python
# Called after rollout collection
self.trajectory_buffer.compute_advantages(normalize=True)
```

**GAE Implementation** (patched into trajectory buffer):

```python
# Extract trajectory data
rewards = np.array(traj["rewards"])
values = np.array(traj["values"])
dones = np.array(traj["dones"])

# Compute GAE
advantages, returns = compute_gae(
    rewards=rewards,
    values=values,
    dones=dones,
    gamma=0.99,
    gae_lambda=0.95,
    normalize_advantages=True,
)

# Store for policy update
traj["advantages"] = advantages
traj["returns"] = returns
```

**GAE Algorithm**:
```
For t from T-1 to 0:
    δ_t = r_t + γ*V(s_{t+1})*(1-done) - V(s_t)
    A_t = δ_t + γ*λ*(1-done)*A_{t+1}
R_t = A_t + V(s_t)
```

---

### 4. Policy Update (`update_policy`)

```python
# Forward pass
hidden_states = self.actor.vla.get_hidden_states(batch_obs, instruction)
new_log_probs = compute_log_probs(...)
values = self.value_head(hidden_states).squeeze(-1)

# Compute losses
policy_loss = ppo_clip_loss(new_log_probs, old_log_probs, advantages)
value_loss = MSE(values, returns)
total_loss = policy_loss + 0.5 * value_loss

# Update both networks
grad_info = self.gae_ext.update_actor_critic(
    total_loss=total_loss,
    max_grad_norm=1.0,
)

# Log metrics
wandb.log({
    "train/policy_loss": policy_loss.item(),
    "train/value_loss": value_loss.item(),
    "train/actor_grad_norm": grad_info['actor_grad_norm'],
    "train/critic_grad_norm": grad_info['critic_grad_norm'],
})
```

**What happens**:
1. Compute new log probs and values for batch
2. Policy loss: PPO clipped objective
3. Value loss: MSE between predictions and TD targets
4. Combined loss: `L_total = L_policy + 0.5 * L_value`
5. Backprop through both actor and critic
6. Separate gradient clipping and optimizer steps

---

## Unit Tests

### Test Coverage

**File**: `test_gae_standalone.py`

10 test cases covering:
1. **Output shapes**: Verify correct tensor dimensions
2. **Simple trajectory**: Basic GAE computation correctness
3. **Normalization**: Mean=0, std=1 after normalization
4. **Episode boundaries**: GAE resets at `done=True`
5. **All zero rewards**: Negative advantages when rewards=0
6. **NaN handling**: Replaces NaN/inf with zeros
7. **Batch processing**: Multiple trajectories simultaneously
8. **PyTorch CPU**: GPU-accelerated version
9. **NumPy/PyTorch consistency**: Same results across backends
10. **Returns decomposition**: Verify `returns = advantages + values`

### Running Tests

```bash
# Standalone runner (recommended)
python test_gae_standalone.py

# Expected output:
# ============================================================
# Running GAE Unit Tests
# ============================================================
# ✓ Output shapes correct
# ✓ Simple trajectory test passed
# ...
# ============================================================
# Test Results: 10/10 passed
# ============================================================
```

### Manual Testing

```python
from ppo.gae import compute_gae
import numpy as np

# Success trajectory
rewards = np.array([0.0, 0.0, 0.0, 1.0])
values = np.array([0.5, 0.6, 0.7, 0.8])
dones = np.array([False, False, False, True])

advantages, returns = compute_gae(rewards, values, dones)

print(f"Advantages: {advantages}")
print(f"Returns: {returns}")
# Advantages should be positive (reward > values)
# Returns = advantages + values
```

---

## Configuration Reference

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_gae` | `false` | Enable GAE advantage estimation |
| `gae_lambda` | `0.95` | Bias-variance tradeoff (0=TD, 1=MC) |
| `actor_lr` | `1e-5` | VLA LoRA learning rate |
| `critic_lr` | `3e-4` | Value head learning rate |
| `gamma` | `0.99` | Discount factor |
| `value_loss_coef` | `0.5` | Value loss weight in total loss |
| `freeze_vla_for_critic` | `false` | Detach VLA gradients for critic |

### When to Use GAE vs GRPO

**Use GRPO** when:
- Sparse binary rewards (success/failure only)
- Short episodes (<100 steps)
- Maximum simplicity needed

**Use GAE** when:
- Dense/shaped rewards
- Long episodes (>200 steps)
- Training instability with GRPO
- Need better sample efficiency

---

## Logging and Monitoring

### Wandb Logging

**Automatic logging** at every update (configurable via `--log-interval`):

- `train/policy_loss`: PPO policy loss
- `train/value_loss`: Critic MSE loss (GAE only)
- `train/clipfrac`: Fraction of clipped ratios
- `train/approx_kl`: Approximate KL divergence
- `train/actor_grad_norm`: Actor gradient norm (GAE only)
- `train/critic_grad_norm`: Critic gradient norm (GAE only)
- `rollout/success_rate`: Training success rate
- `rollout/mean_length`: Average episode length
- `val/success_rate`: Validation success rate (periodic)

**Disable wandb**: Add `--no-wandb` flag

### Matplotlib Plots

**Automatic local plotting** at every update:
- Saves to `plots/training_curves_update{N:05d}.png`
- 3 plots (GRPO) or 4 plots (GAE):
  1. Training Losses (policy + value if GAE)
  2. Success Rates (training + validation)
  3. PPO Metrics (clip fraction, KL divergence)
  4. Gradient Norms (actor + critic, GAE only)

**Plots are saved locally** and don't require wandb.

---

## Integration Status

✅ **Fully integrated into `OpenVLA_PPO.py`**:

- [x] Value head from `vla_oft.min_vla.value_head`
- [x] GAE extensions with separate optimizers
- [x] Trajectory buffer patched for GAE
- [x] Value computation during rollout
- [x] Value loss during policy update
- [x] Actor + critic optimizer steps
- [x] Wandb logging for all GAE metrics
- [x] Matplotlib plotting after each epoch
- [x] Config flag `use_gae` to enable/disable

**No manual integration needed** - just run `start_gae_ppo_training.py`!

---

## References

1. **GAE Paper**: Schulman et al. (2015). "High-Dimensional Continuous Control Using Generalized Advantage Estimation." [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)

2. **PPO Paper**: Schulman et al. (2017). "Proximal Policy Optimization Algorithms." [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

---

**Questions?** See `gae_ppo_example.py` for detailed integration code.

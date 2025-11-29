# OpenVLA PPO Implementation Guide

**Complete reference for trajectory-based PPO training with action tokenization**

**Last Updated**: November 29, 2025  
**Status**: Ready for Testing

---

## Table of Contents

1. [Implementation Summary](#implementation-summary)
2. [Architecture Overview](#architecture-overview)
3. [Configuration Reference](#configuration-reference)
   - [PPOConfig](#ppoconfig)
   - [OpenVLAActorConfig](#openvlaactorconfig)
4. [Implementation Details](#implementation-details)
5. [Testing & Validation](#testing--validation)
6. [Next Steps](#next-steps)
7. [Troubleshooting](#troubleshooting)

---

<!-- ## Implementation Summary

Successfully implemented trajectory-based PPO with action tokenization for OpenVLA fine-tuning, following the proven SimpleVLA-RL architecture with proper PPO policy gradients, GRPO advantages, and sparse rewards.

### What Was Implemented ✅

#### 1. Action Tokenization Infrastructure
- **File**: `vla-oft/min_vla/action_tokenizer.py` (140 lines)
- 256-bin discretization mapping continuous actions to vocabulary tokens
- Maps to last 256 vocab tokens (31744-32000)
- Round-trip: continuous → tokens → continuous with <1% error

#### 2. Value Head Network (Removed for GRPO)
- **File**: `vla-oft/min_vla/value_head.py` (42 lines)
- Lightweight neural network for critic
- 3-layer MLP: 4096 → 1024 → 512 → 1
- **Status**: Created but not used (GRPO is value-free)

#### 2b. L1 Regression Action Head (Optional - used only for more supervised finetuning or comparsions with OpenVLA-OFT)
- **File**: Loaded from checkpoint `action_head--150000_checkpoint.pt`
- MLP for direct continuous action prediction from hidden states
- **Usage**: Only needed for supervised learning or comparison
- **PPO Training**: Not loaded by default (saves 668MB)
- **Control**: `load_l1_action_head=False` in config

#### 3. Trajectory Buffer
- **File**: `ppo/trajectory_buffer.py` (270 lines)
- Stores complete episodes with variable lengths
- `finish_step` markers for episode completion
- GRPO advantage computation (verifier_gamma=1.0)
- Automatic trajectory masking

#### 4. PPO Core Algorithms
- **File**: `ppo/core_algos.py` (115 lines)
- `logprobs_from_logits()` - extract log probs from action token logits
- `compute_policy_loss()` - PPO clipped surrogate with asymmetric clipping (0.28/0.2)
- `apply_mask_with_grad_control()` - gradient-safe trajectory masking

#### 5. PPO Configuration
- **File**: `ppo/config.py` (331 lines)
- Complete PPOConfig dataclass with all training hyperparameters
- Extensive documentation for each parameter
- Validation logic for multi-task and device settings

#### 6. Main Training Pipeline
- **File**: `OpenVLA_PPO.py` (extensively modified, now 1285 lines)
- Imports PPOConfig and ValueHead from separate modules
- Contains only OpenVLAPPO trainer class
- Trajectory-based rollout collection with sparse rewards
- PPO policy gradient updates (replaces reward-weighted BC)

### Key Architectural Changes

| Component | Before | After |
|-----------|--------|-------|
| **Actions** | L1 Regression (MSE loss) | Tokenized (256 bins, cross-entropy) |
| **Action Prediction** | Direct continuous via MLP | Token logits → sample → detokenize |
| **Loss** | Reward-weighted BC | PPO clipped surrogate |
| **Advantages** | GAE | GRPO (verifier_gamma=1.0, value-free) |
| **Rewards** | Dense (every step) | Sparse (finish_step only) |
| **Buffer** | Step-based | Trajectory-based |
| **Sampling** | Deterministic | Stochastic (temp=1.6) during training |
| **Clipping** | Symmetric (0.2) | Asymmetric (high=0.28, low=0.2) |
| **Batch Size** | 32 | 1 (memory optimization) |
| **n_steps** | 500 | 100 (reduced for 24GB GPU) |

--- -->

## Architecture Overview

### Non-Autoregressive (Full Action) Prediction Architecture

**Two Prediction Pathways** (Hybrid Support):

#### **Pathway 1: Tokenized Actions (Used for PPO)**
```
VLA Forward Pass:
  Observation (image + proprio)
    ↓
  Vision Encoder (SigLIP)
    ↓
  Language Model (LLaMA 7B)
    ↓
  Logits (vocab_size=32000)
    ↓
  Extract action token logits: logits[..., -256-64:-64]  # Last 256 tokens
    ↓
  Sample/Argmax → Token IDs [31744, 32000)
    ↓
  Detokenize → Continuous Actions [-1, 1]^7
```

**Key Points**:
- Actions are treated as **tokens in the vocabulary**
- No separate action head MLP needed
- Log probabilities directly from language model logits
- **This is what PPO uses for training**

#### **Pathway 2: L1 Regression (Used in OpenVLA-OFT for supervised finetuning, NOT USED FOR PPO)**
```
VLA Forward Pass:
  Observation (image + proprio)
    ↓
  Vision Encoder (SigLIP)
    ↓
  Language Model (LLaMA 7B)
    ↓
  Hidden States (4096-dim)
    ↓
  L1 Regression Head (3-layer MLP)
    ↓
  Continuous Actions [-1, 1]^7
```

**Key Points**:
- Direct continuous action prediction
- Used in original supervised pre-training
- **Not used for PPO training**
- Can be loaded for comparison or supervised learning

### Training Loop Flow

```
1. Rollout Collection (with torch.no_grad())
   ├─► Environment step with stochastic sampling (temp=1.6)
   ├─► Store: responses, input_ids, attention_mask, pixel_values, proprio
   ├─► Assign sparse rewards (0 everywhere, success/failure at finish_step)
   └─► Compute GRPO (value-less) advantages

2. Policy Update (with gradient accumulation per sample)
   ├─► For each epoch:
   │   ├─► Shuffle trajectory indices
   │   └─► For each sample (batch_size=1):
   │       ├─► Forward pass VLA to get new log probs
   │       ├─► Forward pass Value Head to get state values
   │       ├─► Compute PPO clipped loss (asymmetric clipping)
   │       ├─► Compute value loss (MSE)
   │       ├─► Backward pass (immediate gradient accumulation)
   │       └─► Clear CUDA cache
   ├─► Clip gradients (max_norm=0.5)
   └─► Optimizer step

3. Validation (with greedy sampling, temp=0.0)
   ├─► Run val_episodes complete episodes
   └─► Report success rate
```

### Memory Distribution (Single GPU, 24GB)

```
Training Phase (PPO Mode - L1 head not loaded):
├─► VLA Model (7B params, bf16): ~15GB
├─► LoRA Adapters: ~400MB
├─► Value Head (not used, GRPO): ~5MB
├─► Trajectory Buffer (100 steps): ~1-2GB
├─► Gradients + Optimizer: ~2GB
├─► Activations (batch_size=1): ~500MB
└─► Total: ~18-19GB (fits comfortably in 24GB)

With gradient checkpointing and aggressive cache clearing

Note: L1 regression head NOT loaded (saves 668MB)
      Set load_l1_action_head=True to load it (+668MB)
```

---

## Configuration Reference

### PPOConfig

**Location**: `ppo/config.py`

Complete configuration for Proximal Policy Optimization training.

#### Training Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `total_timesteps` | int | 100000 | Total environment steps to collect. Example: 100k = ~1000 updates with n_steps=100 |
| `n_steps` | int | 100 | Steps per policy update (rollout length). Reduced from 500 for memory efficiency |
| `batch_size` | int | 1 | Minibatch size for SGD. Set to 1 for per-sample gradient accumulation on 24GB GPU |
| `n_epochs` | int | 10 | Passes through collected data per update. Standard PPO uses 10 epochs |

**Memory Optimization**: With `batch_size=1` and `n_steps=100`, we process one sample at a time with immediate backward() to prevent computation graph buildup. This is critical for 7B models on 24GB GPUs.

#### PPO Algorithm Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `actor_lr` | float | 1e-5 | Actor learning rate. Low for fine-tuning 7B model (prevents catastrophic forgetting) |
| `critic_lr` | float | 3e-4 | Critic learning rate. Higher since value head trains from scratch |
| `clip_ratio_high` | float | 0.28 | PPO upper clip ratio (asymmetric, from SimpleVLA-RL) |
| `clip_ratio_low` | float | 0.2 | PPO lower clip ratio (more conservative on negative side) |
| `gamma` | float | 0.99 | Discount factor. 0.99 = values rewards ~100 steps ahead |
| `gae_lambda` | float | 0.95 | GAE lambda (unused, GRPO is used instead) |
| `verifier_gamma` | float | 1.0 | Discount for GRPO advantage estimation (no discounting for sparse rewards) |
| `entropy_coef` | float | 0.01 | Entropy bonus (not used for deterministic VLA) |
| `value_loss_coef` | float | 0.5 | Value loss coefficient in total loss |
| `max_grad_norm` | float | 0.5 | Max gradient norm for clipping |

**PPO Loss Formula**:
```python
L^CLIP = min(r(θ)A, clip(r(θ), 1-ε_low, 1+ε_high)A)
where r(θ) = π_θ(a|s) / π_θ_old(a|s)
```

**Asymmetric Clipping**: Allows more aggressive positive updates (0.28) while being conservative on negative updates (0.2).

#### Sampling and Exploration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rollout_temperature` | float | 1.6 | Temperature for stochastic sampling during training (from SimpleVLA-RL) |
| `eval_temperature` | float | 0.0 | Temperature for evaluation (greedy, deterministic) |
| `kl_coef` | float | 0.0 | KL divergence penalty coefficient (disabled by default) |

**Temperature Effects**:
- `temp=1.6`: Encourages exploration during training rollouts
- `temp=0.0`: Greedy argmax selection for deterministic evaluation

#### Trajectory Processing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `traj_split_num` | int | 4 | Number of chunks to split trajectory (for gradient accumulation) |
| `traj_mini_batch_size` | int | 8 | Mini-batch size for trajectory processing |
| `separate_rollout_training` | bool | False | Use separate GPU for rollout (advanced, not implemented yet) |

#### Validation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `val_interval` | int | 1000 | Validate every N environment steps |
| `val_episodes` | int | 10 | Episodes per validation phase |

#### Logging and Checkpointing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_wandb` | bool | False | Enable Weights & Biases logging |
| `wandb_entity` | str | None | W&B entity (username or team) |
| `log_interval` | int | 1000 | Print stats every N steps |
| `save_interval` | int | 10000 | Save checkpoint every N steps |

#### Environment Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_suite` | str | "libero_spatial" | LIBERO suite: spatial/object/goal/10 |
| `task_ids` | List[int] | None | Task IDs for multi-task (e.g., [0,1,2,3]) |
| `task_id` | int | 0 | Single task ID (used if task_ids is None) |
| `num_envs` | int | 1 | Parallel environments (must match task_ids length) |
| `obs_mode` | str | "image_state" | Observation mode (must be "image_state" for VLA) |
| `image_size` | Tuple[int,int] | (224, 224) | Input image size (OpenVLA requires 224x224) |

**Multi-task Example**:
```python
# Single task
PPOConfig(task_id=0, num_envs=1)

# Multi-task (4 tasks in parallel)
PPOConfig(task_ids=[0, 1, 2, 3], num_envs=4)
```

#### Device Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | str | "cuda:1" | Primary device for model and rollouts |
| `training_device` | str | "cuda:1" | Device for gradient computation (should match device) |

---

### OpenVLAActorConfig

**Location**: `vla-oft/min_vla/config.py`

Configuration for OpenVLA-OFT 7B model loading and inference.

#### Model Path and Loading

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pretrained_checkpoint` | str | "openvla-7b-oft-finetuned-libero-spatial" | Local path or HF Hub ID |
| `use_local` | bool | True | Prioritize local loading over HF Hub |

**Paths**:
- Local: `"openvla-7b-oft-finetuned-libero-spatial"` (relative to vla-oft/)
- HF Hub: `"moojink/openvla-7b-oft-finetuned-libero-spatial"`

#### GPU Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_multi_gpu` | bool | False | Distribute components across GPUs |
| `gpu_id` | int | 0 | Primary GPU for VLA model |
| `secondary_gpu_id` | int | 1 | Secondary GPU for action head/value head |

**Memory Requirements**:
- Single-GPU: 14.2GB total on gpu_id
- Multi-GPU: 14GB on gpu_id + 200MB on secondary_gpu_id

#### Training Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `freeze_vla_backbone` | bool | False | Freeze VLA backbone during training |

**With LoRA enabled** (recommended):
- Keep `freeze_vla_backbone=False` to enable full model adaptation
- Only ~1-2% of parameters trainable via LoRA adapters
- Memory: ~18-20GB with gradients

#### LoRA Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_lora` | bool | True | Enable LoRA adapters for efficient fine-tuning |
| `lora_rank` | int | 32 | LoRA rank (controls adapter size) |
| `lora_alpha` | int | 16 | LoRA scaling factor (capped at min(rank, 16)) |
| `lora_dropout` | float | 0.0 | LoRA dropout for regularization |
| `lora_target_modules` | str | "all-linear" | Which modules to apply LoRA to |

**LoRA Memory**:
- rank=32: ~200M params, ~4GB memory, best quality
- rank=16: ~100M params, ~2GB memory, good balance
- rank=8: ~50M params, ~1GB memory, faster

#### Model Quantization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `load_in_4bit` | bool | False | Enable 4-bit quantization |

**Comparison**:
- 4-bit: 2GB VRAM, 116ms/action (8.6 Hz), slight quality loss
- bfloat16: 14GB VRAM, 53ms/action (18.8 Hz), full quality

#### Proprioception Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_proprio` | bool | True | Enable robot state input (required for LIBERO) |
| `proprio_dim` | int | 8 | Expected proprio dimension (3 pos + 4 quat + 1 gripper) |

**Proprio Format** (8D):
- Dimensions 0-2: End-effector position (x, y, z)
- Dimensions 3-6: Orientation quaternion (w, x, y, z)
- Dimension 7: Gripper state (normalized)

#### Vision and Action Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_images_in_input` | int | 1 | Number of RGB cameras (1 = agentview) |
| `load_l1_action_head` | bool | False | Load L1 regression action head from checkpoint |
| `freeze_l1_action_head` | bool | True | Freeze L1 head if loaded (read-only) |
| `use_tokenized_actions` | bool | True | Use token logits for action prediction (required for PPO) |
| `use_l1_regression` | bool | True | Deprecated: kept for backward compatibility |
| `finetuned_on_discrete_actions` | bool | False | Whether checkpoint uses discrete actions |
| `deterministic_eval` | bool | True | Use deterministic policy during eval |

**Action Prediction Modes**:

1. **Tokenized Actions (PPO Mode - Default)**:
   ```python
   load_l1_action_head = False
   use_tokenized_actions = True
   ```
   - VLA language model logits → action token probabilities
   - Sample/argmax from last 256 vocab tokens
   - Detokenize to continuous actions
   - **Memory**: Saves 668MB by not loading L1 head
   - **Use for**: PPO training and inference

2. **L1 Regression (Legacy Mode)**:
   ```python
   load_l1_action_head = True
   freeze_l1_action_head = True  # Or False for training
   use_tokenized_actions = False
   ```
   - VLA hidden states → L1 regression MLP → continuous actions
   - **Memory**: +668MB for L1 head (~167M params)
   - **Use for**: Supervised learning, comparison with original checkpoint

3. **Hybrid Mode (Comparison)**:
   ```python
   load_l1_action_head = True   # Load for comparison
   freeze_l1_action_head = True  # Frozen (read-only)
   use_tokenized_actions = True  # Still use tokenized for PPO
   ```
   - L1 head loaded but not used for training
   - Allows switching between modes for ablation studies
   - **Warning**: PPO will show warning and exclude L1 head from training

#### Performance Optimizations

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_flash_attention` | bool | True | Enable Flash Attention 2 (2-4x faster) |

**Performance**:
- With Flash Attention: 53ms/action (18.8 Hz)
- Without: 80-100ms/action (10-12 Hz)

---

<!-- ## Implementation Details

### Phase 1: Foundation ✅

#### Created Files

1. **`vla-oft/min_vla/action_tokenizer.py`**
   ```python
   class ActionTokenizer:
       def __init__(self, vocab_size=32000, n_bins=256):
           # Maps continuous actions to last 256 vocab tokens
           self.action_token_begin = vocab_size - n_bins  # 31744
           
       def discretize_actions(self, actions):
           # Continuous [-1, 1] → bin indices [0, 255]
           # Then shift to vocab range [31744, 32000)
           
       def detokenize_actions(self, token_ids):
           # Vocab tokens → bin indices → continuous actions
   ```

2. **`vla-oft/min_vla/value_head.py`**
   ```python
   class ValueHead(nn.Module):
       def __init__(self, input_dim=4096, hidden_dim=1024):
           # 3-layer MLP for state value estimation
           # 4096 → 1024 → 512 → 1
   ```

3. **`ppo/trajectory_buffer.py`**
   ```python
   class TrajectoryBuffer:
       def add_trajectory(self, trajectory_data):
           # Store complete episodes with finish_step markers
           
       def compute_advantages(self, verifier_gamma=1.0):
           # GRPO advantages for sparse rewards
           # Only propagate from finish_step
           
       def generate_traj_mask(self, finish_step):
           # Create boolean masks for valid trajectory steps
   ```

4. **`ppo/core_algos.py`**
   ```python
   def logprobs_from_logits(logits, labels):
       # Extract log probs for specific action tokens
       
   def compute_policy_loss(log_ratio, advantages, clip_high, clip_low):
       # PPO clipped surrogate with asymmetric clipping
       # L = min(r*A, clip(r, 1-ε_low, 1+ε_high)*A)
       
   def apply_mask_with_grad_control(tensor, mask):
       # Gradient-safe masking for trajectories
   ```

5. **`ppo/config.py`**
   - Complete PPOConfig dataclass (331 lines)
   - Extracted from OpenVLA_PPO.py for better organization

### Phase 2: Core PPO ✅

#### Modified: `OpenVLA_PPO.py`

**Config Updates**:
- ✅ `batch_size`: 32 → 1 (per-sample gradient accumulation)
- ✅ `n_steps`: 500 → 100 (memory optimization)
- ✅ `clip_epsilon` → `clip_ratio_high=0.28, clip_ratio_low=0.2`
- ✅ Added `verifier_gamma=1.0` for GRPO
- ✅ Added `rollout_temperature=1.6, eval_temperature=0.0`

**Action Prediction Changes**:
- ✅ Removed dependency on L1 regression action head
- ✅ Uses tokenized actions via language model logits
- ✅ L1 head optionally loaded but not used for PPO
- ✅ Config verification: requires `use_tokenized_actions=True`
- ✅ Clear warnings if L1 head loaded unnecessarily

**New Methods**:
```python
def predict_action_tokens_with_grad(self, obs, task_prompt, temperature=1.6):
    # Forward pass through VLA to get action token logits
    # Extracts logits[..., -256-64:-64] for action vocabulary
    # Returns: responses, log_probs, continuous_action, etc.
    # Used during policy updates (requires gradients)
```

**Rewritten Methods**:

1. **`get_action()`**: Tokenized action prediction
   ```python
   # Verifies use_tokenized_actions=True
   # Calls predict_action_tokens_with_grad()
   # Returns continuous action + metadata
   ```

1. **`collect_rollouts()`**: Trajectory-based with sparse rewards
   ```python
   # Key changes:
   - Use torch.no_grad() for rollout collection
   - Stochastic sampling with temperature=1.6
   - Store tokenized actions (responses)
   - Assign sparse rewards (0 everywhere, success/failure at finish_step)
   - Compute GRPO advantages in buffer
   ```

2. **`update_policy()`**: PPO policy gradient (replaces reward-weighted BC)
   ```python
   # Per-sample gradient accumulation approach:
   for epoch in range(n_epochs):
       for idx in shuffled_indices:
           # Process ONE sample at a time
           forward_pass()  # Get new log probs
           compute_ppo_loss()  # Single sample loss
           backward()  # Immediate gradient accumulation
           clear_cache()  # Prevent memory buildup
       
       optimizer.step()  # Update after all samples
       optimizer.zero_grad()
   ```

3. **`validate()`**: Greedy sampling for deterministic evaluation
   ```python
   # Use eval_temperature=0.0 for argmax selection
   ```

### Memory Optimization Strategy

**Problem**: 7B model on 24GB GPU with trajectory-based training

**Solutions Implemented**:

1. **Per-Sample Gradient Accumulation** (Critical)
   ```python
   # Instead of:
   for idx in batch:
       forward() → append to list  # Builds computation graph
   stack_list() → compute_loss() → backward()  # OOM!
   
   # We do:
   for idx in batch:
       forward() → compute_loss() → backward()  # Immediate
       clear_cache()  # Prevent buildup
   # Gradients accumulate in model parameters automatically
   ```

2. **Reduced Batch Parameters**
   - `batch_size=1`: Process one sample at a time
   - `n_steps=100`: Smaller rollout buffer (was 500)
   - Total memory per update: ~19-20GB

3. **Gradient Checkpointing**
   ```python
   if hasattr(self.actor.vla.language_model, 'gradient_checkpointing_enable'):
       self.actor.vla.language_model.gradient_checkpointing_enable()
   ```

4. **Detached Tensors in Buffer**
   ```python
   # In trajectory_buffer.py
   tensors = torch.stack(batch_data).detach()  # Prevent gradient retention
   ```

5. **Aggressive Cache Clearing**
   ```python
   del tensor_name
   torch.cuda.empty_cache()  # After each forward pass
   ```

--- -->

<!-- ## Testing & Validation

### Unit Tests Created ✅

**File**: `ppo/tests/test_trajectory_ppo.py` (280 lines)

```bash
cd /home/abhi/Documents/Deep-RL/OpenVLA-OFT-RL
python ppo/tests/test_trajectory_ppo.py
```

**Tests**:
1. ✅ `test_action_tokenizer_round_trip` - Discretize and reconstruct actions
2. ✅ `test_trajectory_buffer_storage` - Store and retrieve trajectories
3. ✅ `test_grpo_advantages` - GRPO advantage computation
4. ✅ `test_ppo_loss_functions` - PPO clipped loss
5. ✅ `test_gradient_flow` - Gradients through masked operations

### Integration Testing Checklist

- [x] Unit tests pass
- [x] Rollout collection succeeds (100 steps tested)
- [ ] Policy update completes without OOM (per-sample gradient accumulation implemented)
- [ ] Training runs for 1000 steps
- [ ] Success rate improves over baseline
- [ ] Checkpoints save/load correctly

### Expected Metrics

#### Rollout Phase
- `rollout/success_rate`: 0-1 (sparse reward signal)
- `rollout/mean_length`: ~100-300 steps per episode
- `rollout/num_trajectories`: Variable based on episode lengths
- `rollout/collection_time`: ~18-20s for 100 steps (~5-6 it/s)

#### Training Phase
- `train/policy_loss`: Should decrease over epochs
- `train/value_loss`: Should decrease and stabilize
- `train/clipfrac`: 0.1-0.3 (indicates policy is changing appropriately)
- `train/approx_kl`: <0.01 for stable training, <0.05 acceptable

#### Validation Phase
- `val/success_rate`: Should increase over training
- `val/mean_reward`: Same as success_rate (sparse rewards)

--- -->

## Next Steps

### Immediate (Before First Training Run)

1. **Kill any existing training processes**
   ```bash
   pkill -f "OpenVLA_PPO.py"
   ps aux | grep OpenVLA_PPO  # Verify killed
   ```

2. **Restart training with gradient accumulation fix**
   ```bash
   cd /home/abhi/Documents/Deep-RL/OpenVLA-OFT-RL
   ./start_ppo_training.sh
   ```

3. **Monitor training progress**
   ```bash
   tail -f ppo_training.log
   # Watch for:
   # - Rollout collection: 100/100 steps ✅
   # - Policy update: Should complete all 10 epochs without OOM
   # - Training: 1%, 2%, etc.
   ```

### Short-Term Enhancements

4. **Implement Reference Policy KL Penalty** (if kl_coef > 0)
   ```python
   # In update_policy()
   if self.ref_vla is not None and self.cfg.kl_coef > 0:
       with torch.no_grad():
           ref_action_data = self.ref_vla.forward(...)
           ref_log_prob = compute_log_prob(ref_action_data)
       kl_div = (new_log_prob - ref_log_prob).mean()
       kl_loss = self.cfg.kl_coef * kl_div
   ```

5. **Add Auxiliary Action Reconstruction Loss**
   ```python
   # Helps bridge discretization gap
   continuous_pred = self.action_tokenizer.detokenize_actions(responses)
   continuous_target = data['actions'][idx]
   recon_loss = 0.1 * F.mse_loss(continuous_pred, continuous_target)
   total_loss = policy_loss + recon_loss
   ```

### Medium-Term Testing

6. **Full Training Run** (10,000 steps)
   ```bash
   python OpenVLA_PPO.py \
     --task-suite libero_spatial \
     --task-id 0 \
     --timesteps 10000 \
     --use-wandb
   ```
   
   Expected:
   - Training time: ~3-4 hours for 10k steps
   - Success rate: 0% → 20-40%
   - Policy loss: Decreases over updates
   - Memory: Stable at ~19-20GB

7. **Multi-Task Training**
   ```bash
   python OpenVLA_PPO.py \
     --task-suite libero_spatial \
     --task-ids 0 1 2 3 \
     --num-envs 4 \
     --timesteps 50000
   ```

### Long-Term Features

8. **Multi-GPU Separation** (if separate_rollout_training=True)
   - Implement separate rollout worker on GPU 0
   - Training worker on GPU 1
   - Ray-based communication pipeline

9. **Hyperparameter Tuning**
   - Clip ratios (currently 0.28/0.2)
   - Temperature (currently 1.6)
   - Batch size vs memory tradeoff
   - Number of PPO epochs

---

## Troubleshooting

### Common Issues

#### OOM Errors During Policy Update

**Symptoms**: Training crashes during "Policy update" phase

**Causes**:
1. Computation graph buildup from batched forward passes
2. Not clearing CUDA cache frequently enough
3. Gradient checkpointing not enabled

**Solutions** (Already Implemented):
- ✅ Per-sample gradient accumulation (immediate backward())
- ✅ Aggressive cache clearing after each forward pass
- ✅ Detached tensors in trajectory buffer
- ✅ Gradient checkpointing enabled
- ✅ batch_size=1, n_steps=100

**If Still OOM**:
- Reduce `n_epochs` from 10 to 5
- Reduce `n_steps` from 100 to 50
- Consider freezing more of VLA backbone

#### Policy Not Learning

**Symptoms**: Success rate stays at 0%, policy loss not decreasing

**Possible Causes**:
1. Sparse rewards not assigned correctly
2. Advantages not computed properly
3. Learning rate too low
4. Clipping too aggressive

**Debug Steps**:
```python
# In collect_rollouts(), check:
print(f"Rewards: {rewards}")  # Should be 0s except at finish_step
print(f"Finish steps: {finish_steps}")  # Should mark episode ends

# In update_policy(), check:
print(f"Advantages: {advantages}")  # Should not be all zeros
print(f"Clipfrac: {clipfrac}")  # Should be >0.1
print(f"Policy loss: {policy_loss}")  # Should decrease
```

**Solutions**:
- Increase `actor_lr` from 1e-5 to 3e-5
- Increase `n_epochs` from 10 to 15
- Check that `verifier_gamma=1.0` for GRPO

#### NaN Losses

**Symptoms**: Training shows NaN in loss values

**Causes**:
1. Divide by zero in advantage normalization
2. Exploding gradients
3. Learning rate too high

**Solutions**:
```python
# Add epsilon to advantage normalization
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# Reduce learning rate
actor_lr = 5e-6  # From 1e-5

# More aggressive gradient clipping
max_grad_norm = 0.3  # From 0.5
```

#### Import Errors

**Symptoms**:
```
Import "prismatic.vla.constants" could not be resolved
```

**Solution**: These are expected in IDE. They resolve when running in proper conda environment:
```bash
conda activate oft_rl
python OpenVLA_PPO.py  # Will work
```

### Performance Issues

#### Slow Rollout Collection

**Expected**: ~5-6 it/s for single environment  
**If Slower**: 
- Check Flash Attention is enabled
- Verify running on GPU not CPU
- Use `torch.no_grad()` during rollouts

#### Slow Policy Updates

**Expected**: ~5-10s per epoch (10 epochs = 50-100s)  
**If Slower**:
- Reduce `n_epochs` from 10 to 5
- Increase `batch_size` if memory allows (1 → 2)
- Profile with `torch.profiler` to find bottlenecks

---

## Command Line Usage

### Basic Training

```bash
# Single task, 10k steps, no wandb (quick test)
python OpenVLA_PPO.py \
  --task-suite libero_spatial \
  --task-id 0 \
  --timesteps 10000 \
  --no-wandb

# Full training with wandb
python OpenVLA_PPO.py \
  --task-suite libero_spatial \
  --task-id 0 \
  --timesteps 100000 \
  --use-wandb
```

### Multi-Task Training

```bash
# 4 tasks in parallel
python OpenVLA_PPO.py \
  --task-suite libero_spatial \
  --task-ids 0 1 2 3 \
  --num-envs 4 \
  --timesteps 200000
```

### Detached Training (SSH-Safe)

```bash
# Use provided script
./start_ppo_training.sh

# Monitor progress
tail -f ppo_training.log

# Check if running
cat ppo_train.pid
ps -p $(cat ppo_train.pid)
```

---

## File Structure

```
OpenVLA-OFT-RL/
├── OpenVLA_PPO.py                    # Main training script (1200+ lines)
├── start_ppo_training.sh             # Detached training script
├── ppo_training.log                  # Training logs
├── ppo_train.pid                     # Process ID file
│
├── ppo/
│   ├── config.py                     # PPOConfig (331 lines) ✅
│   ├── trajectory_buffer.py          # TrajectoryBuffer (270 lines) ✅
│   ├── core_algos.py                 # PPO algorithms (115 lines) ✅
│   ├── rollout_buffer.py             # (Deprecated, kept for reference)
│   ├── ppo_trainer.py                # (Deprecated, kept for reference)
│   ├── PPO_IMPLEMENTATION_GUIDE.md   # This file ✅
│   └── tests/
│       └── test_trajectory_ppo.py    # Unit tests (280 lines)
│
├── vla-oft/min_vla/
│   ├── config.py                     # OpenVLAActorConfig (with L1 head options) ✅
│   ├── actor.py                      # OpenVLAActor (l1_action_head) ✅
│   ├── action_tokenizer.py           # ActionTokenizer (140 lines) ✅
│   └── value_head.py                 # ValueHead (42 lines, not used for GRPO) ✅
│
├── libero_rl/utils/
│   ├── obs_utils.py                  # Observation processing
│   └── task_utils.py                 # LIBERO task loading
│
└── checkpoints/
    └── action_head--150000_checkpoint.pt  # L1 regression head (optional)
```

---

## Performance Expectations

### Memory Usage (Single GPU, 24GB)
```
VLA model (7B params, bf16):     ~15GB
LoRA adapters:                   ~400MB
Value head (not used):             ~5MB
Trajectory buffer (100 steps):   ~1-2GB
Gradients + optimizer:           ~2GB
Activations (batch_size=1):      ~500MB
──────────────────────────────────────
Total:                           ~18-19GB ✅ Comfortable fit in 24GB

Note: L1 regression head NOT loaded by default (saves 668MB)
      Set load_l1_action_head=True if needed (+668MB → ~19-20GB total)
```

### Training Speed
```
Rollout collection:   ~5-6 it/s per env
Policy update:        ~5-10s per epoch
Full update cycle:    ~2-3 min per 100 steps
──────────────────────────────────────
1,000 steps:         ~20-30 minutes
10,000 steps:        ~3-4 hours
```

### Expected Results (libero_spatial task 0)
```
Initial success rate:     0-10%
After 1,000 steps:       20-30%
After 10,000 steps:      40-60%
Baseline (pretrained):   30-50%
Target (full training):  >60%
```

---

<!-- ## Implementation Status

### ✅ Phase 1: Foundation (COMPLETE)
- Action tokenization infrastructure
- Value head network
- Trajectory buffer with GRPO
- PPO core algorithms
- Configuration modules

### ✅ Phase 2: Core PPO (COMPLETE)
- Tokenized action prediction with gradients
- PPO policy gradient loss
- Trajectory-based rollout collection
- Sparse reward assignment
- Per-sample gradient accumulation

### ⏳ Phase 3: Optimization (READY TO START)
- Validate per-sample gradient accumulation fix
- Implement reference policy KL penalty (optional)
- Add auxiliary action reconstruction loss
- Profile and optimize bottlenecks

### 🎯 Phase 4: Advanced Features (TODO)
- Multi-GPU separation (separate_rollout_training=True)
- Hyperparameter tuning
- Multi-task performance analysis
- Long-horizon training (>100k steps)

--- -->

## Key Insights

### Why Per-Sample Gradient Accumulation?

**Problem**: With `batch_size=32`, even on powerful GPUs:
```python
# This builds a huge computation graph:
for idx in batch:
    forward_pass()
    log_probs.append(result)  # Each result retains graph
    
stacked = torch.stack(log_probs)  # Combines all graphs
loss = compute_loss(stacked)
loss.backward()  # OOM! Graph too large
```

**Solution**: Process one sample at a time:
```python
# This keeps graphs small:
for idx in batch:
    forward_pass()
    loss = compute_loss(single_result)  # Small graph
    loss.backward()  # Immediate, prevents buildup
    clear_cache()
    
# Gradients accumulate in model.parameters() automatically!
optimizer.step()  # Update after all samples
```

### Why Asymmetric Clipping?

**Standard PPO** uses symmetric clipping (ε = 0.2):
```python
clip(ratio, 1-0.2, 1+0.2) = clip(ratio, 0.8, 1.2)
```

**SimpleVLA-RL** uses asymmetric clipping:
```python
clip(ratio, 1-0.2, 1+0.28) = clip(ratio, 0.8, 1.28)
```

**Rationale**: Allow more aggressive updates when advantage is positive (good actions), but be conservative when advantage is negative (bad actions). This helps learning in sparse reward settings.

### Why GRPO Instead of GAE?

**GAE** (Generalized Advantage Estimation):
- Requires dense rewards or value predictions
- Bootstrap from V(s_{t+1})
- Complex λ-return computation

**GRPO** (Goal-Conditioned Policy Optimization):
- Works with sparse rewards
- Simple: advantage = reward - baseline
- No bootstrapping needed
- Perfect for episodic tasks with success/failure

### Action Prediction Modes: Tokenized vs L1 Regression

**Why Tokenized Actions for PPO?**

The OpenVLA checkpoint contains **two** action prediction pathways:

1. **Tokenized Actions** (Language Model Logits):
   ```python
   # VLA generates logits for entire vocabulary (32000 tokens)
   logits = vla.forward(obs, prompt)  # (..., 32000)
   
   # Extract action token logits (last 256 tokens)
   action_logits = logits[..., -256-64:-64]  # (..., 256)
   
   # Sample/argmax token IDs
   action_tokens = torch.multinomial(softmax(action_logits / temp))
   
   # Detokenize to continuous actions
   actions = tokenizer.detokenize(action_tokens)  # [-1, 1]^7
   ```
   
   **Pros for PPO**:
   - ✅ Natural probability distribution (softmax over tokens)
   - ✅ Easy to compute log probabilities for policy gradient
   - ✅ Stochastic by design (controllable via temperature)
   - ✅ No additional network needed
   - ✅ Matches SimpleVLA-RL architecture
   
2. **L1 Regression** (Direct MLP Prediction):
   ```python
   # Extract hidden states from VLA
   hidden = vla.get_hidden_states(obs, prompt)  # (..., 4096)
   
   # Pass through L1 regression head (3-layer MLP)
   actions = l1_head(hidden)  # [-1, 1]^7
   ```
   
   **Pros for Supervised Learning**:
   - ✅ Direct continuous output (no discretization)
   - ✅ Smooth action space
   - ✅ Used in original OpenVLA pre-training
   
   **Cons for PPO**:
   - ❌ Deterministic by default (no natural stochasticity)
   - ❌ Needs separate stochastic head for exploration
   - ❌ More complex to compute log probabilities
   - ❌ Additional 167M parameters (~668MB)

**Our Choice**: Use **tokenized actions** for PPO training, matching SimpleVLA-RL's proven approach. The L1 head is optionally loaded for comparison with OpenVLA-OFT or supervised learning but not used for PPO.

**Configuration**:
```python
# PPO training (recommended)
OpenVLAActorConfig(
    load_l1_action_head=False,      # Don't load - saves 668MB
    use_tokenized_actions=True,     # Use token logits
)

# Comparison mode (if needed)
OpenVLAActorConfig(
    load_l1_action_head=True,       # Load for comparison
    freeze_l1_action_head=True,     # Frozen (read-only)
    use_tokenized_actions=True,     # Still use tokenized for PPO
)
```

---

## Conclusion

The trajectory-based PPO implementation is complete and ready for testing. The architecture follows proven patterns from SimpleVLA-RL while adapting for OpenVLA's 7B parameter scale and LIBERO's sparse reward structure.

**Key Achievements**:
- ✅ Action tokenization with <1% reconstruction error
- ✅ Trajectory buffer with GRPO advantages
- ✅ PPO policy gradient with asymmetric clipping
- ✅ Per-sample gradient accumulation for memory efficiency
- ✅ Complete configuration system with documentation
- ✅ Modular code structure (config, value head, algorithms)

**Next Milestone**: Successfully complete first policy update without OOM and validate training loop stability over 1000 steps.

---

**Last Updated**: November 29, 2025  
**Author**: Implementation based on SimpleVLA-RL and OpenVLA-OFT  
**Status**: Phase 1 & 2 Complete, Phase 3 Ready to Test

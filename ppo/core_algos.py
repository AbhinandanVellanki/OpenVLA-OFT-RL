"""
core_algos.py

Core PPO algorithm functions for policy gradient computation.
"""

from typing import Tuple
import torch
import torch.nn.functional as F


def logprobs_from_logits(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Compute log probabilities from logits for given token IDs.
    
    Args:
        logits: Logits from model, shape (batch, seq_len, vocab_size)
        token_ids: Token IDs to compute log probs for, shape (batch, seq_len)
    
    Returns:
        log_probs: Log probabilities, shape (batch, seq_len)
    """
    # Clamp logits to prevent extreme values that cause overflow
    # Range [-100, 100] is safe for log_softmax computation
    logits = torch.clamp(logits, min=-100.0, max=100.0)

    # Compute log softmax
    log_probs_all = F.log_softmax(logits, dim=-1)

    # Assert shapes for safety
    assert log_probs_all.shape[:-1] == token_ids.shape, f"Shape mismatch: log_probs_all {log_probs_all.shape}, token_ids {token_ids.shape}"

    # Gather log probs for specific tokens
    log_probs = torch.gather(log_probs_all, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)

    # Clamp log probs to prevent extreme values
    # log(1e-10) ≈ -23, so [-50, 0] is a safe range
    log_probs = torch.clamp(log_probs, min=-50.0, max=0.0)

    return log_probs


def compute_policy_loss(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_ratio_high: float = 0.28,
    clip_ratio_low: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute PPO clipped surrogate policy loss with asymmetric clipping.
    
    Args:
        old_log_probs: Log probs from rollout policy, shape (batch, seq_len)
        new_log_probs: Log probs from current policy, shape (batch, seq_len)
        advantages: Advantage estimates, shape (batch, seq_len) or (batch,)
        mask: Valid timestep mask, shape (batch, seq_len)
        clip_ratio_high: Upper clip ratio (default: 0.28)
        clip_ratio_low: Lower clip ratio (default: 0.2)
    
    Returns:
        loss: Policy gradient loss (scalar)
        clipfrac: Fraction of ratios that were clipped (scalar)
        approx_kl: Approximate KL divergence (scalar)
    """
    # Compute importance sampling ratio
    log_ratio = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    
    # Broadcast advantages if needed
    if advantages.dim() == 1 and ratio.dim() == 2:
        advantages = advantages.unsqueeze(-1)
    
    # PPO clipped objective with asymmetric clipping
    policy_loss_1 = -advantages * ratio
    policy_loss_2 = -advantages * torch.clamp(
        ratio,
        1.0 - clip_ratio_low,
        1.0 + clip_ratio_high
    )
    
    # Take maximum (more conservative)
    policy_loss = torch.max(policy_loss_1, policy_loss_2)
    
    # Apply mask and average
    if mask is not None:
        policy_loss = (policy_loss * mask).sum() / mask.sum()
    else:
        policy_loss = policy_loss.mean()
    
    # Compute metrics
    with torch.no_grad():
        # Fraction of ratios that were clipped
        clipped = ((ratio < (1.0 - clip_ratio_low)) | (ratio > (1.0 + clip_ratio_high))).float()
        if mask is not None:
            clipfrac = (clipped * mask).sum() / mask.sum()
        else:
            clipfrac = clipped.mean()
        
        # Approximate KL divergence
        approx_kl = ((ratio - 1) - log_ratio).mean()
    
    return policy_loss, clipfrac, approx_kl


def apply_mask_with_grad_control(
    tensor: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Apply mask to tensor while preserving gradient flow.
    
    Uses torch.where to zero masked positions without breaking gradients.
    
    Args:
        tensor: Input tensor, shape (batch, seq_len, ...)
        mask: Boolean mask, shape (batch, seq_len)
    
    Returns:
        masked_tensor: Masked tensor with same shape as input
    """
    # Expand mask to match tensor dimensions
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(-1)
    
    # Use torch.where to preserve gradients
    return torch.where(mask, tensor, torch.zeros_like(tensor))

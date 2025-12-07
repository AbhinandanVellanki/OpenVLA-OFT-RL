"""
Start GAE-PPO Training for OpenVLA

Python entrypoint for GAE-based PPO training with full wandb and matplotlib logging.

Usage:
    # Single task with GAE
    python start_gae_ppo_training.py --task-id 0 --timesteps 100000

    # Multi-task with GAE
    python start_gae_ppo_training.py --task-ids 0 1 2 3 --num-envs 4 --timesteps 200000

    # Different GPU
    python start_gae_ppo_training.py --task-id 0 --gpu 0
"""

import argparse
import os
import sys
from pathlib import Path

# Set environment variables before imports
os.environ['PYTHONWARNINGS'] = 'ignore'

import torch
import wandb

from vla_oft.min_vla.config import OpenVLAActorConfig
from ppo.config import PPOConfig
from OpenVLA_PPO import OpenVLAPPO, training
from libero_rl.utils.task_utils import get_task


def parse_args():
    parser = argparse.ArgumentParser(description="GAE-PPO Training for OpenVLA")

    # Task configuration
    parser.add_argument("--task-suite", type=str, default="libero_spatial",
                        help="LIBERO task suite (default: libero_spatial)")
    parser.add_argument("--task-id", type=int, default=0,
                        help="Single task ID (default: 0)")
    parser.add_argument("--task-ids", type=int, nargs="+", default=None,
                        help="Multi-task IDs for vectorized training")

    # Training configuration
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="Total timesteps (default: 100000)")
    parser.add_argument("--n-steps", type=int, default=512,
                        help="Rollout length (default: 512)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Minibatch size (default: 2)")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="Optimization epochs per update (default: 10)")

    # GAE configuration
    parser.add_argument("--use-gae", action="store_true", default=True,
                        help="Enable GAE advantage estimation (default: True)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda parameter (default: 0.95)")
    parser.add_argument("--freeze-vla-for-critic", action="store_true",
                        help="Detach VLA gradients for critic (default: False)")

    # Learning rates
    parser.add_argument("--actor-lr", type=float, default=1e-5,
                        help="Actor learning rate (default: 1e-5)")
    parser.add_argument("--critic-lr", type=float, default=3e-4,
                        help="Critic learning rate (default: 3e-4)")

    # Environment
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Number of parallel environments (default: 1)")

    # Device
    parser.add_argument("--gpu", type=int, default=1,
                        help="GPU ID (default: 1)")

    # Logging
    parser.add_argument("--wandb-project", type=str, default="openvla-gae-ppo",
                        help="Wandb project name (default: openvla-gae-ppo)")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="Wandb entity (default: None)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name for logging (default: auto-generated)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")

    # Model
    parser.add_argument("--use-lora", action="store_true", default=True,
                        help="Use LoRA for VLA (default: True)")
    parser.add_argument("--lora-rank", type=int, default=64,
                        help="LoRA rank (default: 64)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = f"cuda:0"  # After setting CUDA_VISIBLE_DEVICES, always use cuda:0

    print("="*80)
    print("GAE-PPO Training for OpenVLA")
    print("="*80)
    print(f"Task Suite:       {args.task_suite}")
    print(f"Task ID(s):       {args.task_ids if args.task_ids else args.task_id}")
    print(f"Total Timesteps:  {args.timesteps}")
    print(f"Use GAE:          {args.use_gae}")
    print(f"GAE Lambda:       {args.gae_lambda}")
    print(f"Actor LR:         {args.actor_lr}")
    print(f"Critic LR:        {args.critic_lr}")
    print(f"GPU:              {args.gpu} (mapped to {device})")
    print(f"Num Envs:         {args.num_envs}")
    print("="*80)
    print()

    # Create configs
    vla_config = OpenVLAActorConfig(
        device=device,
        training_device=device,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        freeze_vla_backbone=True,
        use_tokenized_actions=True,
        load_l1_action_head=False,
    )

    ppo_config = PPOConfig(
        # Training
        total_timesteps=args.timesteps,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,

        # GAE
        use_gae=args.use_gae,
        gae_lambda=args.gae_lambda,
        freeze_vla_for_critic=args.freeze_vla_for_critic,

        # Learning rates
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,

        # Environment
        task_suite=args.task_suite,
        task_id=args.task_id,
        task_ids=args.task_ids,
        num_envs=args.num_envs,

        # Logging
        use_wandb=not args.no_wandb,
        wandb_entity=args.wandb_entity,

        # Device
        device=device,
        training_device=device,
    )

    # Initialize wandb
    if ppo_config.use_wandb:
        run_name = args.run_name or f"gae-ppo_{args.task_suite}_task{args.task_id}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                "task_suite": args.task_suite,
                "task_id": args.task_id,
                "task_ids": args.task_ids,
                "use_gae": args.use_gae,
                "gae_lambda": args.gae_lambda,
                "actor_lr": args.actor_lr,
                "critic_lr": args.critic_lr,
                "timesteps": args.timesteps,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "n_epochs": args.n_epochs,
                "lora_rank": args.lora_rank,
                "freeze_vla_for_critic": args.freeze_vla_for_critic,
            },
        )
        print(f"✓ Wandb initialized: {args.wandb_project}/{run_name}")
        print()

    # Get task prompt
    if args.task_ids:
        task_prompts = [get_task(args.task_suite, tid).language for tid in args.task_ids]
    else:
        task = get_task(args.task_suite, args.task_id)
        task_prompts = [task.language]

    print(f"Task prompt(s): {task_prompts}")
    print()

    # Run training
    try:
        training(vla_config=vla_config, ppo_config=ppo_config)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if ppo_config.use_wandb:
            wandb.finish()
            print("\n✓ Wandb run finished")


if __name__ == "__main__":
    main()

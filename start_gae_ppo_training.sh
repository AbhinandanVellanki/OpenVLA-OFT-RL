#!/bin/bash
#
# Start GAE-based PPO Training for OpenVLA on LIBERO
#
# This script launches PPO training with GAE (Generalized Advantage Estimation)
# instead of GRPO for advantage computation.
#
# Usage:
#   ./start_gae_ppo_training.sh                    # Single task, default config
#   ./start_gae_ppo_training.sh --task-id 2        # Different task
#   ./start_gae_ppo_training.sh --timesteps 50000  # Custom timesteps
#

# Exit on error
set -e

# Configuration
TASK_SUITE="libero_spatial"
TASK_ID=0
TIMESTEPS=100000
GPU_ID=1
CONFIG="configs/gae_ppo_config.yaml"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task-suite)
            TASK_SUITE="$2"
            shift 2
            ;;
        --task-id)
            TASK_ID="$2"
            shift 2
            ;;
        --timesteps)
            TIMESTEPS="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Print configuration
echo "=========================================="
echo "GAE-PPO Training Configuration"
echo "=========================================="
echo "Task Suite:     $TASK_SUITE"
echo "Task ID:        $TASK_ID"
echo "Total Steps:    $TIMESTEPS"
echo "GPU ID:         $GPU_ID"
echo "Config File:    $CONFIG"
echo "=========================================="
echo ""

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Run training
python gae_ppo_example.py \
    --task-suite $TASK_SUITE \
    --task-id $TASK_ID \
    --timesteps $TIMESTEPS \
    --config $CONFIG \
    --use-gae

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="

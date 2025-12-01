#!/bin/bash
# Start PPO training completely detached from terminal/SSH
# This will survive SSH disconnects and terminal timeouts

cd /home/abhi/Documents/Deep-RL/OpenVLA-OFT-RL

# Kill any existing training
pkill -f "OpenVLA_PPO.py"

# Start training completely detached
nohup bash -c '
    # Activate conda
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate oft_rl
    
    # Enable expandable memory segments to handle fragmentation
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    
    # Run PPO training with wandb logging
    python OpenVLA_PPO.py \
        --task-suite libero_spatial \
        --task-id 0 \
        --timesteps 100000 \
' > ppo_training.log 2>&1 &

# Get the PID
TRAIN_PID=$!

# Completely detach from shell (both -a and -h for maximum protection)
disown -a

echo "PPO Training started with PID: $TRAIN_PID"
echo "Completely detached from terminal"
echo ""
echo "Configuration:"
echo "  - Task: libero_spatial, task_id=0"
echo "  - Total timesteps: 100000"
echo "  - Device: cuda:1"
echo "  - Batch size: 1"
echo "  - n_steps: 50"
echo "  - Wandb: enabled"
echo ""
echo "To monitor: tail -f ppo_training.log"
echo "To check if running: ps -p $TRAIN_PID"
echo "To kill: kill $TRAIN_PID"
echo ""
echo "PID saved to: ppo_train.pid"
echo $TRAIN_PID > ppo_train.pid

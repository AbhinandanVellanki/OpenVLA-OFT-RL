#!/bin/bash
# Start PPO training with full cleanup and detachment
# Handles CUDA memory cleanup and survives SSH disconnects

export CUDA_VISIBLE_DEVICES=1

echo "========================================="
echo "PPO Training Launcher"
echo "========================================="

cd /home/abhi/Documents/Deep-RL/OpenVLA-OFT-RL

# Activate conda environment first
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate oft_rl
echo "✓ Using Python: $(which python)"
echo "✓ Python version: $(python --version)"

# Kill any existing training processes
echo ""
echo "Cleaning up existing processes..."
pkill -9 -f "python.*OpenVLA_PPO.py" || true
sleep 2

# Clear CUDA memory
echo "Clearing CUDA cache..."
python -c "import torch; torch.cuda.empty_cache(); print('✓ CUDA cache cleared')" 2>/dev/null || echo "⚠ Could not clear CUDA cache"

# Check GPU status
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "⚠ Could not query GPU"

echo ""
echo "========================================="
echo "Starting PPO Training (detached)"
echo "========================================="

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
        --timesteps 10000000 \
' > ppo_training.log 2>&1 &

# Get the PID
TRAIN_PID=$!

# Completely detach from shell
disown -a

# Save PID
echo $TRAIN_PID > ppo_train.pid

echo ""
echo "✓ Training started with PID: $TRAIN_PID"
echo "✓ Completely detached from terminal"
echo ""
echo "Configuration:"
echo "  - Task: libero_spatial, task_id=0"
echo "  - Total timesteps: 10,000,000"
echo "  - Action chunking: 8 actions/query"
echo "  - GPU config: Read from vla_oft/min_vla/config.py"
echo "  - Wandb: enabled"
echo ""
echo "Monitor with:"
echo "  tail -f ppo_training.log"
echo ""
echo "Check status:"
echo "  ps -p $TRAIN_PID"
echo ""
echo "Stop training:"
echo "  kill $TRAIN_PID"
echo ""
echo "========================================="

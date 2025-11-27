# OpenVLA-OFT-RL

OpenVLA with Optimal Fine-Tuning for Reinforcement Learning.

## Installation

### Quick Setup

1. Create a conda environment:
```bash
conda create -n oft_rl python=3.10
conda activate oft_rl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Exact Environment Replication

For reproducing the exact environment:
```bash
pip install -r requirements_frozen.txt
```

## Critical Version Requirements

- **PyTorch**: 2.2.2 (compatible with CUDA 12.x)
- **NumPy**: 1.24.4 (avoids numpy 2.0 breaking changes)
- **Transformers**: 4.40.1 (matches OpenVLA model training version)

## Testing

Run tests to verify installation:
```bash
cd vla-oft
python tests/test_forward_pass.py
```

All tests should pass.

#!/bin/bash
# Setup script for OpenVLA-OFT-RL environment - for 5090
# This installs dependencies in the correct order to avoid flash-attn build errors

set -e  # Exit on error

echo "Creating conda environment oft_rl with Python 3.10..."
conda create -n oft_rl python=3.10 -y

echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate oft_rl

echo "Installing conda packages..."
conda install -y -c defaults \
    blas=1.0=mkl \
    intel-openmp=2023.0.0 \
    mkl=2023.1.0 \
    mkl-service=2.4.0 \
    numpy=1.26.4 \
    farama-notifications=0.0.4 \
    gymnasium=0.28.1 \
    jax-jumpy=1.0.0

echo "Installing PyTorch and related packages first (required for flash-attn)..."
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

echo "Installing flash-attn (this may take several minutes to compile)..."
pip install flash-attn==2.7.3 --no-build-isolation

echo "Installing remaining pip packages..."
pip install \
    absl-py==2.3.1 \
    accelerate==1.12.0 \
    antlr4-python3-runtime==4.9.3 \
    attrs==25.4.0 \
    bddl==1.0.1 \
    bitsandbytes==0.42.0 \
    certifi==2025.11.12 \
    charset-normalizer==3.4.4 \
    click==8.3.1 \
    cloudpickle==2.1.0 \
    cycler==0.12.1 \
    docker-pycreds==0.4.0 \
    easydict==1.9 \
    egl-probe==1.0.2 \
    einops==0.4.1 \
    etils==1.13.0 \
    exceptiongroup==1.3.1 \
    fastjsonschema==2.21.2 \
    filelock==3.20.0 \
    fonttools==4.60.1 \
    fsspec==2025.10.0 \
    future==0.18.2 \
    gitdb==4.0.12 \
    gitpython==3.1.45 \
    glfw==2.10.0 \
    grpcio==1.76.0 \
    gym==0.25.2 \
    gym-notices==0.1.0 \
    h5py==3.15.1 \
    hf-xet==1.2.0 \
    huggingface-hub==0.36.0 \
    hydra-core==1.2.0 \
    idna==3.11 \
    imageio==2.37.2 \
    imageio-ffmpeg==0.6.0 \
    importlib-resources==6.5.2 \
    iniconfig==2.3.0 \
    jinja2==3.1.6 \
    jsonschema==4.25.1 \
    jsonschema-specifications==2025.9.1 \
    jupyter-core==5.9.1 \
    jupytext==1.18.1 \
    kiwisolver==1.4.9 \
    libero==0.1.0 \
    llvmlite==0.45.1 \
    markdown==3.10 \
    markdown-it-py==4.0.0 \
    markupsafe==3.0.3 \
    matplotlib==3.5.3 \
    mdit-py-plugins==0.5.0 \
    mdurl==0.1.2 \
    mpmath==1.3.0 \
    mujoco==3.3.7 \
    nbformat==5.10.4 \
    networkx==3.4.2 \
    numba==0.62.1 \
    nvidia-cublas-cu12==12.1.3.1 \
    nvidia-cuda-cupti-cu12==12.1.105 \
    nvidia-cuda-nvrtc-cu12==12.1.105 \
    nvidia-cuda-runtime-cu12==12.1.105 \
    nvidia-cudnn-cu12==8.9.2.26 \
    nvidia-cufft-cu12==11.0.2.54 \
    nvidia-cufile-cu12==1.13.1.3 \
    nvidia-curand-cu12==10.3.2.106 \
    nvidia-cusolver-cu12==11.4.5.107 \
    nvidia-cusparse-cu12==12.1.0.106 \
    nvidia-cusparselt-cu12==0.7.1 \
    nvidia-nccl-cu12==2.19.3 \
    nvidia-nvjitlink-cu12==12.9.86 \
    nvidia-nvshmem-cu12==3.3.20 \
    nvidia-nvtx-cu12==12.1.105 \
    omegaconf==2.3.0 \
    opencv-python==4.6.0.66 \
    packaging==25.0 \
    pathtools==0.1.2 \
    peft==0.12.0 \
    pillow==12.0.0 \
    platformdirs==4.5.0 \
    pluggy==1.6.0 \
    promise==2.3 \
    protobuf==3.20.3 \
    psutil==7.1.3 \
    pygments==2.19.2 \
    pyopengl==3.1.10 \
    pyparsing==3.2.5 \
    pytest==9.0.1 \
    python-dateutil==2.9.0.post0 \
    pyyaml==6.0.3 \
    referencing==0.37.0 \
    regex==2025.11.3 \
    requests==2.32.5 \
    robomimic==0.2.0 \
    robosuite==1.4.0 \
    rpds-py==0.29.0 \
    safetensors==0.7.0 \
    scipy==1.13.1 \
    sentry-sdk==2.46.0 \
    setproctitle==1.3.7 \
    shortuuid==1.0.13 \
    six==1.17.0 \
    smmap==5.0.2 \
    sympy==1.14.0 \
    tensorboard==2.20.0 \
    tensorboard-data-server==0.7.2 \
    tensorboardx==2.6.4 \
    termcolor==3.2.0 \
    thop==0.1.1-2209072238 \
    timm==0.9.16 \
    tokenizers==0.19.1 \
    tomli==2.3.0 \
    tqdm==4.67.1 \
    traitlets==5.14.3 \
    transformers==4.40.1 \
    triton==2.2.0 \
    urllib3==2.5.0 \
    wandb==0.13.1 \
    werkzeug==3.1.3 \
    zipp==3.23.0

echo ""
echo "✓ Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate oft_rl"
echo ""
echo "To verify PyTorch CUDA setup, run:"
echo "  python -c \"import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')\""

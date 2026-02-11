#!/bin/bash
# Quick-start script for multi-GPU training of HeteroEdgeNet
#
# Usage:
#   ./run_multi_gpu.sh                # Use all GPUs, small model
#   ./run_multi_gpu.sh --large        # Use all GPUs, large model
#   ./run_multi_gpu.sh 0,1            # Use GPUs 0 and 1, small model
#   ./run_multi_gpu.sh 0,1 --large    # Use GPUs 0 and 1, large model

set -e  # Exit on error

# Parse arguments
GPUS=""
MODEL_SIZE="small"

for arg in "$@"; do
    if [ "$arg" = "--large" ]; then
        MODEL_SIZE="large"
    else
        GPUS="$arg"
    fi
done

if [ "$MODEL_SIZE" = "large" ]; then
    TRAIN_SCRIPT="train_hetero_large.py"
else
    TRAIN_SCRIPT="train_hetero.py"
fi

echo "=========================================="
echo "GNN Chess - HeteroEdgeNet Training ($MODEL_SIZE)"
echo "=========================================="
echo

# Detect if pixi is available, otherwise use system python
if command -v pixi &> /dev/null; then
    PYTHON_CMD="pixi run python"
    echo "Using pixi environment"
else
    PYTHON_CMD="python3"
    echo "Using system Python (pixi not found)"
fi

# Check if JAX with GPU support is available
echo "Checking JAX GPU support..."
$PYTHON_CMD -c "
import jax
devices = jax.devices()
print(f'Available devices: {devices}')
print(f'Device count: {jax.device_count()}')
print(f'Backend: {jax.default_backend()}')
if jax.default_backend() != 'gpu':
    print('WARNING: JAX is not using GPU backend!')
    print('May need to reinstall JAX with CUDA support')
" || {
    echo "Note: Running on CPU (no GPU detected or JAX not GPU-enabled)"
}

echo
echo "Starting training..."
echo

if [ -n "$GPUS" ]; then
    echo "Using GPUs: $GPUS"
    CUDA_VISIBLE_DEVICES=$GPUS XLA_PYTHON_CLIENT_ALLOCATOR=platform $PYTHON_CMD $TRAIN_SCRIPT
else
    echo "Using all available GPUs"
    XLA_PYTHON_CLIENT_ALLOCATOR=platform $PYTHON_CMD $TRAIN_SCRIPT
fi

#!/bin/bash
# Quick-start script for multi-GPU training of HeteroEdgeNet
#
# Usage:
#   ./run_multi_gpu.sh          # Use all available GPUs
#   ./run_multi_gpu.sh 0,1      # Use GPUs 0 and 1
#   ./run_multi_gpu.sh 0        # Use only GPU 0 (for testing)

set -e  # Exit on error

# Default: use all GPUs if not specified
GPUS=${1:-""}

echo "=========================================="
echo "GNN Chess - HeteroEdgeNet Training"
echo "=========================================="
echo

# Check if JAX with GPU support is available
echo "Checking JAX GPU support..."
python -c "
import jax
devices = jax.devices()
print(f'Available devices: {devices}')
print(f'Device count: {jax.device_count()}')
print(f'Backend: {jax.default_backend()}')
if jax.default_backend() != 'gpu':
    print('WARNING: JAX is not using GPU backend!')
    print('Install JAX with CUDA support:')
    print('  pip install --upgrade \"jax[cuda12]\"')
    exit(1)
" || exit 1

echo
echo "Starting training..."
echo

if [ -n "$GPUS" ]; then
    echo "Using GPUs: $GPUS"
    CUDA_VISIBLE_DEVICES=$GPUS python train_hetero.py
else
    echo "Using all available GPUs"
    python train_hetero.py
fi

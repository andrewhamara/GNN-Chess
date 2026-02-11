#!/usr/bin/env python3
"""Train large HeteroEdgeNet (inner_size=128, 5 GNN layers) with multi-GPU support.

This is the full-size heterogeneous model, matching EdgeNet2 base dimensions.
Batch sizes are halved vs train_hetero.py to accommodate the ~10x parameter count.

Usage:
    # Single GPU
    CUDA_VISIBLE_DEVICES=0 python train_hetero_large.py

    # Multi-GPU (4 GPUs)
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train_hetero_large.py

    # All GPUs
    python train_hetero_large.py
"""

import faulthandler
faulthandler.enable()

# Import the main training module
import train

# ============================================================================
# Override config for large heterogeneous graph training
# ============================================================================

train.config['hetero_graph'] = True

# Model architecture (full-size, matching EdgeNet2 base dimensions)
train.config['n_gnn_layers'] = 5
train.config['inner_size'] = 128

# Training hyperparameters (halved batch sizes to avoid OOM with ~10x params)
train.config['learning_rate'] = 0.001
train.config['training_batch_size'] = 2**6  # 64 (down from 128)
train.config['selfplay_batch_size'] = 128   # (down from 256)

# MCTS settings
train.config['num_simulations'] = 128
train.config['max_num_steps'] = 512  # 256 for gardner, 512 for chess

# Training schedule
train.config['n_iter'] = 100
train.config['eval_interval'] = 2
train.config['n_training_pass'] = 1

# Data management
train.config['window_size'] = 1_000_000
train.config['shuffle_window'] = True

# ============================================================================
# Debug/profiling settings
# ============================================================================

train.config['debug'] = True  # Skip Aim logging

# Fix LLVM mma16816 error by disabling tensor cores and forcing float32
import os
os.environ['XLA_FLAGS'] = (
    os.environ.get('XLA_FLAGS', '') +
    ' --xla_gpu_enable_triton_gemm=false'
    ' --xla_gpu_triton_gemm_any=false'
    ' --xla_gpu_force_compilation_parallelism=1'
)

# Force float32 (disable bfloat16/float16 which may trigger mma16816)
import jax
jax.config.update('jax_default_matmul_precision', 'float32')

# ============================================================================
# Run training
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Large HeteroEdgeNet Training Configuration")
    print("=" * 80)
    print(f"Heterogeneous graph: {train.config['hetero_graph']}")
    print(f"GNN layers: {train.config['n_gnn_layers']}")
    print(f"Inner size: {train.config['inner_size']}")
    print(f"Learning rate: {train.config['learning_rate']}")
    print(f"Training batch size: {train.config['training_batch_size']}")
    print(f"Selfplay batch size: {train.config['selfplay_batch_size']}")
    print(f"Num simulations: {train.config['num_simulations']}")
    print(f"Devices: {train.devices}")
    print(f"Num devices: {train.num_devices}")
    print("=" * 80)
    print()

    # Start training
    train.main()

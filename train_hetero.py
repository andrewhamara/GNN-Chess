#!/usr/bin/env python3
"""Train HeteroEdgeNet (7-edge-type heterogeneous GNN) with multi-GPU support.

This script enables the heterogeneous multi-graph architecture and adjusts
hyperparameters for optimal training with the new edge types.

Usage:
    # Single GPU
    CUDA_VISIBLE_DEVICES=0 python train_hetero.py

    # Multi-GPU (4 GPUs)
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train_hetero.py

    # All GPUs
    python train_hetero.py
"""

import faulthandler
faulthandler.enable()

# Import the main training module
import train

# ============================================================================
# Override config for heterogeneous graph training
# ============================================================================

train.config['hetero_graph'] = True

# Hyperparameter adjustments for hetero model
# The hetero model has ~3x more parameters due to 7 edge types with separate
# processing, so we might want to:
# - Reduce layers slightly (more capacity per layer)
# - Adjust batch sizes if OOM occurs
# - Monitor convergence carefully

# Model architecture (reduced to avoid LLVM compilation issues)
train.config['n_gnn_layers'] = 2  # Reduced to 2 for initial testing
train.config['inner_size'] = 64   # Reduced to avoid compiler complexity

# Training hyperparameters
train.config['learning_rate'] = 0.001  # Keep same initially
train.config['training_batch_size'] = 2**7  # 128 - reduce to 2**6 if OOM
train.config['selfplay_batch_size'] = 256  # Reduce to 128 if OOM

# MCTS settings
train.config['num_simulations'] = 128  # Keep same (or reduce to 64 if slow)
train.config['max_num_steps'] = 512  # 256 for gardner, 512 for chess

# Training schedule
train.config['n_iter'] = 100  # Number of training iterations
train.config['eval_interval'] = 2  # Evaluate every N iterations
train.config['n_training_pass'] = 1  # Epochs per iteration

# Data management
train.config['window_size'] = 1_000_000  # Replay buffer size
train.config['shuffle_window'] = True

# ============================================================================
# Optional: Debug/profiling settings
# ============================================================================

# Disable Aim experiment tracking (if Aim server not running)
train.config['debug'] = True  # Skip Aim logging

# Fix LLVM mma16816 error by disabling tensor cores and forcing float32
# This is a workaround for an LLVM compiler bug with complex models
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

# Uncomment to enable JAX profiling
# import jax
# jax.config.update('jax_log_compiles', True)  # Log compilation times

# Uncomment for more verbose output
# import jax
# jax.config.update('jax_debug_nans', True)
# jax.config.update('jax_debug_infs', True)

# ============================================================================
# Run training
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("HeteroEdgeNet Training Configuration")
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

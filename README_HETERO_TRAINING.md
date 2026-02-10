# Quick Start: Multi-GPU HeteroEdgeNet Training

## TL;DR

```bash
# 1. Verify GPU setup
python -c "import jax; print(jax.devices())"

# 2. Start training (uses all GPUs automatically)
./run_multi_gpu.sh

# OR specify GPUs manually:
./run_multi_gpu.sh 0,1,2,3    # Use 4 GPUs
```

## What is HeteroEdgeNet?

A heterogeneous multi-graph GNN with **7 edge types**:
1. **Move edges** (dynamic) - legal moves for current player
2. **Grid edges** (static) - 8-connected spatial adjacency
3. **Attack edges** (dynamic) - piece attack relationships
4. **Defense edges** (dynamic) - piece defense relationships
5. **File edges** (static) - same-file connections
6. **Rank edges** (static) - same-rank connections
7. **Diagonal edges** (static) - diagonal connections

Each edge type has separate message passing via GATEAU, then node updates are summed.

## Architecture Overview

```
Input: Chess position (8×8×119 observation)
  ↓
Node embedding: Dense(128)
Edge embeddings: 7× Dense(128) (one per edge type)
  ↓
4× HeteroEGNN blocks:
  ├─ 2× HeteroGATEAU (separate attention per edge type)
  └─ Residual connections on nodes + edges
  ↓
Policy head: BNR → Dense → logits (uses move edges)
Value head: BNR → AttentionPooling → Dense → tanh
```

## Files Created

- **`train_hetero.py`** - Training script with hetero graph enabled
- **`run_multi_gpu.sh`** - Quick-start shell script
- **`TRAINING_GUIDE.md`** - Comprehensive guide with troubleshooting
- **`README_HETERO_TRAINING.md`** - This file (quick reference)

## Configuration

Default settings in `train_hetero.py`:

```python
config = {
    'hetero_graph': True,        # Use heterogeneous graph
    'n_gnn_layers': 4,            # Reduced from 5 (more capacity per layer)
    'inner_size': 128,            # Feature dimension
    'learning_rate': 0.001,       # Adam learning rate
    'training_batch_size': 128,   # Batch size per training step
    'selfplay_batch_size': 256,   # Games generated per iteration
    'num_simulations': 128,       # MCTS simulations per move
    'n_iter': 100,                # Total iterations
    'eval_interval': 2,           # Evaluate every N iterations
}
```

## Expected Behavior

### First Iteration (JIT Compilation)
- **Duration:** 2-10 minutes (one-time compilation)
- **GPU utilization:** Low during compilation, high after
- **Console output:** Compilation traces, then progress bars

### Subsequent Iterations
- **Selfplay:** ~30-120 seconds (generates 256 games)
- **Training:** ~10-30 seconds (gradient updates)
- **Evaluation:** ~60-180 seconds (plays vs baseline every 2 iterations)

### Memory Usage
- **Per GPU:** ~3-5 GB (with default settings)
- **Total (8 GPUs):** ~24-40 GB

## Monitoring

### Real-time GPU stats
```bash
watch -n 1 nvidia-smi
```

### Training metrics
The script prints:
- Loss (policy + value)
- Win/Draw/Loss rates vs baseline
- ELO rating estimates
- Time per phase (selfplay, train, eval)

### Logs
- **Models:** `models/chess_YYYY-MM-DD:HHhMM/*.ckpt`
- **Games:** `games/chess_YYYY-MM-DD:HHhMM/*.pgn`
- **Aim dashboard:** `http://localhost:53800` (if Aim server running)

## Troubleshooting

### OOM (Out of Memory) Errors

Reduce in this order:

```python
# In train_hetero.py:
train.config['training_batch_size'] = 2**6  # 64 instead of 128
train.config['selfplay_batch_size'] = 128   # 128 instead of 256
train.config['num_simulations'] = 64        # 64 instead of 128
train.config['n_gnn_layers'] = 3            # 3 instead of 4
```

### Only 1 GPU Used

Check:
```bash
# See available GPUs
nvidia-smi

# Check JAX detection
python -c "import jax; print(f'{jax.device_count()} GPUs detected')"

# Verify no restriction
echo $CUDA_VISIBLE_DEVICES  # Should be empty or "0,1,2,..."
```

### Slow Training

Expected speed (rough estimates):
- **8 GPUs:** ~5-10 min per iteration (including eval)
- **4 GPUs:** ~10-15 min per iteration
- **1 GPU:** ~30-60 min per iteration

If slower than expected:
1. Check GPU utilization (`nvidia-smi`) - should be >80%
2. Reduce `num_simulations` (MCTS overhead)
3. Ensure SSD storage (not HDD) for model checkpoints

### NaN in Training

If you see NaN errors:
```python
# Reduce learning rate
train.config['learning_rate'] = 0.0001

# Or enable gradient clipping (requires code edit in train.py)
```

## Testing Before Full Training

### Quick test (1 GPU, 5 iterations)
```python
# Edit train_hetero.py temporarily:
train.config['n_iter'] = 5
train.config['selfplay_batch_size'] = 32
train.config['num_simulations'] = 32

# Run:
CUDA_VISIBLE_DEVICES=0 python train_hetero.py
```

Should complete in ~10-15 minutes if everything works.

## Next Steps

1. **Monitor training:** Watch loss decrease and ELO increase
2. **Evaluate checkpoints:** Load best model and test against Stockfish
3. **Analyze edge contributions:** Try ablation studies (disable edge types)
4. **Hyperparameter tuning:** Grid search over layers, hidden size, learning rate

## Parameter Count

- **EdgeNet2 (baseline):** ~2-5M parameters
- **HeteroEdgeNet:** ~5-15M parameters (3× more due to 7 edge types)

This is still much smaller than AlphaZero (~100M), so we have room to scale up.

## Comparison to Baseline

| Metric | EdgeNet2 | HeteroEdgeNet |
|--------|----------|---------------|
| Edge types | 1 (moves) | 7 (moves + spatial + vision) |
| Parameters | ~3M | ~10M |
| Memory/GPU | ~2 GB | ~4 GB |
| Training time | 1× | ~1.5-2× |
| Expected performance | Baseline | +100-300 ELO (estimate) |

## References

- **Plan document:** See plan mode transcript for full architecture details
- **Implementation:** `chess_graph.py` (graph construction), `models.py` (HeteroEdgeNet)
- **Original paper inspiration:** Graph Attention Networks, Relational GCN

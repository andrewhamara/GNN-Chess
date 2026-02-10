# Multi-GPU Training Guide for Heterogeneous GNN Chess

## Prerequisites

1. **JAX with GPU support installed**
   ```bash
   # Check your CUDA version first
   nvidia-smi

   # Install JAX with CUDA support (example for CUDA 12)
   pip install --upgrade "jax[cuda12]"
   ```

2. **Verify GPU detection**
   ```python
   python -c "import jax; print(f'Devices: {jax.devices()}'); print(f'Device count: {jax.device_count()}')"
   ```

   Expected output for 8 GPUs:
   ```
   Devices: [cuda(id=0), cuda(id=1), ..., cuda(id=7)]
   Device count: 8
   ```

## Current Parallelization Status

✅ **The code is already multi-GPU ready!** It uses `jax.pmap` for:
- **Selfplay generation** (`@jax.pmap` on line 184)
- **Loss computation** (`@jax.pmap` on line 216)
- **Training step** (`@jax.pmap` on line 284)

## Enabling Hetero Graph Training

### Option 1: Direct Code Modification

Edit `train.py` line 45:
```python
'hetero_graph': True,  # Changed from False
```

### Option 2: Create a Separate Config (Recommended)

Create `train_hetero.py`:

```python
#!/usr/bin/env python3
"""Train HeteroEdgeNet with multi-GPU support."""

import faulthandler
faulthandler.enable()

# Import the main training script
import train

# Override config for hetero graph
train.config['hetero_graph'] = True

# Optional: Adjust hyperparameters for hetero model
# More parameters due to 7 edge types, so might need larger batch/fewer layers
train.config['n_gnn_layers'] = 4  # Reduce from 5 due to increased capacity
train.config['inner_size'] = 128  # Keep same or increase to 256
train.config['training_batch_size'] = 2**6  # Reduce if OOM
train.config['selfplay_batch_size'] = 128  # Reduce if OOM

# Run training
if __name__ == "__main__":
    train.main()
```

## Running Training

### Single-GPU (for testing)
```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

### Multi-GPU (2 GPUs)
```bash
CUDA_VISIBLE_DEVICES=0,1 python train_hetero.py
```

### Multi-GPU (4 GPUs)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_hetero.py
```

### Multi-GPU (8 GPUs - Full Setup)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_hetero.py
# Or simply:
python train_hetero.py  # Uses all available GPUs by default
```

## Monitoring

### GPU Utilization
In a separate terminal:
```bash
watch -n 1 nvidia-smi
```

### Training Progress
The script outputs:
- Loss (policy + value)
- Evaluation win/loss rate vs baseline
- ELO rating estimates
- Time elapsed

Logs are saved to Aim (accessible at `http://localhost:53800`)

### View Aim Dashboard
```bash
# If Aim server isn't running:
aim up --repo aim://localhost:53800
```

## Expected Memory Usage

### Per-GPU Memory Estimate (8x8 board, batch_size=128)

**EdgeNet2 (baseline):**
- Model params: ~2-5M parameters × 4 bytes ≈ 8-20 MB
- Activations (batch=128): ~500 MB - 2 GB
- **Total: ~2-3 GB per GPU**

**HeteroEdgeNet (new):**
- Model params: ~5-15M parameters × 4 bytes ≈ 20-60 MB (7 edge types!)
- Activations (batch=128): ~1-3 GB
- **Total: ~3-5 GB per GPU**

### Batch Size Tuning

If you encounter OOM errors, reduce these in order:

1. `training_batch_size` (line 60): Try `2**6` (64) or `2**5` (32)
2. `selfplay_batch_size` (line 56): Try `128` or `64`
3. `num_simulations` (line 54): Try `64` or `32`
4. `n_gnn_layers` (line 50): Try `3` or `4`

The code automatically divides batches across GPUs, so:
- 8 GPUs with `selfplay_batch_size=256` → 32 games per GPU
- 8 GPUs with `training_batch_size=128` → 16 samples per GPU

## Batch Size Constraints

The code enforces divisibility (lines 72-83):
```python
config['eval_batch_size'] = reduce_multiple(
    config['eval_batch_size'],
    num_devices
)
config['selfplay_batch_size'] = reduce_multiple(
    config['selfplay_batch_size'],
    max(1, (num_devices * config['training_batch_size']) // config['max_num_steps'])
)
config['window_size'] = reduce_multiple(
    max(config['window_size'], config['selfplay_batch_size'] * config['max_num_steps']),
    config['training_batch_size'] * num_devices
)
```

**This means:**
- Batch sizes must be divisible by `num_devices`
- Choose powers of 2 that divide evenly

## Troubleshooting

### Issue: "RuntimeError: CUDA out of memory"
**Solution:** Reduce batch sizes or number of layers (see "Batch Size Tuning")

### Issue: "ValueError: Batch size X is not divisible by num_devices Y"
**Solution:** Ensure batch sizes are multiples of your GPU count
```python
# For 8 GPUs, use: 8, 16, 32, 64, 128, 256, 512, ...
# For 4 GPUs, use: 4, 8, 16, 32, 64, 128, 256, ...
# For 2 GPUs, use: 2, 4, 8, 16, 32, 64, 128, ...
```

### Issue: "Slow compilation on first iteration"
**Expected behavior:** JAX JIT compiles on first run. This can take 2-10 minutes depending on model size. Subsequent iterations will be fast.

### Issue: Only 1 GPU being used
**Check:**
1. `nvidia-smi` shows multiple GPUs
2. JAX detects them: `python -c "import jax; print(jax.devices())"`
3. No `CUDA_VISIBLE_DEVICES` restriction
4. JAX CUDA installation: `python -c "import jax; print(jax.default_backend())"`

### Issue: NaN in gradients
The code has NaN detection enabled (line 26). If you see NaN errors:
1. Reduce learning rate: `'learning_rate': 0.0001`
2. Enable gradient clipping in `train()` function
3. Check if edge features have extreme values

## Performance Expectations

### Training Speed (rough estimates)
- **1 GPU:** ~5-10 iterations/hour
- **4 GPUs:** ~15-30 iterations/hour (3-4× speedup)
- **8 GPUs:** ~25-50 iterations/hour (5-7× speedup)

*Actual speed depends on GPU model, board size, and hyperparameters*

### Convergence
- Expect 50-100 iterations to see meaningful improvement
- Full training may take 500-1000 iterations
- Monitor ELO vs baseline every `eval_interval` iterations

## Example Session

```bash
# Terminal 1: Start training
cd /Users/andrewhamara/Desktop/dev/GNN-Chess
python train_hetero.py

# Terminal 2: Monitor GPUs
watch -n 1 nvidia-smi

# Terminal 3: View logs (if needed)
tail -f games/chess_$(date +%Y-%m-%d)*/*.pgn
```

## Checkpoints

Models are saved to `models/chess_YYYY-MM-DD:HHhMM/XXXXXX.ckpt` every `eval_interval` iterations.

Load a checkpoint:
```python
from models import load_model
import pgx

env = pgx.make('chess')
model, params = load_model(
    env,
    'models/chess_2026-02-09:15h30/000050.ckpt',
    'my_model'
)
```

## Next Steps After Training

1. **Evaluate vs Stockfish:** See `evaluate.py` (if exists) or create evaluation script
2. **Visualize attention:** Extract edge attention weights from GATEAU layers
3. **Ablation study:** Disable edge types to measure contribution
4. **Hyperparameter tuning:** Grid search over `n_gnn_layers`, `inner_size`, learning rate

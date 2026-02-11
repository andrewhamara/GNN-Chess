# Slow First Iteration - Compilation Analysis

## What You're Seeing

```
Generating ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% -:--:--
```

**Only GPU 0 active, others idle** - This is normal during compilation!

## What's Actually Happening

### During First Iteration:

1. **Graph construction JIT compilation** (2-5 minutes)
   - Heterogeneous graph with 7 edge types
   - Vision edge computation (ray-marching)
   - Complex boolean operations

2. **MCTS/Selfplay JIT compilation** (5-15 minutes)
   - Entire MCTS tree search
   - Model forward pass
   - Game simulation loop
   - **This is what's stuck right now**

3. **Training JIT compilation** (2-5 minutes)
   - Gradient computation
   - Optimizer updates

### Why Only GPU 0 is Active

JAX's `pmap` compilation strategy:
1. Compile on device 0 (primary device)
2. Replicate to all other devices
3. Execute in parallel

**During step 1, only GPU 0 works.** This is expected!

## Timeline Expectations

### Heterogeneous Graph (7 edge types):
- **Graph construction:** 2-5 min
- **MCTS compilation:** 10-20 min ⚠️ (you are here)
- **Training compilation:** 3-5 min
- **First iteration total:** 20-30 min

### Baseline (1 edge type):
- **Graph construction:** 30 sec
- **MCTS compilation:** 3-5 min
- **Training compilation:** 1-2 min
- **First iteration total:** 5-8 min

## How to Tell if It's Stuck vs Compiling

### Still compiling (normal):
```bash
nvidia-smi
# GPU 0: High utilization (80-100%)
# GPU 0: Memory allocated (2-4 GB)
# GPUs 1-7: Idle, but process attached

top -p $(pgrep -f train_hetero)
# Python process: 100-800% CPU
```

### Actually stuck (problem):
```bash
nvidia-smi
# GPU 0: 0% utilization for >5 minutes
# Or: Memory keeps growing beyond 10 GB

top
# Python process: <10% CPU
```

## Workarounds if Too Slow

### Option 1: Reduce Model Complexity (Already Applied)
You're already using:
```python
n_gnn_layers = 2
inner_size = 64
```

### Option 2: Reduce MCTS Simulations
Edit `train_hetero.py`:
```python
train.config['num_simulations'] = 32  # Instead of 128
train.config['selfplay_batch_size'] = 64  # Instead of 256
```

This compiles 4× faster!

### Option 3: Disable Some Edge Types Temporarily
Create `train_hetero_minimal.py`:
```python
# Temporarily test with fewer edge types to verify compilation works
# Then re-enable all 7 edge types
```

### Option 4: Enable Compilation Logging
Add to `train_hetero.py` BEFORE `import train`:
```python
import jax
jax.config.update('jax_log_compiles', True)
```

You'll see each function being compiled (proves it's working, not stuck).

### Option 5: Test Baseline First
```bash
# Kill current training (Ctrl+C)
python train.py  # Baseline EdgeNet2, compiles in ~5 min
```

If baseline works, the issue is hetero graph compilation complexity.

## Current Status Analysis

**You've been waiting:** Unknown (how long?)

**If >10 minutes stuck at 0%:**
- Check GPU 0 utilization with `watch -n 1 nvidia-smi`
- If GPU 0 at 0% for >5 min → actually stuck, try Option 2 above
- If GPU 0 at 80-100% → still compiling, be patient

**If >30 minutes:**
- Likely hit the LLVM mma16816 bug again during MCTS compilation
- Try disabling hetero graph entirely:
  ```python
  train.config['hetero_graph'] = False  # in train_hetero.py
  ```

## Expected Behavior After Compilation

Once compilation finishes (could take 20-30 min for hetero):
```
Generating ████████████████ 100% 0:02:15
# Now ALL 8 GPUs active!
# nvidia-smi shows 80-100% on all GPUs
```

Subsequent iterations: 2-5 min (already compiled)

## My Recommendation

**If stuck >15 minutes at 0%:**

1. **Check if still compiling:**
   ```bash
   watch -n 1 nvidia-smi
   # Is GPU 0 still at >50% utilization?
   ```

2. **If yes (still compiling):** Be patient, it's working!

3. **If no (stuck):** Kill and try with reduced simulations:
   ```python
   # train_hetero.py
   train.config['num_simulations'] = 32
   train.config['selfplay_batch_size'] = 64
   ```

4. **If still fails:** Test baseline first:
   ```bash
   python train.py  # Should compile in ~5 min
   ```

## Why This Happens

Heterogeneous graph construction is **100× more complex** than baseline:
- Baseline: 1 edge type, simple indexing
- Hetero: 7 edge types, ray-marching, boolean logic, 4096-element arrays

XLA needs to:
- Optimize all operations
- Generate GPU kernels
- Schedule operations
- Handle dynamic shapes

This is a one-time cost. After compilation, training is fast!

# What to Expect During Training

## Startup Sequence

When you run `./run_multi_gpu.sh`, you'll see:

### 1. Configuration Print (Immediate)
```
================================================================================
HeteroEdgeNet Training Configuration
================================================================================
Heterogeneous graph: True
GNN layers: 4
Inner size: 128
Learning rate: 0.001
Training batch size: 128
Selfplay batch size: 256
Num simulations: 128
Devices: [cuda(id=0), cuda(id=1), ...]
Num devices: 8
================================================================================
```

### 2. Model Initialization (~30 seconds - 2 minutes)
```
🔧 Initializing model...
   Building graph structure...
   Initializing model parameters...
   ✓ Model initialized with 10,234,567 parameters
```

**This step constructs the heterogeneous graph and initializes all 7 edge types.**

### 3. **LONG PAUSE - JIT Compilation (~5-15 minutes)**

```
⏳ First iteration: JIT compiling (this may take 5-15 minutes)...
   Compiling selfplay, MCTS, and graph construction...
```

**THIS IS NORMAL!** The screen will appear frozen. JAX is compiling:
- Heterogeneous graph construction (7 edge types)
- HeteroGATEAU (7 separate attention mechanisms)
- MCTS rollouts
- Gradient computation

**What's happening:**
- JAX traces all functions with example inputs
- Generates optimized XLA code for GPU
- Compiles CUDA kernels
- This only happens ONCE

**Time estimates:**
- With hetero graph: 10-15 minutes
- Without hetero graph: 5-8 minutes
- CPU-only: 15-30 minutes

### 4. First Iteration Completes

After compilation, you'll see progress bars:

```
Generating ████████████████ 100% 0:02:15 < 0:00:00 256 frames
   Compiling training and gradient computation...

Training   ████████████████ 100% 0:00:45 < 0:00:00 loss: 1.245 (0.89 + 0.35)
Evaluating ████████████████ 100% 0:03:10 < 0:00:00 win rate: 0.125 (elo: -450)
```

**Note:** The first training step also compiles (another 2-5 minutes).

### 5. Subsequent Iterations (~2-5 minutes each)

After the first iteration, everything is compiled and runs fast:

```
Iteration 2:
Generating ████████████████ 100% 0:01:20 < 0:00:00 512 frames
Training   ████████████████ 100% 0:00:15 < 0:00:00 loss: 1.123 (0.82 + 0.30)

Iteration 3:
Generating ████████████████ 100% 0:01:18 < 0:00:00 768 frames
Training   ████████████████ 100% 0:00:14 < 0:00:00 loss: 1.089 (0.79 + 0.29)
Evaluating ████████████████ 100% 0:02:55 < 0:00:00 win rate: 0.234 (elo: -280)
```

## Console Output Breakdown

### Progress Bars

**Generating** (Selfplay):
- Runs MCTS to generate 256 games
- ~1-2 min per iteration after compilation
- Shows total frames collected

**Training**:
- Gradient updates on replay buffer
- ~15-30 sec per iteration
- Shows: `loss: total (policy + value)`

**Evaluating** (every 2 iterations):
- Plays 32 games vs baseline
- ~2-3 min
- Shows: win rate and estimated ELO

### No Progress Bars?

If you don't see progress bars, you'll see plain text:
```
[Iteration 1] Selfplay: 256 games, 512 plies, 131072 frames (120.3s)
[Iteration 1] Training: loss=1.234 (policy=0.89, value=0.34) (45.2s)
[Iteration 1] Evaluation: W/D/L = 4/8/20, ELO = -450 (189.5s)
```

## Checking if Training is Stuck

### Normal (working):
```bash
# In another terminal, check GPU usage:
nvidia-smi

# Should show:
# - High GPU utilization (>80%) during selfplay/training
# - Memory allocated (~3-5 GB per GPU for hetero model)
# - Processes: "python train_hetero.py"
```

### Stuck (compilation):
```bash
nvidia-smi

# Shows:
# - Low GPU utilization (0-20%)
# - Memory allocated but not much computation
# - This is NORMAL during JIT compilation (first 5-15 min)
```

### Actually Frozen:
```bash
# Check if process is alive
ps aux | grep train_hetero.py

# Check Python traceback
# Press Ctrl+\ to send SIGQUIT and see where it's stuck
```

## Timeline Summary

```
0:00:00 - Configuration prints
0:00:05 - Model initialization
0:00:30 - Start iteration 1
0:00:35 - JIT compilation begins (LONG PAUSE - screen frozen)
0:10:00 - Selfplay compilation done, generating games
0:12:30 - Training compilation begins
0:15:00 - Iteration 1 complete! 🎉
0:16:30 - Iteration 2 starts (fast now)
0:18:00 - Iteration 2 done
0:19:00 - Iteration 3 starts
0:21:00 - Iteration 3 + evaluation done
... (continues, ~2-3 min per iteration)
```

## If Stuck for >20 Minutes

1. **Check if it's actually running:**
   ```bash
   top -p $(pgrep -f train_hetero.py)
   # Should show ~100-800% CPU during compilation
   ```

2. **Enable compilation logging:**
   Edit `train_hetero.py`, add:
   ```python
   import jax
   jax.config.update('jax_log_compiles', True)
   ```
   Rerun - you'll see each function being compiled.

3. **Try smaller model first:**
   Edit `train_hetero.py`:
   ```python
   train.config['n_gnn_layers'] = 2  # Instead of 4
   train.config['inner_size'] = 64   # Instead of 128
   train.config['selfplay_batch_size'] = 64  # Instead of 256
   ```
   This will compile much faster.

4. **Test without hetero graph:**
   ```bash
   python train.py  # Baseline EdgeNet2, compiles in ~3-5 min
   ```

## Expected Training Time

### Full Training (100 iterations):
- **First iteration:** 15-20 minutes (compilation + execution)
- **Remaining 99:** ~3 min/iter × 99 = 5 hours
- **Total:** ~5-6 hours for 100 iterations

### To Completion (until convergence):
- **500 iterations:** ~25 hours
- **1000 iterations:** ~50 hours

But you'll see measurable improvement after just 20-50 iterations!

## Patience Tips

- **First run:** Go grab coffee/lunch during first iteration
- **Watch GPU:** `watch -n 1 nvidia-smi` to confirm it's working
- **Read papers:** Catch up on GNN literature while compiling
- **Start small:** Test with 5 iterations and small model first

## Success Indicators

✅ **Loss decreasing** (should drop from ~1.2 to <0.8 over 100 iter)
✅ **Win rate increasing** (from ~10% to >40% over 100 iter)
✅ **GPU utilization high** (>80% during selfplay/training)
✅ **Checkpoints being saved** (every eval_interval iterations)

---

**Remember:** The first iteration is SLOW. Be patient! ☕

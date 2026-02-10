# Multi-GPU HeteroEdgeNet Training - Quick Start

## ⚡ 3-Step Setup

### 1. Verify GPU Setup

**First, run the minimal test:**
```bash
python minimal_test.py
# Or if using pixi:
pixi run python minimal_test.py
```

**Then run the full test suite:**
```bash
python test_hetero_graph.py
# Or if using pixi:
pixi run test
```

Expected output:
```
✓ Test 1 passed! (Graph Construction)
✓ Test 2 passed! (Model Forward Pass)
✓ Test 3 passed! (JIT Compilation)
✓ Test 4 passed! (Batched Graph)
✓ Test 5 passed! (Gradient Computation)
✓ Test 6 passed! (Multi-Device)
✅ All tests passed! Ready for training.
```

### 2. Start Training
```bash
# Use all available GPUs
./run_multi_gpu.sh

# Or specify GPUs (e.g., use only GPUs 0-3)
./run_multi_gpu.sh 0,1,2,3
```

### 3. Monitor Progress
```bash
# In another terminal
watch -n 1 nvidia-smi
```

---

## 📊 What to Expect

### First Iteration (~5-10 min)
- JIT compilation happens
- Progress bar shows: `[cyan]Generating[/cyan]`, `[red]Training[/red]`, `[green]Evaluating[/green]`
- GPU memory usage: ~3-5 GB per GPU

### Subsequent Iterations (~2-5 min each)
- Much faster after compilation
- Displays:
  - Loss (policy + value)
  - Win/Draw/Loss rate vs baseline
  - ELO rating estimate

### Training Progress Example
```
Generating ████████████████ 100% 0:01:23 < 0:00:00 256 frames
Training   ████████████████ 100% 0:00:15 < 0:00:00 loss: 0.42 (0.30 + 0.12)
Evaluating ████████████████ 100% 0:02:10 < 0:00:00 win rate: 0.625 (elo: +87)
```

---

## 🎯 Success Criteria

✅ **Training is working if:**
- Loss decreases over iterations
- Win rate vs baseline increases
- No NaN errors
- GPU utilization >80% during selfplay/training

⚠️ **Warning signs:**
- Loss increases or NaN → reduce learning rate
- OOM errors → reduce batch sizes (see troubleshooting)
- Low GPU utilization (<50%) → check batch sizes are divisible by num_devices

---

## 🔧 Quick Troubleshooting

### Out of Memory (OOM)
Edit `train_hetero.py` and reduce batch sizes:
```python
train.config['training_batch_size'] = 2**6   # 64 instead of 128
train.config['selfplay_batch_size'] = 128    # 128 instead of 256
```

### Only 1 GPU Being Used
```bash
# Check detection
python -c "import jax; print(f'{jax.device_count()} GPUs')"

# If 0 or 1, reinstall JAX with GPU support
pip install --upgrade "jax[cuda12]"  # or cuda11
```

### Training Too Slow
Reduce MCTS simulations (faster but less accurate):
```python
train.config['num_simulations'] = 64  # instead of 128
```

---

## 📁 Output Files

After training starts, you'll find:

```
models/chess_2026-02-09:15h30/
├── 000000.ckpt   # Initial model
├── 000001.ckpt   # After iteration 1
├── 000003.ckpt   # After iteration 3 (eval_interval=2)
└── ...

games/chess_2026-02-09:15h30/
├── 000001.pgn    # Evaluation games
├── 000003.pgn
└── ...
```

Load a checkpoint:
```python
from models import load_model
import pgx

env = pgx.make('chess')
model, params = load_model(env, 'models/chess_2026-02-09:15h30/000050.ckpt', 'my_model')
```

---

## 📚 Full Documentation

- **`TRAINING_GUIDE.md`** - Comprehensive guide with all details
- **`README_HETERO_TRAINING.md`** - Architecture overview and comparison
- **`train_hetero.py`** - Training script (modify config here)
- **`run_multi_gpu.sh`** - Shell wrapper for easy execution

---

## 🚀 Advanced Usage

### Custom Configuration
Edit `train_hetero.py`:
```python
# Deeper model
train.config['n_gnn_layers'] = 6
train.config['inner_size'] = 256

# Longer training
train.config['n_iter'] = 500

# Faster evaluation
train.config['eval_interval'] = 5
```

### Profile Performance
```bash
# Add to train_hetero.py before train.main():
import jax
jax.profiler.start_trace("/tmp/tensorboard")
train.main()
jax.profiler.stop_trace()

# View in TensorBoard
tensorboard --logdir=/tmp/tensorboard
```

### Compare to Baseline
```bash
# Train baseline EdgeNet2 (for comparison)
python train.py  # hetero_graph=False by default

# Then compare checkpoints
```

---

## ❓ FAQ

**Q: How long until I see improvement?**
A: Expect measurable gains (win rate >55%) after 20-50 iterations. Significant improvement (>60%) after 100-200 iterations.

**Q: Can I resume training?**
A: Yes, but requires code modification. See `train.py` lines 507-514 for checkpoint loading example.

**Q: How does this compare to AlphaZero?**
A: This architecture has ~10M parameters vs AlphaZero's ~100M, but uses a more structured graph representation that may converge faster.

**Q: Can I train on smaller boards (5×5 Gardner chess)?**
A: Yes! Set `train.config['gardner'] = True` in `train_hetero.py`.

---

## 🆘 Need Help?

1. Check `TRAINING_GUIDE.md` for detailed troubleshooting
2. Run `test_hetero_graph.py` to verify setup
3. Check GPU logs: `nvidia-smi dmon -s pucvmet`
4. Verify JAX installation: `python -c "import jax; print(jax.devices())"`

---

**Happy Training! 🎲♟️🤖**

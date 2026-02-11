# Training Logs and Monitoring

## What Gets Saved

### 1. Training Metrics (CSV) - **Always Enabled**

**File:** `models/chess_YYYY-MM-DD:HHhMM/training_log.csv`

**Columns:**
- `iteration` - Training iteration number
- `loss` - Total loss (policy + value)
- `policy_loss` - Policy (action prediction) loss
- `value_loss` - Value (position evaluation) loss
- `max_grad` - Maximum gradient magnitude
- `win_rate` - Win rate vs baseline (eval only)
- `draw_rate` - Draw rate vs baseline (eval only)
- `lose_rate` - Lose rate vs baseline (eval only)
- `avg_R` - Average reward vs baseline (eval only)
- `elo_rating` - Estimated ELO rating vs baseline (eval only)
- `selfplay_time` - Cumulative selfplay hours
- `train_time` - Cumulative training hours
- `eval_time` - Cumulative evaluation hours

**Updated:**
- Every iteration (training metrics)
- Every `eval_interval` iterations (eval metrics)

### 2. Model Checkpoints

**Files:** `models/chess_YYYY-MM-DD:HHhMM/NNNNNN.ckpt`

**Contains:**
- Model parameters
- Batch statistics
- Optimizer state
- Configuration
- RNG key
- Training metadata

**Saved:** Every `eval_interval` iterations (default: every 2 iterations)

### 3. Game Records (PGN)

**Files:** `games/chess_YYYY-MM-DD:HHhMM/NNNNNN.pgn`

**Contains:**
- Evaluation games vs baseline
- PGN format (standard chess notation)
- Game metadata (players, result, round)

**Saved:** Every `eval_interval` iterations

### 4. Aim Logs (Optional)

**Location:** `aim://localhost:53800` (if Aim server running)

**Currently:** Disabled (`debug=True`)

To enable:
1. Start Aim server: `aim up --port 53800`
2. Edit `train_hetero.py`: `train.config['debug'] = False`

## Monitoring Training

### Real-Time Console Output

While training runs, you see:
```
Generating ████████████████ 100% 0:01:23 < 0:00:00 131072 frames
Training   ████████████████ 100% 0:00:15 < 0:00:00 loss: 0.42 (0.30 + 0.12)
Evaluating ████████████████ 100% 0:02:10 < 0:00:00 win rate: 0.625 (elo: +87)
```

### Watch CSV Log (Real-Time)

```bash
# In another terminal
tail -f models/chess_*/training_log.csv

# Or with column formatting
watch -n 5 'tail -n 20 models/chess_*/training_log.csv | column -t -s,'
```

### Plot Training Progress

**After training starts (once CSV exists):**

```bash
python plot_training.py
# Automatically finds latest log and creates plots
```

**Or specify log file:**
```bash
python plot_training.py models/chess_2026-02-09:23h15/training_log.csv
```

**Output:**
- `models/chess_YYYY-MM-DD:HHhMM/training_progress.png`
- `./training_progress.png` (copy in current directory)

**Plots generated:**
1. Training Loss (total, policy, value) over iterations
2. Win/Draw/Loss rates vs baseline
3. ELO rating over time
4. Gradient magnitude (log scale)

**Re-run plot script anytime** to update with latest data!

### GPU Monitoring

```bash
# Real-time GPU usage
watch -n 1 nvidia-smi

# Or more detailed
nvidia-smi dmon -s pucvmet
```

## Example Workflow

**Terminal 1: Training**
```bash
./run_multi_gpu.sh
```

**Terminal 2: Monitoring**
```bash
# Watch logs
tail -f models/chess_$(ls -t models/ | head -1)/training_log.csv

# Or plot periodically
while true; do
    sleep 60
    python plot_training.py
done
```

**Terminal 3: GPU Monitoring**
```bash
watch -n 1 nvidia-smi
```

## Finding Your Logs

### List all training runs:
```bash
ls -lth models/
```

### Find latest run:
```bash
ls -t models/ | head -1
```

### View latest log:
```bash
cat models/$(ls -t models/ | head -1)/training_log.csv | column -t -s,
```

## Resuming Training

To resume from a checkpoint (requires code modification):

```python
# In train.py, around line 551
pre_train_it = 50  # Iteration to resume from
pre_train_name = "chess_2026-02-09:23h15/000050"
config['continue'] = pre_train_name
with open(f"./models/{pre_train_name}.ckpt", "rb") as f:
    dic = pickle.load(f)
    params, batch_stats = dic['params'], dic['batch_stats']
```

## Data Format Examples

### CSV Format:
```csv
iteration,loss,policy_loss,value_loss,max_grad,win_rate,draw_rate,lose_rate,avg_R,elo_rating,selfplay_time,train_time,eval_time
0,1.234567,0.891234,0.343333,2.456789,,,,,0.0234,0.0156,
1,1.187654,0.854321,0.333333,2.123456,0.125000,0.250000,0.625000,-0.500000,550.00,0.0412,0.0289,0.0534
2,1.145678,0.823456,0.322222,1.987654,,,,,0.0598,0.0423,
3,1.098765,0.789012,0.309753,1.765432,0.218750,0.281250,0.500000,-0.281250,680.00,0.0789,0.0556,0.1123
```

### PGN Format:
```pgn
[Event "Evaluation"]
[Site "GNN-Chess"]
[Date "2026.02.09"]
[Round "1"]
[White "HeteroEdgeNet"]
[Black "Baseline_EdgeNet2"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 ...
```

## Troubleshooting

### Log file not created
- Wait until first iteration completes
- Check `models/` directory exists
- Verify training hasn't crashed

### Plot script fails
```bash
# Install dependencies
pip install pandas matplotlib

# Or with pixi
pixi add pandas matplotlib
```

### Can't find logs
```bash
# Search for all CSV logs
find models/ -name "training_log.csv"

# Search for all checkpoints
find models/ -name "*.ckpt"
```

## Log Analysis Tips

### Check convergence:
```python
import pandas as pd
df = pd.read_csv('models/chess_2026-02-09:23h15/training_log.csv')

# Loss should decrease
print(df[['iteration', 'loss']].tail(10))

# Win rate should increase
print(df[['iteration', 'win_rate']].dropna().tail(10))
```

### Compare runs:
```python
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv('models/chess_run1/training_log.csv')
df2 = pd.read_csv('models/chess_run2/training_log.csv')

plt.plot(df1['iteration'], df1['loss'], label='Run 1')
plt.plot(df2['iteration'], df2['loss'], label='Run 2')
plt.legend()
plt.show()
```

---

**Summary:** Training logs are automatically saved to CSV. Use `plot_training.py` to visualize progress anytime!

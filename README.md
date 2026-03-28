# Sudoku Solver — Training Framework

## Project Structure

```
sudoku_solver/
├── config.py       # All hyperparameters in one dataclass
├── dataset.py      # Puzzle generation & masking
├── model.py        # CNN + multi-head attention solver
├── train.py        # Training loop (resume, logging, LR schedule)
└── inference.py    # Evaluation on fresh puzzles
```

---

## Quick Start

### Install dependencies
```bash
pip install torch tensorboard wandb   # wandb is optional
```

### Train (fresh run)
```bash
python train.py
```

### Resume a run
```bash
python train.py --resume runs/run_<timestamp>/checkpoints/epoch_0050.pt
```

### Common overrides
```bash
python train.py \
  --run-name my_experiment \
  --epochs 300 \
  --batch-size 256 \
  --lr 1e-4 \
  --wandb          # enable W&B (also keeps TensorBoard by default)
  --no-tb          # disable TensorBoard
```

### View TensorBoard
```bash
tensorboard --logdir runs/
```

### Evaluate a checkpoint
```bash
python inference.py --checkpoint runs/my_run/checkpoints/epoch_0200_best.pt --samples 100
```

---

## Key Improvements over Original

| Area | Before | After |
|---|---|---|
| **Speed** | No AMP, no `pin_memory`, no `persistent_workers` | Mixed precision (AMP), `pin_memory`, `persistent_workers`, `set_to_none=True` |
| **Logging** | `print()` only | stdlib `logging` + TensorBoard + optional W&B |
| **Metrics** | Loss + masked accuracy | Loss, masked-cell acc, full-board solve rate, LR |
| **Resuming** | ❌ | Full checkpoint: model, optimizer, scheduler, scaler, best val loss |
| **LR Schedule** | Fixed Adam | AdamW + cosine decay with linear warmup |
| **Validation** | ❌ | 5% hold-out split, evaluated every epoch |
| **Best model** | ❌ | Auto-saved when val loss improves |
| **Config** | Scattered | Single `TrainConfig` dataclass, saved alongside run |
| **Gradient clipping** | ❌ | `clip_grad_norm_` (default 1.0) |

---

## Configuration

Edit `config.py` or override via CLI. Key defaults:

```python
num_samples   = 50_000   # dataset size per epoch
batch_size    = 128
lr            = 3e-4
num_epochs    = 200
channels      = 256
num_res_blocks= 8
warmup_epochs = 5        # linear LR warmup
save_every    = 5        # checkpoint every N epochs
```
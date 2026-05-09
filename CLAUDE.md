# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install torch numba tensorboard pyyaml wandb   # wandb is optional

# Train with defaults (reads config.yaml)
python train.py

# Train with a custom config file
python train.py --config my_config.yaml

# Common CLI overrides
python train.py --run-name my_exp --epochs 300 --batch-size 256 --lr 1e-4 --wandb

# Resume from a checkpoint (weights only)
python train.py --resume runs/my_run/best.pt

# Evaluate a checkpoint
python inference.py --checkpoint runs/my_run/best.pt --num_puzzles 100 --num_clues 18

# Monitor training
tensorboard --logdir runs/
```

## Architecture

The pipeline is: puzzle tokens → embedding + structured positional encoding → CNN front-end → Transformer blocks → per-cell digit logits.

**Data encoding** (consistent throughout): digit values are 0-indexed (`0–8` = digits `1–9`, `9` = masked/unknown cell). Targets are also 0-indexed (classes `0–8`). The `+1`/`-1` conversion happens only at display time in `inference.py`.

**Model** ([model.py](model.py)): `SudokuSolver` takes `(B, 9, 9)` long tensor and returns `(B, 9, 9, 9)` logits with the **class dimension at dim=1** (not dim=-1), compatible with `nn.CrossEntropyLoss`. The forward pass:
1. Embeds tokens + adds structured positional encodings (separate row/col/box learnable embeddings concatenated and projected)
2. Projects through a lightweight CNN residual front-end to build per-cell spatial features
3. Flattens to sequence `(B, 81, C)` and passes through `N` `SudokuTransformerBlock`s — each block does global self-attention, then three separate `AxisAttention` passes masked to row/col/box peers respectively
4. Projects each cell's representation to 9 logits via the output head

The row/col/box attention masks are pre-computed as buffers (`register_buffer`) so they move with `.to(device)` automatically.

**Dataset** ([dataset.py](dataset.py)): Puzzles are generated fully on-the-fly in `__getitem__`. `Sudoku.generate_solved_board()` seeds the 3 diagonal 3×3 boxes randomly, then completes via a Numba-JIT MRV backtracking solver (`_solve_mrv`). Masking randomly removes 1–64 cells. **Numba JIT compiles on first run** — expect a ~30s warmup delay on the first epoch.

**Config** ([config.py](config.py), [config.yaml](config.yaml)): `TrainConfig` is a flat dataclass. The YAML file uses nested sections (`data:`, `model:`, `training:`, etc.) which `TrainConfig.from_yaml()` flattens. CLI args override specific fields via `cfg.merge_args(args)`. The active config is saved to `runs/<run_name>/config.yaml` at the start of each run.

**Trainer** ([trainer.py](trainer.py)): Manages the full train/val loop. Checkpoints save **weights only** (`model.state_dict()`), not optimizer/scheduler state. `best.pt` is written whenever val loss improves; `last.pt` is overwritten every epoch. The model architecture in `_build_model` is currently **hardcoded** (the config-driven instantiation is commented out at lines 79–85).

**Inference** ([inference.py](inference.py)): Two solve modes — `solve()` (single forward pass, fill all masked cells at once) and `solve_iterative()` (fix the single most-confident masked cell per step, re-run until complete). Iterative mode generally achieves higher accuracy on hard puzzles.

**Logging** ([logger.py](logger.py)): `MetricLogger` is a thin W&B wrapper. W&B is disabled unless `use_wandb: true` in config or `--wandb` CLI flag is passed. TensorBoard is not wired — logged metrics only go to W&B (and stdout via stdlib `logging`).

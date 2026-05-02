"""
train.py — Entry point for training the Sudoku solver.

Usage
-----
# fresh run with defaults
python train.py

# custom config file
python train.py --config my_config.yaml

# resume from checkpoint (weights only)
python train.py --resume runs/my_run/best.pt

# CLI overrides (applied on top of the config file)
python train.py --config my_config.yaml --epochs 50 --lr 5e-4 --wandb
"""

import argparse
import logging

from config import TrainConfig
from trainer import Trainer

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Sudoku Solver")
    p.add_argument(
        "--config",
        metavar="PATH",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    p.add_argument(
        "--resume", metavar="PATH", help="Resume from checkpoint (weights only)"
    )
    p.add_argument(
        "--run-name", dest="run_name", metavar="NAME", help="Name for this run"
    )
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    p.add_argument("--epochs", type=int, help="Override num_epochs")
    p.add_argument(
        "--batch-size", dest="batch_size", type=int, help="Override batch_size"
    )
    p.add_argument("--lr", type=float, help="Override learning rate")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig.from_yaml(args.config)
    cfg.merge_args(args)

    Trainer(cfg).train()

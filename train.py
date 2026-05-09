"""
train.py — Entry point for training the Sudoku solver.

Usage
-----
# Auto-select device (GPU 0 if available, else CPU)
python train.py

# Pin a single GPU
python train.py --gpus 1

# Multi-GPU DDP on GPUs 0, 1, 2
python train.py --gpus 0,1,2

# With other overrides
python train.py --gpus 0,1 --epochs 300 --lr 5e-4 --wandb

# Resume from checkpoint
python train.py --gpus 0,1 --resume runs/my_run/best.pt
"""

import argparse
import logging
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from config import TrainConfig
from trainer import Trainer

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _ddp_worker(rank: int, cfg: TrainConfig, gpus: list) -> None:
    torch.cuda.set_device(gpus[rank])
    dist.init_process_group("nccl", rank=rank, world_size=len(gpus))
    if rank != 0:
        logging.disable(logging.CRITICAL)
    try:
        Trainer(cfg, rank=rank, world_size=len(gpus), gpus=gpus).train()
    finally:
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Sudoku Solver")
    p.add_argument(
        "--config",
        metavar="PATH",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    p.add_argument(
        "--gpus",
        metavar="IDS",
        default=None,
        help="Comma-separated GPU indices, e.g. '0' or '0,1,2'. Omit to auto-select.",
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

    gpus = None
    if args.gpus is not None:
        requested = [int(g.strip()) for g in args.gpus.split(",")]
        # Set CUDA_VISIBLE_DEVICES before any CUDA API call so the runtime never
        # creates a primary context on GPU 0 when we aren't using it.
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in requested)
        # CUDA remaps the visible GPUs to 0-indexed, so [1,2,3] becomes [0,1,2].
        gpus = list(range(len(requested)))
        available = torch.cuda.device_count()
        if available < len(gpus):
            raise SystemExit(
                f"{len(gpus)} GPU(s) requested but only {available} visible "
                f"(CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})."
            )

    if gpus and len(gpus) > 1:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", str(_find_free_port()))
        mp.spawn(_ddp_worker, args=(cfg, gpus), nprocs=len(gpus), join=True)
    else:
        if gpus:
            torch.cuda.set_device(gpus[0])
        Trainer(cfg, gpus=gpus).train()

"""
train.py — Sudoku solver training script.

Usage
-----
# fresh run with tensorboard
python train.py

# resume from checkpoint
python train.py --resume runs/my_run/checkpoints/epoch_010.pt

# with W&B
python train.py --wandb --run-name experiment_1
"""

import argparse
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split

from config import TrainConfig
from dataset import SudokuDataset
from model import SudokuSolver

# ── stdlib logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def masked_accuracy(
    logits: torch.Tensor,  # (B, 9, 9, 9)
    target: torch.Tensor,  # (B, 9, 9)
    mask: torch.Tensor,  # (B, 9, 9) bool
) -> Tuple[float, float]:
    """Returns (masked_acc, full_board_acc)."""
    pred = logits.argmax(dim=1)  # (B, 9, 9)
    correct = pred == target  # (B, 9, 9)

    masked_acc = (correct & mask).sum().item() / mask.sum().clamp(min=1).item()

    # Full-board accuracy: fraction of boards solved perfectly
    board_correct = correct.view(correct.shape[0], -1).all(dim=1)
    full_acc = board_correct.float().mean().item()

    return masked_acc, full_acc


def get_lr(optimizer: optim.Optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


# ──────────────────────────────────────────────────────────────────────────────
# Logger wrapper (TensorBoard / W&B / both)
# ──────────────────────────────────────────────────────────────────────────────


class MetricLogger:
    def __init__(self, cfg: TrainConfig, run_dir: Path):
        self.writers = []

        if cfg.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_dir = run_dir / "tensorboard"
                tb_dir.mkdir(parents=True, exist_ok=True)
                self.tb = SummaryWriter(log_dir=str(tb_dir))
                self.writers.append("tensorboard")
                log.info("TensorBoard logging → %s", tb_dir)
            except ImportError:
                log.warning("tensorboard not installed — skipping TB logging")
                self.tb = None
        else:
            self.tb = None

        if cfg.use_wandb:
            try:
                import wandb

                wandb.init(
                    project="sudoku-solver",
                    name=cfg.run_name,
                    config=vars(cfg),
                    dir=str(run_dir),
                    resume="allow",
                )
                self.wandb = wandb
                self.writers.append("wandb")
                log.info("W&B run: %s", wandb.run.url)
            except ImportError:
                log.warning("wandb not installed — skipping W&B logging")
                self.wandb = None
        else:
            self.wandb = None

    def log(self, metrics: Dict[str, float], step: int) -> None:
        if self.tb:
            for k, v in metrics.items():
                self.tb.add_scalar(k, v, step)
        if self.wandb:
            self.wandb.log(metrics, step=step)

    def close(self) -> None:
        if self.tb:
            self.tb.close()
        if self.wandb:
            self.wandb.finish()


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Device: %s", self.device)

        # ── Run directory ──────────────────────────────────────────────────────
        run_name = cfg.run_name or f"run_{int(time.time())}"
        self.run_dir = Path(cfg.output_dir) / run_name
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        cfg.save(str(self.run_dir / "config.json"))

        # ── Data ──────────────────────────────────────────────────────────────
        full_ds = SudokuDataset(cfg.num_samples, cfg.min_mask, cfg.max_mask)
        val_size = max(1, int(0.05 * cfg.num_samples))
        train_ds, val_ds = random_split(full_ds, [cfg.num_samples - val_size, val_size])

        self.train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=cfg.num_workers > 0,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size * 2,
            num_workers=cfg.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=cfg.num_workers > 0,
        )

        # ── Model ─────────────────────────────────────────────────────────────
        self.model = SudokuSolver(
            embed_dim=cfg.embed_dim,
            channels=cfg.channels,
            num_res_blocks=cfg.num_res_blocks,
            num_heads=cfg.num_heads,
            dropout_rate=cfg.dropout_rate,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log.info("Model parameters: %s", f"{n_params:,}")

        # ── Optimizer & scheduler ─────────────────────────────────────────────
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        total_steps = cfg.num_epochs * len(self.train_loader)
        warmup_steps = cfg.warmup_epochs * len(self.train_loader)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            min_ratio = cfg.lr_min / cfg.lr
            return max(min_ratio, cosine)

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.criterion = nn.CrossEntropyLoss(reduction="none")

        # ── AMP (mixed precision) ─────────────────────────────────────────────
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # ── Logger ────────────────────────────────────────────────────────────
        self.logger = MetricLogger(cfg, self.run_dir)

        # ── State ─────────────────────────────────────────────────────────────
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        if cfg.resume_from:
            self._load_checkpoint(cfg.resume_from)

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, tag: str = "") -> Path:
        name = f"epoch_{epoch:04d}{('_' + tag) if tag else ''}.pt"
        path = self.ckpt_dir / name
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "best_val_loss": self.best_val_loss,
                "cfg": vars(self.cfg),
            },
            path,
        )
        log.info("Checkpoint saved → %s", path)
        return path

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.start_epoch = ckpt["epoch"] + 1
        self.global_step = ckpt["global_step"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        log.info(
            "Resumed from %s  (epoch %d, step %d)",
            path,
            ckpt["epoch"],
            self.global_step,
        )

    # ── Training loop ─────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = total_masked_acc = total_full_acc = 0.0
        t0 = time.perf_counter()

        for step, (inp, target, mask) in enumerate(self.train_loader):
            inp = inp.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(inp)  # (B, 9, 9, 9)
                loss_map = self.criterion(logits, target)  # (B, 9, 9)
                loss = loss_map[mask].mean() if mask.any() else loss_map.mean()

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            m_acc, f_acc = masked_accuracy(logits.detach(), target, mask)
            total_loss += loss.item()
            total_masked_acc += m_acc
            total_full_acc += f_acc
            self.global_step += 1

            if self.global_step % self.cfg.log_every == 0:
                self.logger.log(
                    {
                        "train/loss": loss.item(),
                        "train/masked_acc": m_acc,
                        "train/full_board_acc": f_acc,
                        "train/lr": get_lr(self.optimizer),
                    },
                    self.global_step,
                )

        n = len(self.train_loader)
        elapsed = time.perf_counter() - t0
        return {
            "loss": total_loss / n,
            "masked_acc": total_masked_acc / n,
            "full_acc": total_full_acc / n,
            "epoch_time": elapsed,
        }

    @torch.no_grad()
    def _val_epoch(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = total_masked_acc = total_full_acc = 0.0

        for inp, target, mask in self.val_loader:
            inp = inp.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(inp)
                loss_map = self.criterion(logits, target)
                loss = loss_map[mask].mean() if mask.any() else loss_map.mean()

            m_acc, f_acc = masked_accuracy(logits, target, mask)
            total_loss += loss.item()
            total_masked_acc += m_acc
            total_full_acc += f_acc

        n = len(self.val_loader)
        return {
            "loss": total_loss / n,
            "masked_acc": total_masked_acc / n,
            "full_acc": total_full_acc / n,
        }

    def train(self) -> None:
        log.info(
            "Starting training — epochs %d→%d, %d steps/epoch",
            self.start_epoch + 1,
            self.cfg.num_epochs,
            len(self.train_loader),
        )

        for epoch in range(self.start_epoch, self.cfg.num_epochs):
            train_stats = self._train_epoch(epoch)

            log.info(
                "Epoch %4d/%d | loss %.4f | mask_acc %.4f | full_acc %.4f | "
                "lr %.2e | %.1fs",
                epoch + 1,
                self.cfg.num_epochs,
                train_stats["loss"],
                train_stats["masked_acc"],
                train_stats["full_acc"],
                get_lr(self.optimizer),
                train_stats["epoch_time"],
            )

            # ── Validation ────────────────────────────────────────────────────
            if (epoch + 1) % self.cfg.eval_every == 0:
                val_stats = self._val_epoch()
                log.info(
                    "  Val        | loss %.4f | mask_acc %.4f | full_acc %.4f",
                    val_stats["loss"],
                    val_stats["masked_acc"],
                    val_stats["full_acc"],
                )
                self.logger.log(
                    {
                        "val/loss": val_stats["loss"],
                        "val/masked_acc": val_stats["masked_acc"],
                        "val/full_board_acc": val_stats["full_acc"],
                        "epoch": epoch + 1,
                    },
                    self.global_step,
                )

                if val_stats["loss"] < self.best_val_loss:
                    self.best_val_loss = val_stats["loss"]
                    self._save_checkpoint(epoch, tag="best")
                    log.info("  ↳ New best val loss: %.4f", self.best_val_loss)

            # ── Periodic checkpoint ───────────────────────────────────────────
            if (epoch + 1) % self.cfg.save_every == 0:
                self._save_checkpoint(epoch)

        # Always save at the end
        self._save_checkpoint(self.cfg.num_epochs - 1, tag="final")
        self.logger.close()
        log.info("Training complete. Best val loss: %.4f", self.best_val_loss)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Sudoku Solver")
    p.add_argument("--resume", metavar="PATH", help="Resume from checkpoint")
    p.add_argument("--run-name", metavar="NAME", help="Name for this run")
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    p.add_argument("--no-tb", action="store_true", help="Disable TensorBoard")
    p.add_argument("--epochs", type=int, help="Override num_epochs")
    p.add_argument("--batch-size", type=int, help="Override batch_size")
    p.add_argument("--lr", type=float, help="Override learning rate")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = TrainConfig()

    # CLI overrides
    if args.resume:
        cfg.resume_from = args.resume
    if args.run_name:
        cfg.run_name = args.run_name
    if args.wandb:
        cfg.use_wandb = True
    if args.no_tb:
        cfg.use_tensorboard = False
    if args.epochs:
        cfg.num_epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.lr = args.lr

    trainer = Trainer(cfg)
    trainer.train()

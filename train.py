"""
train.py — Sudoku solver training script.

Usage
-----
# fresh run
python train.py

# resume from checkpoint (weights only)
python train.py --resume runs/my_run/best.pt

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

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def masked_accuracy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[float, float]:
    pred = logits.argmax(dim=1)
    correct = pred == target
    masked_acc = (correct & mask).sum().item() / mask.sum().clamp(min=1).item()
    board_correct = correct.view(correct.shape[0], -1).all(dim=1)
    full_acc = board_correct.float().mean().item()
    return masked_acc, full_acc


def get_lr(optimizer: optim.Optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


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


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Device: %s", self.device)

        run_name = cfg.run_name or f"run_{int(time.time())}"
        self.run_dir = Path(cfg.output_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
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

        # Load weights if resuming (weights only — training state resets)
        if cfg.resume_from:
            self._load_checkpoint(cfg.resume_from)

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

        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.logger = MetricLogger(cfg, self.run_dir)

        self.global_step = 0
        self.best_val_loss = float("inf")

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def _save_checkpoint(self, filename: str) -> None:
        """Save model weights only to run_dir/<filename>."""
        path = self.run_dir / filename
        torch.save(self.model.state_dict(), path)
        log.info("Checkpoint saved → %s", path)

    def _load_checkpoint(self, path: str) -> None:
        """Load model weights only. Optimizer/scheduler start fresh."""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        log.info("Weights loaded from %s", path)

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
                logits = self.model(inp)
                loss_map = self.criterion(logits, target)
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
            "Starting training — %d epochs, %d steps/epoch",
            self.cfg.num_epochs,
            len(self.train_loader),
        )

        for epoch in range(self.cfg.num_epochs):
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
                    self._save_checkpoint("best.pt")
                    log.info("  ↳ New best val loss: %.4f", self.best_val_loss)

            # Save last.pt at the end of each iteration
            self._save_checkpoint("last.pt")

        self.logger.close()
        log.info("Training complete. Best val loss: %.4f", self.best_val_loss)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Sudoku Solver")
    p.add_argument(
        "--resume", metavar="PATH", help="Resume from checkpoint (weights only)"
    )
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

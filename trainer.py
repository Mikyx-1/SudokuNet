"""
trainer.py — Trainer: data loading, optimiser, train/val loops, checkpointing.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split

from config import TrainConfig
from dataset import SudokuDataset
from logger import MetricLogger
from metrics import masked_accuracy
from model import SudokuSolver

log = logging.getLogger(__name__)

# Since Sudoku is fixed-size (9x9), we can benefit
# from cuDNNoptimisations by enabling benchmark mode.
torch.backends.cudnn.benchmark = True


def _get_lr(optimizer: optim.Optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


class Trainer:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Device: %s", self.device)

        self.run_dir = Path(cfg.output_dir) / cfg.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        cfg.save(self.run_dir / "config.yaml")

        self._build_dataloaders()
        self._build_model()
        self._build_optimiser()

        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.logger = MetricLogger(cfg, self.run_dir)

        self.global_step = 0
        self.best_val_loss = float("inf")

    # ── Setup helpers ─────────────────────────────────────────────────────────

    def _build_dataloaders(self) -> None:
        cfg = self.cfg
        full_ds = SudokuDataset(cfg.num_samples, cfg.min_mask, cfg.max_mask)
        val_size = max(1, int(0.05 * cfg.num_samples))
        train_ds, val_ds = random_split(full_ds, [cfg.num_samples - val_size, val_size])

        loader_kwargs = dict(
            num_workers=cfg.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=cfg.num_workers > 0,
        )
        self.train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, **loader_kwargs
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size * 2, **loader_kwargs
        )

    def _build_model(self) -> None:
        cfg = self.cfg
        # self.model = SudokuSolver(
        #     embed_dim=cfg.embed_dim,
        #     channels=cfg.channels,
        #     num_res_blocks=cfg.num_res_blocks,
        #     num_heads=cfg.num_heads,
        #     dropout_rate=cfg.dropout_rate,
        # ).to(self.device)

        self.model = SudokuSolver(
            embed_dim=64,
            channels=256,
            num_res_blocks=4,
            num_transformer_blocks=8,
            num_heads=8,
            dropout_rate=0.1,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log.info("Model parameters: %s", f"{n_params:,}")

        if cfg.resume_from:
            self._load_checkpoint(cfg.resume_from)

    def _build_optimiser(self) -> None:
        cfg = self.cfg
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            fused=True,
        )

        total_steps = cfg.num_epochs * len(self.train_loader)
        warmup_steps = cfg.warmup_epochs * len(self.train_loader)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return max(cfg.lr_min / cfg.lr, cosine)

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def _save_checkpoint(self, filename: str) -> None:
        path = self.run_dir / filename
        torch.save(self.model.state_dict(), path)
        log.info("Checkpoint saved → %s", path)

    def _load_checkpoint(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        log.info("Weights loaded from %s", path)

    # ── Train / val epochs ────────────────────────────────────────────────────
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = total_masked_acc = total_full_acc = 0.0
        skipped = 0
        t0 = time.perf_counter()

        for inp, target, mask in self.train_loader:
            inp = inp.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(inp)
                loss_map = self.criterion(logits, target)
                loss = loss_map[mask].mean() if mask.any() else loss_map.mean()

            # ── Guard 1: catch NaN/Inf loss before backward poisons gradients ──
            if not torch.isfinite(loss):
                log.warning(
                    "Non-finite loss at step %d — skipping batch", self.global_step
                )
                self.global_step += 1
                self.scheduler.step()
                skipped += 1
                continue

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            # ── Guard 2: catch exploded gradients before they corrupt weights ──
            total_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip
            )
            if not torch.isfinite(total_norm):
                log.warning(
                    "Non-finite grad norm (%.2e) at step %d — skipping update",
                    total_norm,
                    self.global_step,
                )
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                self.global_step += 1
                self.scheduler.step()
                skipped += 1
                continue

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
                        "train/lr": _get_lr(self.optimizer),
                    },
                    self.global_step,
                )

        n = len(self.train_loader)
        good = n - skipped
        if skipped:
            log.warning(
                "Epoch %d: skipped %d/%d batches due to NaN", epoch + 1, skipped, n
            )
        return {
            "loss": total_loss / max(good, 1),
            "masked_acc": total_masked_acc / max(good, 1),
            "full_acc": total_full_acc / max(good, 1),
            "epoch_time": time.perf_counter() - t0,
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

    # ── Main loop ─────────────────────────────────────────────────────────────

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
                _get_lr(self.optimizer),
                train_stats["epoch_time"],
            )

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

            self._save_checkpoint("last.pt")

        self.logger.close()
        log.info("Training complete. Best val loss: %.4f", self.best_val_loss)

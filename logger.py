"""
logger.py — W&B metric logger.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict

from config import TrainConfig

log = logging.getLogger(__name__)


class MetricLogger:
    """Thin wrapper around W&B (optional)."""

    def __init__(self, cfg: TrainConfig, run_dir: Path) -> None:
        self.wandb = None
        if not cfg.use_wandb:
            return
        try:
            import wandb

            wandb.init(
                project=cfg.project,
                group=cfg.group,
                name=cfg.run_name,
                config=asdict(cfg),
                dir=str(run_dir),
                resume="allow",
            )
            self.wandb = wandb
            log.info("W&B run: %s", wandb.run.url)
        except ImportError:
            log.warning("wandb not installed — skipping W&B logging")

    def log(self, metrics: Dict[str, float], step: int) -> None:
        if self.wandb:
            self.wandb.log(metrics, step=step)

    def close(self) -> None:
        if self.wandb:
            self.wandb.finish()

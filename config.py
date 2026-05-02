"""
config.py — Training configuration loaded from a YAML file.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class TrainConfig:
    # ── Data ──────────────────────────────────────────────────────────────────
    num_samples: int = 50_000
    min_mask: int = 1
    max_mask: int = 64
    num_workers: int = 16

    # ── Model ─────────────────────────────────────────────────────────────────
    embed_dim: int = 64
    channels: int = 256
    num_res_blocks: int = 8
    num_heads: int = 8
    dropout_rate: float = 0.1

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_epochs: int = 200
    grad_clip: float = 1.0

    # ── LR Schedule ───────────────────────────────────────────────────────────
    warmup_epochs: int = 5
    lr_min: float = 1e-6

    # ── Logging / Checkpointing ────────────────────────────────────────────────
    log_every: int = 10
    eval_every: int = 1
    save_every: int = 5
    run_name: Optional[str] = None
    output_dir: str = "runs"
    use_wandb: bool = False

    # ── Resume ────────────────────────────────────────────────────────────────
    resume_from: Optional[str] = None

    # ── Derived ───────────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        if self.run_name is None:
            self.run_name = f"run_{int(time.time())}"

    # ── Serialisation ─────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        """Load config from a nested YAML file, ignoring unknown keys."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        # Flatten nested sections (e.g. data.num_samples → num_samples)
        flat: dict = {}
        for value in data.values():
            if isinstance(value, dict):
                flat.update(value)
            else:
                flat[value] = value  # top-level scalar, unlikely but safe
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in flat.items() if k in valid_fields}
        return cls(**filtered)

    def save(self, path: str | Path) -> None:
        """Persist current config snapshot to YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def merge_args(self, args) -> None:
        """Override fields from parsed CLI args (only non-None / truthy values)."""
        mapping = {
            "resume": "resume_from",
            "run_name": "run_name",
            "epochs": "num_epochs",
            "batch_size": "batch_size",
            "lr": "lr",
        }
        for arg_attr, cfg_attr in mapping.items():
            val = getattr(args, arg_attr, None)
            if val is not None:
                setattr(self, cfg_attr, val)
        # store_true flags: False means "not passed", not "set to False"
        if getattr(args, "wandb", False):
            self.use_wandb = True

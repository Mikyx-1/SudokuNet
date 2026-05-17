"""
config.py — Training configuration loaded from a YAML file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
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

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_epochs: int = 200
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    use_amp: bool = False

    # ── Curriculum ────────────────────────────────────────────────────────────
    curriculum: bool = True
    curriculum_start_mask: int = 20
    curriculum_ramp_frac: float = 0.4

    # ── LR Schedule ───────────────────────────────────────────────────────────
    warmup_epochs: int = 5
    lr_min: float = 1e-6

    # ── Logging ───────────────────────────────────────────────────────────────
    log_every: int = 10
    eval_every: int = 1
    save_every: int = 5
    output_dir: str = "runs"

    # ── W&B ───────────────────────────────────────────────────────────────────
    use_wandb: bool = False
    project: str = "sudoku-solver"
    group: str = "sudoku_cnn"
    run_name: Optional[str] = None  # auto-generated if null

    # ── Resume ────────────────────────────────────────────────────────────────
    resume_from: Optional[str] = None

    # ── Derived ───────────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        if self.run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"{self.group}_{timestamp}"

    # ── Serialisation ─────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        """Load config from a nested YAML file, ignoring unknown keys."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        # Flatten nested sections (e.g. data.num_samples -> num_samples)
        flat: dict = {}
        for value in data.values():
            if isinstance(value, dict):
                flat.update(value)
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
        if getattr(args, "wandb", False):
            self.use_wandb = True

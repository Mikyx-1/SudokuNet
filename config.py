import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    # ── Data ──────────────────────────────────────────────────────────────────
    num_samples: int = 50_000
    min_mask: int = 1
    max_mask: int = 64
    num_workers: int = 4

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
    log_every: int = 10  # steps between scalar logs
    eval_every: int = 1  # epochs between evaluations
    save_every: int = 5  # epochs between checkpoints
    run_name: Optional[str] = None
    output_dir: str = "runs"
    use_wandb: bool = False
    use_tensorboard: bool = True

    # ── Resume ────────────────────────────────────────────────────────────────
    resume_from: Optional[str] = None  # path to a checkpoint .pt file

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            return cls(**json.load(f))

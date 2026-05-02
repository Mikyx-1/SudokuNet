"""
metrics.py — Accuracy metric helpers.
"""

from __future__ import annotations

from typing import Tuple

import torch


def masked_accuracy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[float, float]:
    """
    Returns
    -------
    masked_acc : fraction of *masked* cells predicted correctly
    full_acc   : fraction of boards where *every* cell is correct
    """
    pred = logits.argmax(dim=1)
    correct = pred == target
    masked_acc = (correct & mask).sum().item() / mask.sum().clamp(min=1).item()
    board_correct = correct.view(correct.shape[0], -1).all(dim=1)
    full_acc = board_correct.float().mean().item()
    return masked_acc, full_acc

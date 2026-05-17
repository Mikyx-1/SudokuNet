"""
evaluate_accuracy.py — Sweep a trained SudokuSolver over a range of clue counts
and report single-pass and iterative accuracy.

Outputs:
  - results/accuracy_vs_clues.csv
  - results/accuracy_vs_clues.png

The accuracy metric matches inference.py: a solved board is "correct" iff every
row, column, and 3x3 box contains the digits 1-9 exactly once. For high clue
counts the puzzle's solution is essentially unique, so this coincides with
matching the ground-truth solution.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import Sudoku
from model import SudokuSolver

torch.backends.cudnn.benchmark = True

MASK_TOKEN = 9  # consistent with dataset.py / inference.py


def load_model(path: str, device: torch.device) -> SudokuSolver:
    model = SudokuSolver(
        embed_dim=64,
        channels=256,
        num_res_blocks=2,
        num_transformer_blocks=4,
        num_heads=4,
    ).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def generate_batch(
    num_clues: int, batch_size: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (puzzles, solutions) — both (B, 9, 9) int8, 0-indexed, mask=9."""
    n_mask = 81 - num_clues
    puzzles = np.empty((batch_size, 9, 9), dtype=np.int64)
    solutions = np.empty((batch_size, 9, 9), dtype=np.int64)
    for i in range(batch_size):
        solved = Sudoku.generate_solved_board() - 1  # 0-indexed
        puzzle = solved.copy()
        idx = rng.permutation(81)[:n_mask]
        puzzle.ravel()[idx] = MASK_TOKEN
        puzzles[i] = puzzle
        solutions[i] = solved
    return puzzles, solutions


def batch_is_valid(boards: torch.Tensor) -> torch.Tensor:
    """
    boards: (B, 9, 9) long tensor of values 0-8 (no masks remaining).
    Returns: (B,) bool tensor — True iff every row/col/box is a permutation of 0-8.
    """
    B = boards.size(0)
    target = torch.arange(9, device=boards.device).expand(B, 9, 9)

    # No mask tokens should remain. Anything else (e.g. residual 9) → invalid.
    no_mask = (boards != MASK_TOKEN).view(B, -1).all(dim=1)

    rows_sorted = boards.sort(dim=2).values
    rows_valid = (rows_sorted == target).all(dim=2).all(dim=1)

    cols_sorted = boards.transpose(1, 2).sort(dim=2).values
    cols_valid = (cols_sorted == target).all(dim=2).all(dim=1)

    # Re-pack into 9 boxes of 9 cells: (B, 3, 3, 3, 3) -> permute -> (B, 9, 9).
    boxes = boards.view(B, 3, 3, 3, 3).permute(0, 1, 3, 2, 4).reshape(B, 9, 9)
    boxes_sorted = boxes.sort(dim=2).values
    boxes_valid = (boxes_sorted == target).all(dim=2).all(dim=1)

    return no_mask & rows_valid & cols_valid & boxes_valid


@torch.no_grad()
def solve_batch_single(model: SudokuSolver, puzzles: torch.Tensor) -> torch.Tensor:
    """One forward pass; fill all masked cells from argmax. Clues are kept."""
    logits = model(puzzles)  # (B, 9, 9, 9), classes at dim=1
    pred = logits.argmax(dim=1)  # (B, 9, 9)
    masked = puzzles == MASK_TOKEN
    return torch.where(masked, pred, puzzles)


@torch.no_grad()
def solve_batch_iterative(
    model: SudokuSolver, puzzles: torch.Tensor, max_steps: int = 81
) -> torch.Tensor:
    """
    Each step, fix the single most-confident masked cell per puzzle, then re-run.
    Vectorised across the batch: one forward pass per step, regardless of B.
    """
    board = puzzles.clone()
    B = board.size(0)
    batch_idx = torch.arange(B, device=board.device)
    neg_inf = torch.tensor(-1.0, device=board.device)

    for _ in range(max_steps):
        masked = board == MASK_TOKEN  # (B, 9, 9)
        any_masked = masked.view(B, -1).any(dim=1)  # (B,)
        if not any_masked.any():
            break

        logits = model(board)  # (B, 9, 9, 9)
        probs = torch.softmax(logits, dim=1)  # (B, 9, 9, 9)
        max_probs, max_digits = probs.max(dim=1)  # (B, 9, 9)

        # Only consider masked cells; non-masked get -inf so argmax ignores them.
        confidence = torch.where(masked, max_probs, neg_inf)
        flat = confidence.view(B, -1)
        best_idx = flat.argmax(dim=1)  # (B,)
        r = best_idx // 9
        c = best_idx % 9

        # Update only puzzles that still have masked cells.
        active = any_masked
        digits = max_digits[batch_idx, r, c]
        board[batch_idx[active], r[active], c[active]] = digits[active]

    return board


def run_sweep(
    model: SudokuSolver,
    device: torch.device,
    clue_counts: List[int],
    num_puzzles: int,
    batch_size: int,
    seed: int,
) -> List[dict]:
    rng = np.random.default_rng(seed)
    results = []

    # Warmup: first forward pays cuDNN/kernel init and Numba JIT costs.
    warmup_puzzles, _ = generate_batch(40, min(8, batch_size), rng)
    warm = torch.from_numpy(warmup_puzzles).long().to(device)
    solve_batch_single(model, warm)
    solve_batch_iterative(model, warm, max_steps=10)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    for num_clues in clue_counts:
        correct_single = 0
        correct_iter = 0
        t_single = 0.0
        t_iter = 0.0
        n_done = 0

        while n_done < num_puzzles:
            bs = min(batch_size, num_puzzles - n_done)
            puzzles_np, _ = generate_batch(num_clues, bs, rng)
            puzzles = torch.from_numpy(puzzles_np).long().to(device)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            out_single = solve_batch_single(model, puzzles)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_single += time.perf_counter() - t0

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            out_iter = solve_batch_iterative(model, puzzles)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_iter += time.perf_counter() - t0

            correct_single += int(batch_is_valid(out_single).sum().item())
            correct_iter += int(batch_is_valid(out_iter).sum().item())
            n_done += bs

        acc_single = correct_single / num_puzzles
        acc_iter = correct_iter / num_puzzles
        ms_single = t_single / num_puzzles * 1000
        ms_iter = t_iter / num_puzzles * 1000

        print(
            f"clues={num_clues:>2d} | "
            f"single {correct_single:>4d}/{num_puzzles} = {acc_single:.3f} "
            f"({ms_single:6.2f} ms) | "
            f"iter   {correct_iter:>4d}/{num_puzzles} = {acc_iter:.3f} "
            f"({ms_iter:6.2f} ms)"
        )

        results.append(
            {
                "num_clues": num_clues,
                "acc_single": acc_single,
                "acc_iter": acc_iter,
                "correct_single": correct_single,
                "correct_iter": correct_iter,
                "ms_per_puzzle_single": ms_single,
                "ms_per_puzzle_iter": ms_iter,
            }
        )

    return results


def save_csv(results: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def save_plot(results: List[dict], path: Path, checkpoint: str, n: int) -> None:
    clues = [r["num_clues"] for r in results]
    single = [r["acc_single"] * 100 for r in results]
    it = [r["acc_iter"] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(clues, single, "o-", label="Single-pass", linewidth=2)
    ax.plot(clues, it, "s-", label="Iterative", linewidth=2)
    ax.set_xlabel("Number of given clues")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(
        f"SudokuSolver accuracy vs. number of clues\n"
        f"checkpoint: {checkpoint}  ({n} puzzles per point)"
    )
    ax.set_xticks(clues)
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.invert_xaxis()  # easy → hard left-to-right
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default="runs/sudoku_cnn_20260517_163449/last.pt",
    )
    p.add_argument(
        "--clues",
        default="80,75,70,65,60,55,50,45,40,35,30,25,20,17",
        help="Comma-separated clue counts to evaluate.",
    )
    p.add_argument("--num_puzzles", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=250)
    p.add_argument("--device", default="cuda:2" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}\n")

    clue_counts = [int(c) for c in args.clues.split(",")]
    print(f"Clue sweep: {clue_counts}")
    print(f"Puzzles per point: {args.num_puzzles}  (batch={args.batch_size})\n")

    results = run_sweep(
        model=model,
        device=device,
        clue_counts=clue_counts,
        num_puzzles=args.num_puzzles,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    out_dir = Path(args.out_dir)
    csv_path = out_dir / "accuracy_vs_clues.csv"
    png_path = out_dir / "accuracy_vs_clues.png"
    save_csv(results, csv_path)
    save_plot(results, png_path, args.checkpoint, args.num_puzzles)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()

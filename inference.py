"""
test.py — Load a trained SudokuSolver and run inference on sample puzzles.
"""

import argparse
import time

import numpy as np
import torch

from dataset import Sudoku
from model import SudokuSolver

torch.backends.cudnn.benchmark = True


def load_model(path: str, device: torch.device) -> SudokuSolver:
    model = SudokuSolver(
        embed_dim=64,
        channels=256,
        num_res_blocks=2,
        num_transformer_blocks=4,
        num_heads=4,
        dropout_rate=0.1,
    ).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def make_puzzle(num_clues: int):
    """Return (puzzle, solution) as (9,9) numpy arrays with 0-indexed values (0–8)."""
    solved = Sudoku.generate_solved_board() - 1  # 0-indexed
    puzzle = solved.copy()
    masked_indices = np.random.permutation(81)[: 81 - num_clues]
    puzzle.ravel()[masked_indices] = 9  # 9 = unknown token
    return puzzle, solved


def print_board(board: np.ndarray, title: str = "") -> None:
    digits = np.where(board == 9, 0, board + 1)  # back to 1-indexed, 0 = blank
    if title:
        print(title)
    for r in range(9):
        row = ""
        for c in range(9):
            row += (str(digits[r, c]) if digits[r, c] else ".") + " "
            if c in (2, 5):
                row += "| "
        print(row)
        if r in (2, 5):
            print("-" * 21)
    print()


def is_valid_solution(pred: np.ndarray) -> bool:
    digits = pred + 1  # 1-indexed
    target = set(range(1, 10))
    for i in range(9):
        if set(digits[i, :]) != target:
            return False
        if set(digits[:, i]) != target:
            return False
        box = digits[i // 3 * 3 : i // 3 * 3 + 3, (i % 3) * 3 : (i % 3) * 3 + 3]
        if set(box.ravel()) != target:
            return False
    return True


@torch.no_grad()
def solve(model: SudokuSolver, puzzle: np.ndarray, device: torch.device) -> np.ndarray:
    inp = torch.from_numpy(puzzle).long().unsqueeze(0).to(device)  # (1, 9, 9)
    logits = model(inp)  # (1, 9, 9, 9)
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (9, 9)
    # Keep given clues as-is, fill only masked cells
    out = puzzle.copy()
    out[puzzle == 9] = pred[puzzle == 9]
    return out


def _sync(device: torch.device) -> None:
    """Block until pending CUDA work finishes so wall-clock timing is honest."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def solve_classic(puzzle: np.ndarray) -> np.ndarray:
    """
    Classic MRV backtracking solver (no model). Wraps `Sudoku.solve_board_mrv`,
    which expects 1-indexed boards with 0 for empty cells. Converts from the
    inference encoding (9 = unknown, 0–8 = digits) and back.
    """
    board = np.where(puzzle == 9, 0, puzzle + 1).astype(np.int64)
    Sudoku.solve_board_mrv(board)
    return board - 1


@torch.no_grad()
def solve_iterative(
    model: SudokuSolver, puzzle: np.ndarray, device: torch.device
) -> np.ndarray:
    """
    Iterative inference: each step fix only the single most-confident masked
    cell, then re-run the model with that cell now revealed. Repeat until all
    cells are filled or the model stops making progress.
    """
    board = puzzle.copy()

    while True:
        masked = np.argwhere(board == 9)  # remaining unknown cells
        if len(masked) == 0:
            break

        inp = torch.from_numpy(board).long().unsqueeze(0).to(device)  # (1,9,9)
        logits = model(inp)  # (1,9,9,9)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (9,9,9)

        # Find the masked cell with the highest max-class probability
        best_conf, best_r, best_c, best_digit = -1.0, -1, -1, -1
        for r, c in masked:
            conf = probs[:, r, c].max()
            digit = probs[:, r, c].argmax()
            if conf > best_conf:
                best_conf, best_r, best_c, best_digit = conf, r, c, digit

        if best_r == -1:  # no progress (shouldn't happen)
            break

        board[best_r, best_c] = best_digit

    return board


def parse_args():
    parser = argparse.ArgumentParser(description="SudokuSolver inference")
    parser.add_argument(
        "--checkpoint", type=str, default="runs/sudoku_cnn_20260509_111124/best.pt"
    )
    parser.add_argument("--num_puzzles", type=int, default=50)
    parser.add_argument(
        "--num_clues",
        type=int,
        default=18,
        help="Number of given cells (rest are masked)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}")
    model = load_model(args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}\n")

    # Warmup: first forward pass on CUDA pays cuDNN/kernel init costs that
    # would otherwise inflate the first puzzle's timing. Numba also JIT-compiles
    # the classic solver on first call — fold that into warmup too.
    warm_puzzle, _ = make_puzzle(args.num_clues)
    solve(model, warm_puzzle, device)
    _sync(device)
    solve_classic(warm_puzzle)

    correct_single, correct_iter, correct_classic = 0, 0, 0
    time_single, time_iter, time_classic = 0.0, 0.0, 0.0
    for i in range(args.num_puzzles):
        puzzle, solution = make_puzzle(args.num_clues)

        _sync(device)
        t0 = time.perf_counter()
        pred_single = solve(model, puzzle, device)
        _sync(device)
        time_single += time.perf_counter() - t0

        _sync(device)
        t0 = time.perf_counter()
        pred_iter = solve_iterative(model, puzzle, device)
        _sync(device)
        time_iter += time.perf_counter() - t0

        t0 = time.perf_counter()
        pred_classic = solve_classic(puzzle)
        time_classic += time.perf_counter() - t0

        valid_single = is_valid_solution(pred_single)
        valid_iter = is_valid_solution(pred_iter)
        valid_classic = is_valid_solution(pred_classic)
        if valid_single:
            correct_single += 1
        if valid_iter:
            correct_iter += 1
        if valid_classic:
            correct_classic += 1

        print(f"=== Puzzle {i + 1} ===")
        print_board(puzzle, "Input:")
        print_board(pred_single, f"Single-pass (valid={valid_single}):")
        print_board(pred_iter, f"Iterative   (valid={valid_iter}):")
        print_board(pred_classic, f"Classic MRV (valid={valid_classic}):")
        print_board(solution, "Ground truth:")
        print("=" * 40 + "\n")

    n = args.num_puzzles
    print(f"{'Method':<18} {'Accuracy':<12} {'Total':<10} {'Avg/puzzle'}")
    print("-" * 55)
    print(
        f"{'Single-pass':<18} {correct_single}/{n:<10} {time_single:7.2f}s   {time_single / n * 1000:7.1f} ms"
    )
    print(
        f"{'Iterative':<18} {correct_iter}/{n:<10} {time_iter:7.2f}s   {time_iter / n * 1000:7.1f} ms"
    )
    print(
        f"{'Classic MRV':<18} {correct_classic}/{n:<10} {time_classic:7.2f}s   {time_classic / n * 1000:7.1f} ms"
    )


if __name__ == "__main__":
    main()

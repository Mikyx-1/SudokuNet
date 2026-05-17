"""
test.py — Load a trained SudokuSolver and run inference on sample puzzles.
"""

import argparse
from operator import itemgetter

import numpy as np
import torch

from dataset import Sudoku
from model import SudokuSolver


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


def _allowed_digits(board: np.ndarray, r: int, c: int) -> list:
    """Digits (0-8) still consistent with the row/col/box of cell (r,c)."""
    used = set(board[r, :].tolist())
    used.update(board[:, c].tolist())
    br, bc = (r // 3) * 3, (c // 3) * 3
    used.update(board[br : br + 3, bc : bc + 3].ravel().tolist())
    used.discard(9)  # 9 = unknown, not a real digit
    return [d for d in range(9) if d not in used]


def _pick_mrv_cell(board: np.ndarray):
    """
    Most-constrained masked cell (MRV). Returns (r, c, allowed) for the masked
    cell with the fewest allowed digits, or None if some masked cell has zero
    allowed digits (dead-end board).
    """
    best = None
    for r in range(9):
        for c in range(9):
            if board[r, c] != 9:
                continue
            allowed = _allowed_digits(board, r, c)
            if not allowed:
                return None
            if best is None or len(allowed) < len(best[2]):
                best = (r, c, allowed)
    return best


@torch.no_grad()
def solve_beam(
    model: SudokuSolver,
    puzzle: np.ndarray,
    device: torch.device,
    beam_width: int = 32,
) -> np.ndarray:
    """
    Beam search with constraint masking and MRV cell ordering.

    Each beam carries a (partial board, cumulative log-prob). Per step, every
    active beam picks its most-constrained masked cell and branches on every
    digit still consistent with that cell's row/col/box. The top `beam_width`
    candidates by total log-prob survive; completed beams stay in the pool
    and compete on score. Returns the highest-scoring completed board.
    """
    beams = [(puzzle.copy(), 0.0)]

    while True:
        active = [(b, s) for b, s in beams if (b == 9).any()]
        complete = [(b, s) for b, s in beams if not (b == 9).any()]
        if not active:
            break

        inp = torch.from_numpy(np.stack([b for b, _ in active])).long().to(device)
        logits = model(inp)  # (B, 9, 9, 9), class dim at 1
        log_probs = torch.log_softmax(logits, dim=1).cpu().numpy()

        candidates = list(complete)
        for i, (board, score) in enumerate(active):
            picked = _pick_mrv_cell(board)
            if picked is None:
                continue  # dead-end beam, prune
            r, c, allowed = picked
            for d in allowed:
                new_board = board.copy()
                new_board[r, c] = d
                candidates.append((new_board, score + float(log_probs[i, d, r, c])))

        if not candidates:
            return puzzle.copy()  # every beam died; give up

        candidates.sort(key=itemgetter(1), reverse=True)
        beams = candidates[:beam_width]

    return beams[0][0]


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
        "--beam_width",
        type=int,
        default=32,
        help="Beam search width for solve_beam",
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

    correct_single, correct_iter, correct_beam = 0, 0, 0
    for i in range(args.num_puzzles):
        puzzle, solution = make_puzzle(args.num_clues)
        pred_single = solve(model, puzzle, device)
        pred_iter = solve_iterative(model, puzzle, device)
        pred_beam = solve_beam(model, puzzle, device, beam_width=args.beam_width)
        valid_single = is_valid_solution(pred_single)
        valid_iter = is_valid_solution(pred_iter)
        valid_beam = is_valid_solution(pred_beam)
        if valid_single:
            correct_single += 1
        if valid_iter:
            correct_iter += 1
        if valid_beam:
            correct_beam += 1

        print(f"=== Puzzle {i + 1} ===")
        print_board(puzzle, "Input:")
        print_board(pred_single, f"Single-pass (valid={valid_single}):")
        print_board(pred_iter, f"Iterative   (valid={valid_iter}):")
        print_board(pred_beam, f"Beam B={args.beam_width} (valid={valid_beam}):")
        print_board(solution, "Ground truth:")
        print("=" * 40 + "\n")

    print(f"Single-pass accuracy:  {correct_single}/{args.num_puzzles}")
    print(f"Iterative   accuracy:  {correct_iter}/{args.num_puzzles}")
    print(f"Beam B={args.beam_width} accuracy: {correct_beam}/{args.num_puzzles}")


if __name__ == "__main__":
    main()

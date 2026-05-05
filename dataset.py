import random
from typing import Tuple

import numpy as np
import torch
from numba import njit
from torch.utils.data import Dataset


@njit(cache=True)
def _solve_mrv(board, rows, cols, boxes):
    VALID_BITS = 0b1111111110

    best_pos = -1
    best_count = 10
    best_free = 0

    for r in range(9):
        for c in range(9):
            if board[r, c] == 0:
                used = rows[r] | cols[c] | boxes[(r // 3) * 3 + c // 3]
                free = VALID_BITS & ~used

                tmp = free
                count = 0
                while tmp:
                    tmp &= tmp - 1
                    count += 1
                if count == 0:
                    return False
                if count < best_count:
                    best_count = count
                    best_pos = r * 9 + c
                    best_free = free
        if best_count == 1:
            break

    if best_pos == -1:
        return True

    r, c = best_pos // 9, best_pos % 9
    b = (r // 3) * 3 + c // 3

    # Collect candidates into a fixed-size array
    candidates = np.empty(9, dtype=np.int64)
    n_candidates = 0
    free = best_free
    while free:
        bit = free & (-free)
        free &= free - 1
        candidates[n_candidates] = bit
        n_candidates += 1

    # Fisher-Yates shuffle for randomness
    for i in range(n_candidates - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        candidates[i], candidates[j] = candidates[j], candidates[i]

    for k in range(n_candidates):
        bit = candidates[k]
        num = 0
        tmp = bit
        while tmp > 1:
            tmp >>= 1
            num += 1

        board[r, c] = num
        rows[r] |= bit
        cols[c] |= bit
        boxes[b] |= bit

        if _solve_mrv(board, rows, cols, boxes):
            return True

        board[r, c] = 0
        rows[r] ^= bit
        cols[c] ^= bit
        boxes[b] ^= bit

    return False


class Sudoku:

    @staticmethod
    def solve_board_mrv(board: np.ndarray) -> bool:
        rows = np.zeros(9, dtype=np.int64)
        cols = np.zeros(9, dtype=np.int64)
        boxes = np.zeros(9, dtype=np.int64)

        for r in range(9):
            for c in range(9):
                n = int(board[r, c])
                if n:
                    mask = 1 << n
                    rows[r] |= mask
                    cols[c] |= mask
                    boxes[(r // 3) * 3 + c // 3] |= mask

        return _solve_mrv(board, rows, cols, boxes)

    @staticmethod
    def generate_solved_board() -> np.ndarray:
        board = np.zeros((9, 9), dtype=np.int64)
        # Seed diagonal boxes for randomness without needing shuffle in the solver
        for box in range(3):
            nums = np.random.permutation(np.arange(1, 10, dtype=np.int64))
            for i in range(3):
                for j in range(3):
                    board[box * 3 + i, box * 3 + j] = nums[i * 3 + j]
        Sudoku.solve_board_mrv(board)
        return board


class SudokuDataset(Dataset):
    """
    Generates Sudoku puzzles on-the-fly.

    Values are 0-indexed (0–8 = digits 1–9, 9 = masked/unknown).
    The mask is True for cells the model must predict.
    """

    def __init__(self, num_samples: int, min_mask: int = 1, max_mask: int = 64):
        self.num_samples = num_samples
        self.min_mask = min_mask
        self.max_mask = max_mask

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        solved = Sudoku.generate_solved_board() - 1  # shape (9,9), values 0–8
        masked, mask = self._mask(solved)
        return (
            torch.from_numpy(masked).long(),  # input:  (9,9)
            torch.from_numpy(solved).long(),  # target: (9,9)
            torch.from_numpy(mask),  # mask:   (9,9) bool
        )

    def _mask(self, solved: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mask = np.zeros((9, 9), dtype=bool)
        masked = solved.copy()
        indices = np.random.permutation(81)
        n = random.randint(self.min_mask, min(self.max_mask, 81))
        flat = masked.ravel()
        flat_mask = mask.ravel()
        selected = indices[:n]
        flat[selected] = 9
        flat_mask[selected] = True
        return masked, mask

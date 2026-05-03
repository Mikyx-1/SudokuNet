import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class Sudoku:
    @staticmethod
    def is_valid(board: np.ndarray, row: int, col: int, num: int) -> bool:
        if num in board[row, :]:
            return False
        if num in board[:, col]:
            return False
        sr, sc = row - row % 3, col - col % 3
        if num in board[sr : sr + 3, sc : sc + 3]:
            return False
        return True

    @staticmethod
    def solve_board_backtracking(board: np.ndarray) -> bool:
        for row in range(9):
            for col in range(9):
                if board[row, col] == 0:
                    nums = list(range(1, 10))
                    random.shuffle(nums)
                    for num in nums:
                        if Sudoku.is_valid(board, row, col, num):
                            board[row, col] = num
                            if Sudoku.solve_board_backtracking(board):
                                return True
                            board[row, col] = 0
                    return False
        return True

    @staticmethod
    def solve_board_mrv(board: np.ndarray) -> bool:
        rows = [0] * 9
        cols = [0] * 9
        boxes = [0] * 9

        for r in range(9):
            for c in range(9):
                n = int(board[r, c])
                if n:
                    mask = 1 << n
                    rows[r] |= mask
                    cols[c] |= mask
                    boxes[(r // 3) * 3 + c // 3] |= mask

        VALID_BITS = 0b1111111110  # bits 1-9 only, bit 0 excluded

        def solve() -> bool:
            best_pos = -1
            best_count = 10
            best_free = 0

            for r in range(9):
                for c in range(9):
                    if board[r, c] == 0:
                        used = rows[r] | cols[c] | boxes[(r // 3) * 3 + c // 3]
                        free = VALID_BITS & ~used  # only bits 1-9
                        count = bin(free).count("1")
                        if count == 0:
                            return False
                        if count < best_count:
                            best_count = count
                            best_pos = r * 9 + c
                            best_free = free
                            if count == 1:
                                break
                if best_count == 1:
                    break

            if best_pos == -1:
                return True

            r, c = divmod(best_pos, 9)
            b = (r // 3) * 3 + c // 3

            # Collect all candidate bits
            candidates = []
            free = best_free
            while free:
                bit = free & -free
                free &= free - 1
                candidates.append(bit)

            # Shuffle to introduce randomness
            random.shuffle(candidates)

            # Try candidates in random order
            for bit in candidates:
                num = bit.bit_length() - 1  # still 1–9

                board[r, c] = num
                rows[r] |= bit
                cols[c] |= bit
                boxes[b] |= bit

                if solve():
                    return True

                board[r, c] = 0
                rows[r] ^= bit
                cols[c] ^= bit
                boxes[b] ^= bit

            return False

        return solve()

    @staticmethod
    def generate_solved_board() -> np.ndarray:
        board = np.zeros((9, 9), dtype=np.int64)
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
        for k in indices[:n]:
            flat[k] = 9  # sentinel for "unknown"
            flat_mask[k] = True
        return masked, mask

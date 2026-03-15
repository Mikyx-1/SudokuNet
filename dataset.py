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
    def solve_board(board: np.ndarray) -> bool:
        for row in range(9):
            for col in range(9):
                if board[row, col] == 0:
                    nums = list(range(1, 10))
                    random.shuffle(nums)
                    for num in nums:
                        if Sudoku.is_valid(board, row, col, num):
                            board[row, col] = num
                            if Sudoku.solve_board(board):
                                return True
                            board[row, col] = 0
                    return False
        return True

    @staticmethod
    def generate_solved_board() -> np.ndarray:
        board = np.zeros((9, 9), dtype=np.int64)
        Sudoku.solve_board(board)
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

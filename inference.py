"""
inference.py — evaluate a trained checkpoint on fresh puzzles.

Usage
-----
python inference.py --checkpoint runs/my_run/checkpoints/epoch_0010_best.pt
python inference.py --checkpoint runs/my_run/checkpoints/epoch_0010_best.pt --samples 50
"""

import argparse
from pathlib import Path

import torch

from config import TrainConfig
from dataset import SudokuDataset
from model import SudokuSolver


def load_model(ckpt_path: str, device: torch.device) -> SudokuSolver:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt.get("cfg", {})
    model = SudokuSolver(
        embed_dim=cfg_dict.get("embed_dim", 64),
        channels=cfg_dict.get("channels", 256),
        num_res_blocks=cfg_dict.get("num_res_blocks", 8),
        num_heads=cfg_dict.get("num_heads", 8),
        dropout_rate=0.0,  # no dropout at inference
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model


@torch.no_grad()
def evaluate(model: SudokuSolver, dataset: SudokuDataset, device: torch.device):
    total_masked = total_masked_correct = total_boards = total_boards_correct = 0

    for i in range(len(dataset)):
        inp, target, mask = dataset[i]
        inp = inp.unsqueeze(0).to(device)
        target = target.to(device)
        mask = mask.to(device)

        logits = model(inp).squeeze(0)  # (9, 9, 9) — remove batch dim
        pred = logits.argmax(dim=0)  # (9, 9)

        correct = pred == target
        masked_correct = (correct & mask).sum().item()
        n_masked = mask.sum().item()

        total_masked += n_masked
        total_masked_correct += masked_correct
        total_boards += 1
        total_boards_correct += int(correct.all().item())

        if i < 3:  # print first 3 examples
            print(f"\n── Sample {i+1} ──")
            print("Input   (9=masked):\n", inp.squeeze(0).cpu().numpy() + 1)
            print("Target:\n", target.cpu().numpy() + 1)
            print("Pred:\n", pred.cpu().numpy() + 1)
            print(f"Masked cells correct: {masked_correct}/{n_masked}")

    masked_acc = total_masked_correct / max(1, total_masked) * 100
    full_board_acc = total_boards_correct / total_boards * 100
    print(f"\n── Results over {total_boards} puzzles ──")
    print(f"  Masked-cell accuracy : {masked_acc:.2f}%")
    print(f"  Full-board accuracy  : {full_board_acc:.2f}%")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--samples", type=int, default=20, help="Number of test puzzles")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    ds = SudokuDataset(num_samples=args.samples)
    evaluate(model, ds, device)

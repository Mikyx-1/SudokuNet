"""
Improved Sudoku Solver
======================
Key upgrades over the original:
  1. Structured positional encoding  – separate row / col / box embeddings
     so the attention layers always know *where* each cell sits.
  2. Deep Transformer trunk           – N stacked TransformerEncoderLayers
     (self-attention + FFN + LayerNorm) instead of a single MHA call.
  3. Constraint-axis attention        – dedicated attention passes that are
     masked to only attend within the same row, same column, and same 3×3
     box.  This gives the model an explicit inductive bias that matches
     Sudoku's hard constraints without forcing it to learn them from scratch.
  4. Lighter residual-CNN front-end   – still used, but now just to build a
     rich per-cell feature before handing off to the Transformer.
  5. Correct output permute           – logits are returned as (B, 9, 9, 9)
     (batch, row, col, digit_class) so they plug directly into
     F.cross_entropy after a .view(B*81, 9).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AxisAttention(nn.Module):
    """
    Multi-head attention restricted to one Sudoku axis (row/col/box).

    Tokens outside the axis are masked out with -inf, so the softmax
    never attends to them.  This is O(81·k) instead of O(81²) per head
    where k≤9, but we keep the standard MHA implementation for simplicity.
    """

    def __init__(self, channels: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            channels, num_heads, batch_first=True, dropout=dropout
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, 81, C)
        mask : (81, 81) bool – True where attention is *allowed*
        """
        # Pre-norm: normalize before attention for stable gradient flow.
        # Boolean attn_mask: True = blocked. ~mask inverts the allow-mask.
        # Using bool avoids float16 overflow when called inside autocast.
        normed = self.norm(x)
        out, _ = self.attn(normed, normed, normed, attn_mask=~mask)
        return x + out


class SudokuTransformerBlock(nn.Module):
    """
    One full block:
      1. Global self-attention  (all 81 cells)
      2. Row-axis  attention
      3. Col-axis  attention
      4. Box-axis  attention
      5. Position-wise FFN
    """

    def __init__(
        self, channels: int, num_heads: int, ffn_mult: int = 4, dropout: float = 0.1
    ):
        super().__init__()
        self.global_attn = nn.MultiheadAttention(
            channels, num_heads, batch_first=True, dropout=dropout
        )
        self.global_norm = nn.LayerNorm(channels)

        self.row_attn = AxisAttention(channels, num_heads, dropout)
        self.col_attn = AxisAttention(channels, num_heads, dropout)
        self.box_attn = AxisAttention(channels, num_heads, dropout)

        hidden = channels * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(channels)

    def forward(
        self,
        x: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
        box_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Pre-norm throughout: normalize before each sub-layer, add residual after.

        # 1. Global attention
        normed = self.global_norm(x)
        g, _ = self.global_attn(normed, normed, normed)
        x = x + g

        # 2-4. Constraint-axis attention (pre-norm inside AxisAttention)
        x = self.row_attn(x, row_mask)
        x = self.col_attn(x, col_mask)
        x = self.box_attn(x, box_mask)

        # 5. FFN
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ResidualBlock(nn.Module):
    """Lightweight spatial residual block (kept for the CNN front-end)."""

    def __init__(self, channels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class SudokuSolver(nn.Module):
    """
    Improved CNN + Constraint-aware Transformer Sudoku solver.

    Input:  (B, 9, 9) long tensor, values 0–9  (9 = masked / unknown cell)
    Output: (B, 9, 9, 9) logits over digit classes 1-9 per cell

    Usage
    -----
    logits = model(puzzle)                       # (B, 9, 9, 9)
    loss   = F.cross_entropy(
                 logits.view(-1, 9),
                 target.view(-1)                 # (B*81,) values 0-8
             )
    pred   = logits.argmax(-1) + 1              # (B, 9, 9) digit predictions
    """

    def __init__(
        self,
        embed_dim: int = 64,
        channels: int = 256,
        num_res_blocks: int = 4,  # fewer CNN blocks; Transformer does more
        num_transformer_blocks: int = 8,
        num_heads: int = 8,
        ffn_mult: int = 4,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        # ── Embedding ────────────────────────────────────────────────────────
        # 10 tokens: digits 1-9 → indices 0-8, unknown → index 9
        self.embedding = nn.Embedding(10, embed_dim)

        # ── Structured positional encodings ──────────────────────────────────
        # Learnable row, column, and box embeddings (9 each).
        # Concatenated then projected so the model explicitly knows the
        # three constraint-relevant coordinates of every cell.
        self.row_emb = nn.Embedding(9, embed_dim)
        self.col_emb = nn.Embedding(9, embed_dim)
        self.box_emb = nn.Embedding(9, embed_dim)
        self.pos_proj = nn.Linear(embed_dim * 3, embed_dim, bias=False)

        # ── CNN front-end ─────────────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Conv2d(embed_dim, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels, dropout_rate) for _ in range(num_res_blocks)]
        )

        # ── Transformer trunk ─────────────────────────────────────────────────
        self.transformer_blocks = nn.ModuleList(
            [
                SudokuTransformerBlock(channels, num_heads, ffn_mult, dropout_rate)
                for _ in range(num_transformer_blocks)
            ]
        )

        # ── Output head ───────────────────────────────────────────────────────
        self.output_head = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, 9),  # 9 digit classes
        )

        # Pre-compute axis masks (registered as buffers → move with .to(device))
        row_idx = torch.arange(81) // 9
        col_idx = torch.arange(81) % 9
        box_idx = (row_idx // 3) * 3 + col_idx // 3
        row_mask = row_idx.unsqueeze(1) == row_idx.unsqueeze(0)
        col_mask = col_idx.unsqueeze(1) == col_idx.unsqueeze(0)
        box_mask = box_idx.unsqueeze(1) == box_idx.unsqueeze(0)
        self.register_buffer("row_mask", row_mask)
        self.register_buffer("col_mask", col_mask)
        self.register_buffer("box_mask", box_mask)

        # Pre-compute positional index tensors
        rows = torch.arange(9).unsqueeze(1).expand(9, 9).reshape(81)
        cols = torch.arange(9).unsqueeze(0).expand(9, 9).reshape(81)
        boxes = (rows // 3) * 3 + cols // 3
        self.register_buffer("pos_rows", rows)
        self.register_buffer("pos_cols", cols)
        self.register_buffer("pos_boxes", boxes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 9, 9) long, values 0-9
        returns (B, 9, 9, 9) logits
        """
        B = x.size(0)

        # ── Token embedding ───────────────────────────────────────────────────
        tok = self.embedding(x)  # (B, 9, 9, E)

        # ── Structured positional encoding ────────────────────────────────────
        r_emb = self.row_emb(self.pos_rows)  # (81, E)
        c_emb = self.col_emb(self.pos_cols)  # (81, E)
        b_emb = self.box_emb(self.pos_boxes)  # (81, E)
        pos = self.pos_proj(torch.cat([r_emb, c_emb, b_emb], dim=-1))  # (81, E)
        pos = pos.view(9, 9, -1)  # (9, 9, E)

        tok = tok + pos.unsqueeze(0)  # (B, 9, 9, E)  broadcast over B

        # ── CNN front-end ─────────────────────────────────────────────────────
        feat = tok.permute(0, 3, 1, 2)  # (B, E, 9, 9)
        feat = self.input_proj(feat)  # (B, C, 9, 9)
        feat = self.res_blocks(feat)  # (B, C, 9, 9)

        # ── Flatten to sequence ───────────────────────────────────────────────
        seq = feat.permute(0, 2, 3, 1).reshape(B, 81, -1)  # (B, 81, C)

        # ── Transformer blocks ────────────────────────────────────────────────
        for block in self.transformer_blocks:
            seq = block(seq, self.row_mask, self.col_mask, self.box_mask)

        # ── Output ────────────────────────────────────────────────────────────
        logits = self.output_head(seq)  # (B, 81, 9)
        return logits.view(B, 9, 9, 9).permute(
            0, 3, 1, 2
        )  # (B, 9, 9, 9) — class at dim 1, compatible with nn.CrossEntropyLoss


# ---------------------------------------------------------------------------
# Example training snippet
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SudokuSolver(
        embed_dim=64,
        channels=256,
        num_res_blocks=4,
        num_transformer_blocks=8,
        num_heads=8,
        dropout_rate=0.1,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")

    # Dummy batch: values 0-8 = digits 1-9, value 9 = unknown cell
    puzzle = torch.randint(0, 10, (4, 9, 9), device=device)
    target = torch.randint(0, 9, (4, 9, 9), device=device)  # 0-8 → digits 1-9

    logits = model(puzzle)  # (4, 9, 9, 9)
    loss = F.cross_entropy(logits.reshape(-1, 9), target.reshape(-1))
    print(f"Output shape : {logits.shape}")
    print(f"Loss         : {loss.item():.4f}")

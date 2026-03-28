import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class SudokuSolver(nn.Module):
    """
    CNN + optional multi-head self-attention Sudoku solver.

    Input:  (B, 9, 9)  long tensor, values 0–9  (9 = masked)
    Output: (B, 9, 9, 9) logits over 9 digit classes per cell
    """

    def __init__(
        self,
        embed_dim: int = 64,
        channels: int = 256,
        num_res_blocks: int = 8,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        # 10 embeddings: digits 0-8 (1-9 offset) + 9 (masked token)
        self.embedding = nn.Embedding(10, embed_dim)

        self.input_proj = nn.Sequential(
            nn.Conv2d(embed_dim, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels, dropout_rate) for _ in range(num_res_blocks)]
        )

        self.attn = nn.MultiheadAttention(
            channels, num_heads, batch_first=True, dropout=dropout_rate
        )
        self.attn_norm = nn.LayerNorm(channels)

        self.output_proj = nn.Conv2d(channels, 9, 1)  # 9 classes per cell

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 9, 9)
        x = self.embedding(x)  # (B, 9, 9, E)
        x = x.permute(0, 3, 1, 2)  # (B, E, 9, 9)

        x = self.input_proj(x)  # (B, C, 9, 9)
        x = self.res_blocks(x)  # (B, C, 9, 9)

        b, c, h, w = x.shape
        seq = x.view(b, c, h * w).permute(0, 2, 1)  # (B, 81, C)
        attn_out, _ = self.attn(seq, seq, seq)
        seq = self.attn_norm(seq + attn_out)  # residual + norm
        x = seq.permute(0, 2, 1).view(b, c, h, w)  # (B, C, 9, 9)

        return self.output_proj(x)  # (B, 9, 9, 9)

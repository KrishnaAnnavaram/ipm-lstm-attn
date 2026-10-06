"""Unrolled learned solver for ``J dy = -F`` (needs torch).

Each entry of ``dy`` is one "token". At every inner step a token gets two
features: its current value and the gradient of ``0.5 * ||J dy + F||^2`` with
respect to it. An optional mixing block (attention or message passing) lets
tokens exchange information. A stack of LSTM cells with shared weights then
proposes an update for each token.

The system is scaled by ``||F||`` first. The Newton system is linear in ``F``,
so the solver works on a unit-size problem and the result is scaled back.
"""
from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from ..registry import VARIANTS


def structure_mask(J: torch.Tensor) -> torch.Tensor:
    """(B, N, N) bool: True where tokens k and l share a row of J (the pattern of J^T J)."""
    nz = (J.abs() > 0).to(J.dtype)
    conn = (nz.transpose(1, 2) @ nz) > 0
    eye = torch.eye(J.shape[-1], dtype=torch.bool, device=J.device)
    return conn | eye


class AttentionBlock(nn.Module):
    """Multi-head self-attention over tokens with a residual path and layer norm."""

    def __init__(self, d_model: int, heads: int, masked: bool) -> None:
        super().__init__()
        self.heads = heads
        self.masked = masked
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, allowed: torch.Tensor | None) -> torch.Tensor:
        attn_mask = None
        if self.masked:
            if allowed is None:
                raise ValueError("masked attention needs the structure mask")
            attn_mask = (~allowed).repeat_interleave(self.heads, dim=0)  # True = blocked
        out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        return self.norm(h + out)


class MessagePassingBlock(nn.Module):
    """One round of messages on the bipartite graph of J: tokens -> rows -> tokens."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.to_row = nn.Linear(d_model, d_model)
        self.update = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, row_w: torch.Tensor, col_w: torch.Tensor) -> torch.Tensor:
        rows = torch.relu(self.to_row(row_w @ h))  # (B, rows, d)
        back = col_w @ rows  # (B, tokens, d)
        return self.norm(h + self.update(torch.cat([h, back], dim=-1)))


class LearnedNewtonSolver(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        variant = VARIANTS[cfg.variant]
        self.variant = variant.name
        self.block_kind = variant.block
        self.inner_steps = cfg.inner_steps
        self.hidden_dim = cfg.hidden_dim
        in_dim = 2
        if self.block_kind == "none":
            self.embed = None
            self.block = None
            lstm_in = in_dim
        else:
            self.embed = nn.Linear(in_dim, cfg.d_model)
            if self.block_kind == "gnn":
                self.block = MessagePassingBlock(cfg.d_model)
            else:
                self.block = AttentionBlock(cfg.d_model, cfg.heads, masked=self.block_kind == "masked_attention")
            lstm_in = cfg.d_model
        self.cells = nn.ModuleList(
            nn.LSTMCell(lstm_in if i == 0 else cfg.hidden_dim, cfg.hidden_dim) for i in range(variant.num_layers)
        )
        # The only dropout module. It acts between stacked LSTM layers in training mode.
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden_dim, 1)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    @property
    def num_layers(self) -> int:
        return len(self.cells)

    def forward(self, J: torch.Tensor, F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (dy, loss, relative residual of dy) for J (B, N, N) and F (B, N)."""
        B, N = F.shape
        scale = F.norm(dim=1, keepdim=True).clamp_min(1e-12)
        f = F / scale
        Jt = J.transpose(1, 2)
        allowed = structure_mask(J) if self.block_kind == "masked_attention" else None
        row_w = col_w = None
        if self.block_kind == "gnn":
            a = J.abs()
            row_w = a / a.sum(dim=2, keepdim=True).clamp_min(1e-12)
            col_w = a.transpose(1, 2) / a.sum(dim=1).unsqueeze(-1).clamp_min(1e-12)
        states = [
            (f.new_zeros(B * N, self.hidden_dim), f.new_zeros(B * N, self.hidden_dim)) for _ in self.cells
        ]
        dy = torch.zeros_like(f)
        best = dy
        best_res = 0.5 * (f * f).sum(dim=1)
        loss = f.new_zeros(())
        for _ in range(self.inner_steps):
            r = (J @ dy.unsqueeze(-1)).squeeze(-1) + f
            grad = (Jt @ r.unsqueeze(-1)).squeeze(-1)
            h = torch.stack([dy, grad], dim=-1)  # (B, N, 2)
            if self.embed is not None:
                h = self.embed(h)
                if self.block_kind == "gnn":
                    h = self.block(h, row_w, col_w)
                else:
                    h = self.block(h, allowed)
            x = h.reshape(B * N, -1)
            for i, cell in enumerate(self.cells):
                hx, cx = cell(x, states[i])
                states[i] = (hx, cx)
                x = self.dropout(hx) if i < len(self.cells) - 1 else hx
            dy = dy - self.head(x).view(B, N)
            r = (J @ dy.unsqueeze(-1)).squeeze(-1) + f
            res = 0.5 * (r * r).sum(dim=1)
            loss = loss + res.mean() / self.inner_steps
            better = res < best_res
            best = torch.where(better.unsqueeze(-1), dy, best)
            best_res = torch.where(better, res, best_res)
        rel = torch.sqrt(2.0 * best_res)
        return best * scale, loss, rel


def build_model(cfg: ModelConfig) -> LearnedNewtonSolver:
    return LearnedNewtonSolver(cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

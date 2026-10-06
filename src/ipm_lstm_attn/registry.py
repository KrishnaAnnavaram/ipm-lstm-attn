"""Model variants for the ablation. Each variant adds exactly one change to the one before it.

This module has no torch import, so the config can validate variant names
without the ``torch`` extra.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Variant:
    name: str
    num_layers: int
    block: str  # "none", "attention", "masked_attention" or "gnn"
    change: str  # the one change against the parent variant
    parent: str | None


VARIANTS: dict[str, Variant] = {
    v.name: v
    for v in (
        Variant("lstm1", 1, "none", "baseline: one LSTM layer, no mixing block (upstream design)", None),
        Variant("lstm2", 2, "none", "second LSTM layer", "lstm1"),
        Variant("lstm2_attn", 2, "attention", "dense multi-head attention with residual and layer norm", "lstm2"),
        Variant("lstm2_mask", 2, "masked_attention", "attention masked by the sparsity of J^T J", "lstm2_attn"),
        Variant("lstm2_gnn", 2, "gnn", "message passing over the rows and columns of J instead of attention", "lstm2"),
    )
}

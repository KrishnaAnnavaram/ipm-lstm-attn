"""Solvers for the batched Newton system ``J dy = -F``.

A Newton solver takes ``J`` with shape (B, N, N) and ``F`` with shape (B, N) and
returns ``dy`` with shape (B, N). The learned solver in
``ipm_lstm_attn.models.adapter`` uses the same interface.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class NewtonSolver(Protocol):
    name: str

    def solve(self, J: np.ndarray, F: np.ndarray) -> np.ndarray:  # pragma: no cover - protocol
        ...


class DirectSolver:
    """Exact solve with LU factorisation. Falls back to least squares for a singular system."""

    name = "direct"

    def solve(self, J: np.ndarray, F: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.solve(J, -F[..., None])[..., 0]
        except np.linalg.LinAlgError:
            out = np.empty_like(F)
            for i in range(J.shape[0]):
                out[i] = np.linalg.lstsq(J[i], -F[i], rcond=None)[0]
            return out


class CGNormalSolver:
    """Conjugate gradient on the normal equations ``J^T J dy = -J^T F`` with a fixed budget.

    This is the classical, non-learned counterpart of the learned solver: both
    see ``J`` and ``F`` and both stop after a fixed number of inner steps.
    """

    def __init__(self, iters: int = 50) -> None:
        if iters < 1:
            raise ValueError("iters must be >= 1")
        self.iters = iters
        self.name = f"cg{iters}"

    def solve(self, J: np.ndarray, F: np.ndarray) -> np.ndarray:
        Jt = np.swapaxes(J, -1, -2)
        rhs = -(Jt @ F[..., None])[..., 0]
        x = np.zeros_like(F)
        r = rhs.copy()
        d = r.copy()
        rr = np.sum(r * r, axis=-1)
        for _ in range(self.iters):
            Ad = (Jt @ (J @ d[..., None]))[..., 0]
            dAd = np.sum(d * Ad, axis=-1)
            alpha = np.where(dAd > 1e-300, rr / np.where(dAd > 1e-300, dAd, 1.0), 0.0)
            x = x + alpha[:, None] * d
            r = r - alpha[:, None] * Ad
            rr_new = np.sum(r * r, axis=-1)
            beta = np.where(rr > 1e-300, rr_new / np.where(rr > 1e-300, rr, 1.0), 0.0)
            d = r + beta[:, None] * d
            rr = rr_new
        return x


def relative_residual(J: np.ndarray, F: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """``||J dy + F|| / ||F||`` for each system in the batch."""
    res = (J @ dy[..., None])[..., 0] + F
    return np.linalg.norm(res, axis=-1) / np.maximum(np.linalg.norm(F, axis=-1), 1e-300)

"""Wrap a trained model in the ``NewtonSolver`` interface of ``ipm_lstm_attn.linsolve``."""
from __future__ import annotations

import numpy as np
import torch

from .nets import LearnedNewtonSolver


class LearnedSolver:
    def __init__(self, model: LearnedNewtonSolver, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.name = f"learned:{model.variant}"

    @torch.no_grad()
    def solve(self, J: np.ndarray, F: np.ndarray) -> np.ndarray:
        Jt = torch.as_tensor(J, dtype=torch.float32, device=self.device)
        Ft = torch.as_tensor(F, dtype=torch.float32, device=self.device)
        dy, _, _ = self.model(Jt, Ft)
        return dy.double().cpu().numpy()

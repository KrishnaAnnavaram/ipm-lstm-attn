"""Measure the effect of fp16 weight storage (needs torch).

The check stores every weight as float16, reads it back as float32 and runs the
same evaluation as the float32 model. The IPM residuals stay in float64 in
both runs. The output is a measurement, not a claim: it reports the change in
the KKT error and in the warm-start iterations.
"""
from __future__ import annotations

import copy
from typing import Sequence

import numpy as np
import torch

from ..config import IPMConfig
from ..datasets import Dataset
from ..evaluate import evaluate_solver, reference_objectives
from .adapter import LearnedSolver
from .nets import LearnedNewtonSolver


def fp16_roundtrip(model: LearnedNewtonSolver) -> LearnedNewtonSolver:
    clone = copy.deepcopy(model)
    with torch.no_grad():
        for p in clone.parameters():
            p.copy_(p.half().float())
    return clone.eval()


def precision_report(model: LearnedNewtonSolver, ds: Dataset, idx: Sequence[int], cfg: IPMConfig,
                     device: str = "cpu") -> dict:
    half = fp16_roundtrip(model)
    max_change = max(float((a.detach() - b.detach()).abs().max()) for a, b in zip(model.parameters(), half.parameters()))
    n_params = sum(p.numel() for p in model.parameters())
    f_star = reference_objectives(ds.instances(idx), cfg)
    r32 = evaluate_solver(ds, idx, LearnedSolver(model, device), cfg, f_star=f_star)
    r16 = evaluate_solver(ds, idx, LearnedSolver(half, device), cfg, f_star=f_star)
    d_kkt = r16.column("log10_kkt") - r32.column("log10_kkt")
    d_iters = r16.column("warm_iters") - r32.column("warm_iters")
    return {
        "parameters": n_params,
        "bytes_fp32": 4 * n_params,
        "bytes_fp16": 2 * n_params,
        "max_abs_weight_change": max_change,
        "fp32": {"log10_kkt_mean": r32.summary["log10_kkt"]["mean"], "warm_iters_mean": r32.summary["warm_iters"]["mean"]},
        "fp16": {"log10_kkt_mean": r16.summary["log10_kkt"]["mean"], "warm_iters_mean": r16.summary["warm_iters"]["mean"]},
        "delta_log10_kkt_max_abs": float(np.max(np.abs(d_kkt))),
        "delta_warm_iters_max_abs": float(np.max(np.abs(d_iters))),
    }

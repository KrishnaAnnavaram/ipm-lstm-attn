"""Like-for-like timing of the two-stage pipeline (approximate IPM, then warm-started solve).

Every number is a per-instance median over repeats, after warm-up calls.
Batched times are reported in their own rows with their batch size and are
never added to per-instance times.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .baselines import WarmStartBackend
from .config import BenchConfig, IPMConfig
from .evaluate import approximate
from .linsolve import NewtonSolver
from .problems import Instance
from .timing import environment, measure


def benchmark(insts: Sequence[Instance], solver: NewtonSolver, backend: WarmStartBackend, ipm_cfg: IPMConfig,
              bench_cfg: BenchConfig, device: str = "cpu") -> dict:
    insts = list(insts)[: bench_cfg.max_instances]
    k = len(insts)
    bs = min(bench_cfg.batch_size, k)
    sync = None
    if device.startswith("cuda"):  # pragma: no cover - GPU only
        import torch

        sync = torch.cuda.synchronize
    w, r = bench_cfg.warmup, bench_cfg.repeats

    approx_single = measure(lambda: [approximate([i], solver, ipm_cfg) for i in insts],
                            f"{solver.name} IPM", "per_instance", 1, k, w, r, sync)
    approx_batched = measure(lambda: approximate(insts[:bs], solver, ipm_cfg),
                             f"{solver.name} IPM", "batched", bs, bs, w, r, sync)
    y_approx = approximate(insts, solver, ipm_cfg).y
    cold = measure(lambda: [backend.solve(i) for i in insts], f"{backend.name} cold", "per_instance", 1, k, w, r)
    warm = measure(lambda: [backend.solve(i, y_approx[j]) for j, i in enumerate(insts)],
                   f"{backend.name} warm", "per_instance", 1, k, w, r)
    pipeline = approx_single.median_per_instance_s + warm.median_per_instance_s
    cold_iters = np.array([backend.solve(i).iterations for i in insts])
    warm_iters = np.array([backend.solve(i, y_approx[j]).iterations for j, i in enumerate(insts)])
    return {
        "environment": {**environment(), "device": device},
        "instances": k,
        "rows": [t.as_dict() for t in (approx_single, approx_batched, cold, warm)],
        "per_instance_pipeline_s": pipeline,
        "per_instance_cold_s": cold.median_per_instance_s,
        "speedup_per_instance": cold.median_per_instance_s / pipeline if pipeline > 0 else float("nan"),
        "cold_iters_mean": float(cold_iters.mean()),
        "warm_iters_mean": float(warm_iters.mean()),
    }

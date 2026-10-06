"""Evaluate a Newton solver: a fixed-budget IPM run, then a warm-started reference solve.

The procedure is the same for every solver (learned or classical):

1. Solve each instance to ``reference_tol`` with the exact IPM. This gives f*.
2. Run the IPM with the solver under test for ``outer_iters`` Newton steps.
3. Measure the objective gap, the violations and the KKT error of that point.
4. Solve each instance with the warm-start back-end twice: from the cold start
   and from the point of step 2. Count the iterations of both.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from . import ipm
from .baselines import ExactIPMBackend, WarmStartBackend
from .config import IPMConfig
from .datasets import Dataset
from .linsolve import NewtonSolver
from .metrics import solution_metrics, summarize
from .problems import Instance


@dataclass
class EvalReport:
    solver: str
    backend: str
    n_instances: int
    records: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    kkt_history: list[float] = field(default_factory=list)  # mean log10 KKT error per outer step

    def column(self, key: str) -> np.ndarray:
        return np.array([r[key] for r in self.records], dtype=float)

    def as_dict(self) -> dict:
        return {"solver": self.solver, "backend": self.backend, "n_instances": self.n_instances,
                "summary": self.summary, "kkt_history": self.kkt_history, "records": self.records}


def reference_objectives(insts: Sequence[Instance], cfg: IPMConfig) -> np.ndarray:
    ref = ExactIPMBackend(tol=cfg.reference_tol, max_iter=cfg.reference_max_iter, sigma=cfg.sigma, tau=cfg.tau)
    res = ref.solve_batch(insts)
    return np.array([inst.objective(res.y[i][: inst.n]) for i, inst in enumerate(insts)])


def approximate(insts: Sequence[Instance], solver: NewtonSolver, cfg: IPMConfig) -> ipm.IPMResult:
    settings = ipm.IPMSettings(sigma=cfg.sigma, tau=cfg.tau, max_iter=cfg.outer_iters,
                               tol=cfg.warm_tol, precondition=cfg.precondition)
    return ipm.run(list(insts), solver, settings, stop_at_tol=False)


def evaluate_solver(ds: Dataset, idx: Sequence[int], solver: NewtonSolver, cfg: IPMConfig,
                    backend: WarmStartBackend | None = None, f_star: np.ndarray | None = None) -> EvalReport:
    insts = ds.instances(idx)
    backend = backend or ExactIPMBackend(tol=cfg.warm_tol, max_iter=cfg.reference_max_iter,
                                         sigma=cfg.sigma, tau=cfg.tau)
    if f_star is None:
        f_star = reference_objectives(insts, cfg)
    approx = approximate(insts, solver, cfg)
    report = EvalReport(solver=solver.name, backend=backend.name, n_instances=len(insts))
    for k, inst in enumerate(insts):
        m = solution_metrics(inst, approx.y[k], float(f_star[k]))
        cold = backend.solve(inst)
        warm = backend.solve(inst, approx.y[k])
        report.records.append({
            "index": int(idx[k]),
            **m.as_dict(),
            "log10_kkt": float(np.log10(max(m.kkt_worst, 1e-16))),
            "cold_iters": cold.iterations,
            "warm_iters": warm.iterations,
            "iters_saved": cold.iterations - warm.iterations,
            "cold_s": cold.seconds,
            "warm_s": warm.seconds,
            "warm_converged": warm.converged,
            "cold_converged": cold.converged,
        })
    report.kkt_history = [float(np.mean(np.log10(np.maximum(h, 1e-16)))) for h in approx.history]
    for key in ("rel_gap", "eq_max", "ineq_max", "kkt_worst", "log10_kkt", "cold_iters", "warm_iters", "iters_saved"):
        report.summary[key] = summarize(report.column(key))
    report.summary["warm_converged_rate"] = float(np.mean([r["warm_converged"] for r in report.records]))
    report.summary["nonfinite_steps"] = approx.nonfinite_steps
    return report

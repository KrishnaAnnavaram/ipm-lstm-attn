"""Solution quality metrics and paired statistics."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import kkt
from .problems import Instance


@dataclass(frozen=True)
class SolutionMetrics:
    objective: float
    reference_objective: float
    rel_gap: float  # |f - f*| / max(1, |f*|)
    eq_max: float
    eq_mean: float
    ineq_max: float
    ineq_mean: float
    kkt_worst: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def solution_metrics(inst: Instance, y: np.ndarray, reference_objective: float) -> SolutionMetrics:
    lay = kkt.Layout.of(inst)
    x = y[lay.x]
    f = inst.objective(x)
    eq = inst.eq_violation(x)
    ineq = inst.ineq_violation(x)
    return SolutionMetrics(
        objective=f,
        reference_objective=reference_objective,
        rel_gap=abs(f - reference_objective) / max(1.0, abs(reference_objective)),
        eq_max=float(eq.max()) if eq.size else 0.0,
        eq_mean=float(eq.mean()) if eq.size else 0.0,
        ineq_max=float(ineq.max()) if ineq.size else 0.0,
        ineq_mean=float(ineq.mean()) if ineq.size else 0.0,
        kkt_worst=kkt.kkt_error(inst, y).worst,
    )


@dataclass(frozen=True)
class PairedDiff:
    """Mean of (treatment - control) over paired units with a percentile bootstrap CI."""

    mean: float
    ci_low: float
    ci_high: float
    n: int

    @property
    def significant(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0

    def as_dict(self) -> dict[str, float | int | bool]:
        return {**asdict(self), "significant": self.significant}


def paired_bootstrap(
    treatment: np.ndarray, control: np.ndarray, n_boot: int = 2000, level: float = 0.95, seed: int = 0
) -> PairedDiff:
    t = np.asarray(treatment, dtype=float)
    c = np.asarray(control, dtype=float)
    if t.shape != c.shape or t.ndim != 1 or t.size == 0:
        raise ValueError("treatment and control must be 1-D arrays of the same non-zero length")
    d = t - c
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return PairedDiff(float(d.mean()), float(lo), float(hi), int(d.size))


def summarize(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
    return {"mean": float(v.mean()), "median": float(med), "q1": float(q1), "q3": float(q3),
            "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}

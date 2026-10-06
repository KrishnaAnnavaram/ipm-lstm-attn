"""Primal-dual interior-point loop with a pluggable Newton solver.

The loop runs on a batch of instances of the same size. Each instance stops
on its own when its KKT error is below the tolerance.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from . import kkt
from .linsolve import NewtonSolver
from .problems import Instance


@dataclass(frozen=True)
class IPMSettings:
    sigma: float = 0.1  # centering: mu = sigma * mean(eta * s)
    tau: float = 0.995  # fraction-to-boundary factor
    max_iter: int = 100
    tol: float = 1e-6
    precondition: bool = True  # row equilibration of J dy = -F
    convexify: bool = True  # shift an indefinite Hessian block (nonconvex family)


@dataclass
class IPMResult:
    y: np.ndarray  # (B, N) final iterates
    iterations: np.ndarray  # (B,) Newton steps taken
    converged: np.ndarray  # (B,) bool
    errors: np.ndarray  # (B,) final worst KKT error
    history: list[np.ndarray] = field(default_factory=list)  # worst KKT error per iteration, (B,) each
    lin_residual: list[np.ndarray] = field(default_factory=list)  # ||J dy + F|| / ||F||, (B,) each
    seconds: float = 0.0
    nonfinite_steps: int = 0


def newton_system(
    inst: Instance, y: np.ndarray, sigma: float, precondition: bool = True, convexify: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Return (J, F) of the Newton system at ``y`` for mu = sigma * mean(eta * s)."""
    mu = sigma * kkt.complementarity(inst, y)
    F = kkt.residual(inst, y, mu)
    J = kkt.jacobian(inst, y)
    if convexify and inst.family == "nonconvex":
        lay = kkt.Layout.of(inst)
        hx = J[lay.x, lay.x]
        lam_min = float(np.linalg.eigvalsh(0.5 * (hx + hx.T))[0])
        if lam_min < 1e-8:
            J[lay.x, lay.x] = hx + (1e-8 - lam_min) * np.eye(inst.n)
    if precondition:
        J, F = kkt.row_equilibrate(J, F)
    return J, F


def step_length(v: np.ndarray, dv: np.ndarray, tau: float) -> float:
    """Largest alpha in (0, 1] with v + alpha * dv >= (1 - tau) * v."""
    neg = dv < 0
    if not np.any(neg):
        return 1.0
    return float(min(1.0, tau * np.min(-v[neg] / dv[neg])))


def take_step(inst: Instance, y: np.ndarray, dy: np.ndarray, tau: float) -> np.ndarray:
    """Apply a Newton step with separate primal and dual step lengths."""
    lay = kkt.Layout.of(inst)
    x, eta, s, lam = lay.split(y)
    dx, deta, ds, dlam = lay.split(dy)
    a_p = step_length(s, ds, tau) if inst.m else 1.0
    a_d = step_length(eta, deta, tau) if inst.m else 1.0
    return lay.join(x + a_p * dx, eta + a_d * deta, s + a_p * ds, lam + a_d * dlam)


def run(
    instances: Sequence[Instance],
    solver: NewtonSolver,
    settings: IPMSettings = IPMSettings(),
    y0: np.ndarray | None = None,
    stop_at_tol: bool = True,
) -> IPMResult:
    """Run the IPM on a batch. ``y0`` is a warm start; the default is the cold start."""
    if not instances:
        raise ValueError("no instances")
    size = kkt.Layout.of(instances[0]).size
    if any(kkt.Layout.of(i).size != size for i in instances):
        raise ValueError("all instances in a batch must have the same size")
    B = len(instances)
    y = np.stack([kkt.initial_point(i) for i in instances]) if y0 is None else np.array(y0, dtype=float, copy=True)
    if y.shape != (B, size):
        raise ValueError(f"y0 must have shape {(B, size)}")

    iters = np.zeros(B, dtype=int)
    errors = np.array([kkt.kkt_error(i, y[b]).worst for b, i in enumerate(instances)])
    done = errors <= settings.tol if stop_at_tol else np.zeros(B, dtype=bool)
    result = IPMResult(y=y, iterations=iters, converged=done.copy(), errors=errors)
    t0 = time.perf_counter()
    for _ in range(settings.max_iter):
        active = np.flatnonzero(~done)
        if active.size == 0:
            break
        systems = [newton_system(instances[b], y[b], settings.sigma, settings.precondition, settings.convexify) for b in active]
        J = np.stack([s[0] for s in systems])
        F = np.stack([s[1] for s in systems])
        dy = np.asarray(solver.solve(J, F), dtype=float)
        bad = ~np.all(np.isfinite(dy), axis=1)
        if np.any(bad):
            result.nonfinite_steps += int(bad.sum())
            dy[bad] = 0.0
        lin = np.full(B, np.nan)
        res = (J @ dy[..., None])[..., 0] + F
        lin[active] = np.linalg.norm(res, axis=1) / np.maximum(np.linalg.norm(F, axis=1), 1e-300)
        result.lin_residual.append(lin)
        for k, b in enumerate(active):
            y[b] = take_step(instances[b], y[b], dy[k], settings.tau)
            iters[b] += 1
            errors[b] = kkt.kkt_error(instances[b], y[b]).worst
        if stop_at_tol:
            done = errors <= settings.tol
        result.history.append(errors.copy())
    result.seconds = time.perf_counter() - t0
    result.y, result.iterations, result.errors = y, iters, errors
    result.converged = errors <= settings.tol
    return result

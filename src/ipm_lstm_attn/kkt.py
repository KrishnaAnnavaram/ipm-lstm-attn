"""Primal-dual KKT system of the barrier problem.

The unknown vector is ``y = [x, eta, s, lam]`` with sizes ``n, m, m, p``:

* ``x``   primal variables
* ``eta`` multipliers of the inequality constraints (kept > 0)
* ``s``   slacks with ``g(x) + s = 0`` (kept > 0)
* ``lam`` multipliers of the equality constraints

The residual for a barrier parameter ``mu`` is::

    F = [ grad f(x) + A^T lam + Jg(x)^T eta
          g(x) + s
          eta * s - mu
          A x - b ]

and ``J`` is its Jacobian with respect to ``y``. One IPM iteration solves the
Newton system ``J dy = -F``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .problems import Instance


@dataclass(frozen=True)
class Layout:
    """Index ranges of the four blocks inside ``y``."""

    n: int
    m: int
    p: int

    @classmethod
    def of(cls, inst: Instance) -> "Layout":
        return cls(inst.n, inst.m, inst.p_eq)

    @property
    def size(self) -> int:
        return self.n + 2 * self.m + self.p

    @property
    def x(self) -> slice:
        return slice(0, self.n)

    @property
    def eta(self) -> slice:
        return slice(self.n, self.n + self.m)

    @property
    def s(self) -> slice:
        return slice(self.n + self.m, self.n + 2 * self.m)

    @property
    def lam(self) -> slice:
        return slice(self.n + 2 * self.m, self.size)

    def split(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return y[self.x], y[self.eta], y[self.s], y[self.lam]

    def join(self, x: np.ndarray, eta: np.ndarray, s: np.ndarray, lam: np.ndarray) -> np.ndarray:
        return np.concatenate([x, eta, s, lam])


def initial_point(inst: Instance, slack_floor: float = 1.0) -> np.ndarray:
    """Standard cold start: x = 0, s = max(-g(0), floor), eta = 1, lam = 0."""
    lay = Layout.of(inst)
    x = np.zeros(inst.n)
    s = np.maximum(-inst.ineq(x), slack_floor)
    eta = np.ones(inst.m)
    lam = np.zeros(inst.p_eq)
    return lay.join(x, eta, s, lam)


def complementarity(inst: Instance, y: np.ndarray) -> float:
    lay = Layout.of(inst)
    _, eta, s, _ = lay.split(y)
    return float(np.mean(eta * s)) if inst.m else 0.0


def residual(inst: Instance, y: np.ndarray, mu: float) -> np.ndarray:
    lay = Layout.of(inst)
    x, eta, s, lam = lay.split(y)
    jg = inst.ineq_jacobian(x)
    r_dual = inst.gradient(x) + inst.A.T @ lam + jg.T @ eta
    r_ineq = inst.ineq(x) + s
    r_comp = eta * s - mu
    r_eq = inst.A @ x - inst.b
    return np.concatenate([r_dual, r_ineq, r_comp, r_eq])


def jacobian(inst: Instance, y: np.ndarray) -> np.ndarray:
    lay = Layout.of(inst)
    x, eta, s, _ = lay.split(y)
    n, m, p = lay.n, lay.m, lay.p
    J = np.zeros((lay.size, lay.size))
    jg = inst.ineq_jacobian(x)
    # dual residual rows
    J[lay.x, lay.x] = inst.hessian(x) + inst.ineq_hessian_sum(eta)
    J[lay.x, lay.eta] = jg.T
    J[lay.x, lay.lam] = inst.A.T
    # g(x) + s rows
    r1 = slice(n, n + m)
    J[r1, lay.x] = jg
    J[r1, lay.s] = np.eye(m)
    # complementarity rows
    r2 = slice(n + m, n + 2 * m)
    J[r2, lay.eta] = np.diag(s)
    J[r2, lay.s] = np.diag(eta)
    # equality rows
    r3 = slice(n + 2 * m, n + 2 * m + p)
    J[r3, lay.x] = inst.A
    return J


@dataclass(frozen=True)
class KKTError:
    """Optimality measures at one point (barrier parameter zero)."""

    dual: float  # max |grad L|
    primal_eq: float  # max |A x - b|
    primal_ineq: float  # max |g(x) + s|
    comp: float  # mean eta * s

    @property
    def worst(self) -> float:
        return max(self.dual, self.primal_eq, self.primal_ineq, self.comp)


def kkt_error(inst: Instance, y: np.ndarray) -> KKTError:
    lay = Layout.of(inst)
    r = residual(inst, y, 0.0)
    n, m = lay.n, lay.m

    def _max(v: np.ndarray) -> float:
        return float(np.max(np.abs(v))) if v.size else 0.0

    return KKTError(
        dual=_max(r[:n]),
        primal_eq=_max(r[n + 2 * m :]),
        primal_ineq=_max(r[n : n + m]),
        comp=complementarity(inst, y),
    )


def row_equilibrate(J: np.ndarray, F: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Scale each row of ``J dy = -F`` to unit 2-norm. The solution ``dy`` does not change."""
    norms = np.linalg.norm(J, axis=-1, keepdims=True)
    scale = 1.0 / np.maximum(norms, eps)
    return J * scale, F * scale[..., 0]


def structure_mask(J: np.ndarray) -> np.ndarray:
    """Boolean (..., N, N): entries k and l of ``dy`` interact if columns k and l of J share a row.

    This is the sparsity pattern of ``J^T J``. The diagonal is always True.
    Works for one matrix or for a batch of matrices.
    """
    nz = (np.abs(J) > 0).astype(np.float64)
    conn = (np.swapaxes(nz, -1, -2) @ nz) > 0
    return conn | np.eye(J.shape[-1], dtype=bool)

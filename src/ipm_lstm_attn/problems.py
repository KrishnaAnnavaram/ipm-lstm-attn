"""Problem families: objective, constraints and their derivatives for one instance.

Every family has the same shape:

    minimize    f(x)
    subject to  A x = b          (p linear equality constraints)
                g(x) <= 0        (m inequality constraints)

The equality right-hand side ``b`` changes from instance to instance. All other
matrices are shared by the instances of one dataset (the "RHS" setting of the
DC3 benchmark).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FAMILIES = ("qp", "qcqp", "nonconvex")


@dataclass(frozen=True)
class Instance:
    """One optimisation problem. Arrays are float64 and never modified."""

    family: str
    Q: np.ndarray  # (n, n) symmetric
    p: np.ndarray  # (n,)
    A: np.ndarray  # (p_eq, n)
    b: np.ndarray  # (p_eq,)
    G: np.ndarray  # (m, n)
    c: np.ndarray  # (m,)
    H: np.ndarray | None = field(default=None)  # (m, n, n) for qcqp only

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family {self.family!r}; use one of {FAMILIES}")
        n = self.Q.shape[0]
        if self.Q.shape != (n, n) or self.p.shape != (n,):
            raise ValueError("Q must be (n, n) and p must be (n,)")
        if self.A.shape[1] != n or self.b.shape != (self.A.shape[0],):
            raise ValueError("A must be (p, n) and b must be (p,)")
        if self.G.shape[1] != n or self.c.shape != (self.G.shape[0],):
            raise ValueError("G must be (m, n) and c must be (m,)")
        if self.family == "qcqp":
            if self.H is None or self.H.shape != (self.G.shape[0], n, n):
                raise ValueError("qcqp needs H with shape (m, n, n)")
        for arr in (self.Q, self.p, self.A, self.b, self.G, self.c):
            arr.setflags(write=False)

    # sizes -----------------------------------------------------------------
    @property
    def n(self) -> int:
        return self.Q.shape[0]

    @property
    def m(self) -> int:
        return self.G.shape[0]

    @property
    def p_eq(self) -> int:
        return self.A.shape[0]

    # objective -------------------------------------------------------------
    def objective(self, x: np.ndarray) -> float:
        quad = 0.5 * x @ self.Q @ x
        if self.family == "nonconvex":
            return float(quad + self.p @ np.sin(x))
        return float(quad + self.p @ x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        if self.family == "nonconvex":
            return self.Q @ x + self.p * np.cos(x)
        return self.Q @ x + self.p

    def hessian(self, x: np.ndarray) -> np.ndarray:
        if self.family == "nonconvex":
            return self.Q - np.diag(self.p * np.sin(x))
        return np.array(self.Q, copy=True)

    # inequality constraints g(x) <= 0 -------------------------------------
    def ineq(self, x: np.ndarray) -> np.ndarray:
        lin = self.G @ x - self.c
        if self.family == "qcqp":
            assert self.H is not None
            lin = lin + 0.5 * np.einsum("i,kij,j->k", x, self.H, x)
        return lin

    def ineq_jacobian(self, x: np.ndarray) -> np.ndarray:
        if self.family == "qcqp":
            assert self.H is not None
            return self.G + np.einsum("kij,j->ki", self.H, x)
        return np.array(self.G, copy=True)

    def ineq_hessian_sum(self, eta: np.ndarray) -> np.ndarray:
        """Return sum_k eta_k * Hessian(g_k). Zero for linear constraints."""
        if self.family == "qcqp":
            assert self.H is not None
            return np.einsum("k,kij->ij", eta, self.H)
        return np.zeros((self.n, self.n))

    # feasibility -----------------------------------------------------------
    def eq_violation(self, x: np.ndarray) -> np.ndarray:
        return np.abs(self.A @ x - self.b)

    def ineq_violation(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(self.ineq(x), 0.0)

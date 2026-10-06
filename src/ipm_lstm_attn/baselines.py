"""Reference solvers and warm-start back-ends.

* ``ExactIPMBackend``: the IPM of this package with the exact (LU) Newton
  solver. It needs only numpy, so the offline demo and CI use it.
* ``IpoptBackend``: IPOPT through ``cyipopt`` (optional extra ``ipopt``).
  IPOPT needs a system library. Install it with conda-forge.
* ``scipy_reference``: SciPy ``trust-constr``, an independent check for tests.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from . import ipm, kkt
from .linsolve import DirectSolver
from .problems import Instance


@dataclass(frozen=True)
class SolveStats:
    iterations: int
    objective: float
    seconds: float
    converged: bool
    y: np.ndarray


class WarmStartBackend(Protocol):
    name: str

    def solve(self, inst: Instance, y0: np.ndarray | None = None) -> SolveStats:  # pragma: no cover
        ...


class ExactIPMBackend:
    name = "exact_ipm"

    def __init__(self, tol: float = 1e-6, max_iter: int = 200, sigma: float = 0.1, tau: float = 0.995) -> None:
        self.settings = ipm.IPMSettings(sigma=sigma, tau=tau, max_iter=max_iter, tol=tol)
        self.solver = DirectSolver()

    def solve(self, inst: Instance, y0: np.ndarray | None = None) -> SolveStats:
        t0 = time.perf_counter()
        res = ipm.run([inst], self.solver, self.settings, None if y0 is None else y0[None, :])
        dt = time.perf_counter() - t0
        lay = kkt.Layout.of(inst)
        y = res.y[0]
        return SolveStats(int(res.iterations[0]), inst.objective(y[lay.x]), dt, bool(res.converged[0]), y)

    def solve_batch(self, insts: Sequence[Instance], y0: np.ndarray | None = None) -> ipm.IPMResult:
        return ipm.run(list(insts), self.solver, self.settings, y0)


class IpoptBackend:
    """IPOPT with an optional primal-dual warm start. Import of ``cyipopt`` is lazy."""

    name = "ipopt"

    def __init__(self, tol: float = 1e-6, max_iter: int = 500, print_level: int = 0) -> None:
        try:
            import cyipopt  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on the system
            raise ImportError("IpoptBackend needs the 'ipopt' extra and the IPOPT library (conda-forge cyipopt)") from exc
        self.tol, self.max_iter, self.print_level = tol, max_iter, print_level

    def solve(self, inst: Instance, y0: np.ndarray | None = None) -> SolveStats:  # pragma: no cover - needs IPOPT
        import cyipopt

        lay = kkt.Layout.of(inst)
        n, m, p = inst.n, inst.m, inst.p_eq

        class _Prob:
            def objective(self, x):
                return inst.objective(x)

            def gradient(self, x):
                return inst.gradient(x)

            def constraints(self, x):
                return np.concatenate([inst.A @ x - inst.b, inst.ineq(x)])

            def jacobian(self, x):
                return np.vstack([inst.A, inst.ineq_jacobian(x)]).ravel()

            def hessianstructure(self):
                return np.tril_indices(n)

            def hessian(self, x, lagrange, obj_factor):
                h = obj_factor * inst.hessian(x) + inst.ineq_hessian_sum(lagrange[p:])
                return h[np.tril_indices(n)]

            def intermediate(self, *args):
                self.iters = args[1]

        prob_obj = _Prob()
        nlp = cyipopt.Problem(n=n, m=p + m, problem_obj=prob_obj, lb=[-1e20] * n, ub=[1e20] * n,
                              cl=np.concatenate([np.zeros(p), np.full(m, -1e20)]),
                              cu=np.zeros(p + m))
        nlp.add_option("tol", self.tol)
        nlp.add_option("max_iter", self.max_iter)
        nlp.add_option("print_level", self.print_level)
        if y0 is None:
            x0 = np.zeros(n)
            kwargs = {}
        else:
            x0, eta, _, lam = lay.split(y0)
            nlp.add_option("warm_start_init_point", "yes")
            nlp.add_option("warm_start_bound_push", 1e-9)
            nlp.add_option("warm_start_mult_bound_push", 1e-9)
            kwargs = {"lagrange": np.concatenate([lam, eta])}
        t0 = time.perf_counter()
        x, info = nlp.solve(x0, **kwargs)
        dt = time.perf_counter() - t0
        mult = info["mult_g"]
        s = np.maximum(-inst.ineq(x), 0.0)
        y = lay.join(x, mult[p:], s, mult[:p])
        return SolveStats(int(getattr(prob_obj, "iters", -1)), float(info["obj_val"]), dt, info["status"] == 0, y)


def make_backend(name: str, tol: float, max_iter: int, sigma: float = 0.1, tau: float = 0.995) -> WarmStartBackend:
    if name == "exact_ipm":
        return ExactIPMBackend(tol=tol, max_iter=max_iter, sigma=sigma, tau=tau)
    if name == "ipopt":
        return IpoptBackend(tol=tol, max_iter=max_iter)
    raise ValueError(f"unknown backend {name!r}; use 'exact_ipm' or 'ipopt'")


def scipy_reference(inst: Instance, x0: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Solve with SciPy ``trust-constr``. Slow, for tests on small instances only."""
    from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize

    cons = [LinearConstraint(inst.A, inst.b, inst.b)]
    cons.append(NonlinearConstraint(inst.ineq, -np.inf, 0.0, jac=inst.ineq_jacobian))
    res = minimize(inst.objective, np.zeros(inst.n) if x0 is None else x0, jac=inst.gradient,
                   hess=inst.hessian if inst.family != "qcqp" else None,
                   constraints=cons, method="trust-constr",
                   options={"gtol": 1e-10, "xtol": 1e-12, "maxiter": 5000})
    return res.x, float(res.fun)

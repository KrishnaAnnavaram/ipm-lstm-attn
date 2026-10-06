import numpy as np
import pytest

from ipm_lstm_attn import baselines, datasets, ipm, kkt
from ipm_lstm_attn.linsolve import CGNormalSolver, DirectSolver, relative_residual


def test_exact_ipm_matches_scipy(family):
    ds = datasets.generate(family, 8, 3, 4, 3, seed=11)
    inst = ds.instance(0)
    res = ipm.run([inst], DirectSolver(), ipm.IPMSettings(tol=1e-9, max_iter=200))
    assert res.converged[0]
    f_ipm = inst.objective(res.y[0][: inst.n])
    _, f_ref = baselines.scipy_reference(inst)
    assert f_ipm == pytest.approx(f_ref, abs=1e-5)


def test_ipm_solution_is_feasible():
    ds = datasets.generate("qp", 10, 4, 5, 4, seed=3)
    res = ipm.run(ds.instances(range(4)), DirectSolver(), ipm.IPMSettings(tol=1e-8))
    for b in range(4):
        inst = ds.instance(b)
        x = res.y[b][: inst.n]
        assert inst.eq_violation(x).max() < 1e-7
        assert inst.ineq_violation(x).max() < 1e-7


def test_step_length_keeps_positivity():
    v = np.array([1.0, 0.5, 2.0])
    dv = np.array([-4.0, 1.0, -1.0])
    a = ipm.step_length(v, dv, 0.99)
    assert 0 < a <= 1
    assert np.all(v + a * dv > 0)
    assert ipm.step_length(v, np.ones(3), 0.99) == 1.0


def test_take_step_keeps_interior_for_any_direction():
    inst = datasets.generate("qcqp", 6, 2, 3, 3, seed=4).instance(0)
    y = kkt.initial_point(inst)
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = ipm.take_step(inst, y, rng.normal(scale=10, size=y.size), 0.995)
        _, eta, s, _ = kkt.Layout.of(inst).split(y)
        assert np.all(eta > 0) and np.all(s > 0)


def test_cg_normal_converges_with_large_budget():
    rng = np.random.default_rng(1)
    J = rng.normal(size=(3, 6, 6)) + 6 * np.eye(6)
    F = rng.normal(size=(3, 6))
    dy = CGNormalSolver(60).solve(J, F)
    assert np.all(relative_residual(J, F, dy) < 1e-8)
    with pytest.raises(ValueError):
        CGNormalSolver(0)


def test_nonfinite_direction_is_not_applied():
    class Broken:
        name = "broken"

        def solve(self, J, F):
            return np.full_like(F, np.nan)

    inst = datasets.generate("qp", 5, 2, 2, 3, seed=0).instance(0)
    res = ipm.run([inst], Broken(), ipm.IPMSettings(max_iter=2), stop_at_tol=False)
    assert res.nonfinite_steps == 2
    assert np.all(np.isfinite(res.y))


def test_warm_start_from_solution_needs_fewer_iterations():
    inst = datasets.generate("qp", 10, 4, 5, 3, seed=8).instance(1)
    backend = baselines.ExactIPMBackend(tol=1e-6)
    cold = backend.solve(inst)
    near = ipm.run([inst], DirectSolver(), ipm.IPMSettings(max_iter=cold.iterations - 2), stop_at_tol=False).y[0]
    warm = backend.solve(inst, near)
    assert cold.converged and warm.converged
    assert warm.iterations < cold.iterations


def test_batch_with_mixed_sizes_is_rejected():
    a = datasets.generate("qp", 5, 2, 2, 3, seed=0).instance(0)
    b = datasets.generate("qp", 6, 2, 2, 3, seed=0).instance(0)
    with pytest.raises(ValueError):
        ipm.run([a, b], DirectSolver())


def test_unknown_backend():
    with pytest.raises(ValueError):
        baselines.make_backend("gurobi", 1e-6, 10)

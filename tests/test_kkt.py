import numpy as np
import pytest

from ipm_lstm_attn import datasets, kkt
from ipm_lstm_attn.problems import Instance


def _point(inst, rng):
    lay = kkt.Layout.of(inst)
    return lay.join(rng.normal(size=inst.n), rng.uniform(0.5, 2, inst.m), rng.uniform(0.5, 2, inst.m),
                    rng.normal(size=inst.p_eq))


def test_jacobian_matches_finite_differences(family):
    ds = datasets.generate(family, 7, 3, 4, 5, seed=1)
    inst = ds.instance(0)
    rng = np.random.default_rng(0)
    y = _point(inst, rng)
    J = kkt.jacobian(inst, y)
    eps = 1e-6
    num = np.zeros_like(J)
    for k in range(y.size):
        e = np.zeros_like(y)
        e[k] = eps
        num[:, k] = (kkt.residual(inst, y + e, 0.3) - kkt.residual(inst, y - e, 0.3)) / (2 * eps)
    assert np.allclose(J, num, atol=1e-6)


def test_layout_round_trip():
    lay = kkt.Layout(4, 2, 1)
    y = np.arange(lay.size, dtype=float)
    assert np.array_equal(lay.join(*lay.split(y)), y)
    assert lay.size == 9


def test_initial_point_is_interior(family):
    inst = datasets.generate(family, 6, 2, 3, 3, seed=2).instance(0)
    _, eta, s, _ = kkt.Layout.of(inst).split(kkt.initial_point(inst))
    assert np.all(eta > 0) and np.all(s > 0)


def test_row_equilibration_keeps_solution():
    rng = np.random.default_rng(3)
    J = rng.normal(size=(5, 5)) * rng.uniform(0.01, 100, size=(5, 1))
    F = rng.normal(size=5)
    Js, Fs = kkt.row_equilibrate(J, F)
    assert np.allclose(np.linalg.norm(Js, axis=1), 1.0)
    assert np.allclose(np.linalg.solve(J, -F), np.linalg.solve(Js, -Fs))


def test_structure_mask_is_pattern_of_jtj():
    J = np.array([[1.0, 0, 0], [0, 2.0, 3.0], [0, 0, 0]])
    m = kkt.structure_mask(J)
    assert m.tolist() == [[True, False, False], [False, True, True], [False, True, True]]
    batch = kkt.structure_mask(np.stack([J, J]))
    assert batch.shape == (2, 3, 3)


def test_instance_rejects_bad_shapes():
    with pytest.raises(ValueError):
        Instance("qp", np.eye(2), np.zeros(3), np.ones((1, 2)), np.ones(1), np.ones((1, 2)), np.ones(1))
    with pytest.raises(ValueError):
        Instance("lp", np.eye(2), np.zeros(2), np.ones((1, 2)), np.ones(1), np.ones((1, 2)), np.ones(1))


def test_instance_arrays_are_read_only():
    inst = datasets.generate("qp", 5, 2, 2, 3, seed=0).instance(0)
    with pytest.raises(ValueError):
        inst.b[0] = 1.0

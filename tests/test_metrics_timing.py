import numpy as np
import pytest

from ipm_lstm_attn import datasets, ipm
from ipm_lstm_attn.linsolve import DirectSolver
from ipm_lstm_attn.metrics import paired_bootstrap, solution_metrics, summarize
from ipm_lstm_attn.timing import environment, measure


def test_solution_metrics_at_optimum():
    inst = datasets.generate("qp", 8, 3, 3, 3, seed=2).instance(0)
    res = ipm.run([inst], DirectSolver(), ipm.IPMSettings(tol=1e-9))
    f = inst.objective(res.y[0][: inst.n])
    m = solution_metrics(inst, res.y[0], f)
    assert m.rel_gap == 0.0 and m.eq_max < 1e-8 and m.kkt_worst < 1e-8


def test_paired_bootstrap_detects_a_shift():
    rng = np.random.default_rng(0)
    c = rng.normal(size=200)
    d = paired_bootstrap(c - 1.0 + rng.normal(scale=0.1, size=200), c)
    assert d.significant and d.ci_high < 0 and d.mean == pytest.approx(-1.0, abs=0.05)
    same = paired_bootstrap(c + rng.normal(scale=0.1, size=200), c)
    assert same.ci_low < 0 < same.ci_high


def test_paired_bootstrap_rejects_unpaired():
    with pytest.raises(ValueError):
        paired_bootstrap(np.ones(3), np.ones(4))


def test_summarize():
    s = summarize(np.arange(5))
    assert s["median"] == 2 and s["n"] == 5


def test_measure_excludes_warmup_and_counts_repeats():
    calls = []
    t = measure(lambda: calls.append(1), "x", "per_instance", 1, 4, warmup=3, repeats=5)
    assert len(calls) == 8 and len(t.samples_s) == 5
    assert t.median_per_instance_s >= 0 and t.as_dict()["mode"] == "per_instance"
    with pytest.raises(ValueError):
        measure(lambda: None, "x", "mixed", 1, 1)


def test_environment_records_versions():
    env = environment()
    assert "numpy" in env and "torch" in env

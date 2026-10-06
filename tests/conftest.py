import pytest

from ipm_lstm_attn import datasets
from ipm_lstm_attn.config import ExperimentConfig

TINY = {
    "name": "tiny",
    "data": {"family": "qp", "n_var": 6, "n_eq": 3, "n_ineq": 3, "n_instances": 30, "seed": 5,
             "val_frac": 0.2, "test_frac": 0.2},
    "ipm": {"outer_iters": 3, "reference_max_iter": 100},
    "model": {"variant": "lstm1", "hidden_dim": 8, "d_model": 4, "heads": 2, "inner_steps": 3},
    "train": {"epochs": 2, "batch_size": 8, "patience": 2},
    "ablation": {"variants": ["lstm1", "lstm2"], "seeds": [0, 1], "baseline": "lstm1", "bootstrap": 200},
    "bench": {"warmup": 1, "repeats": 2, "batch_size": 4, "max_instances": 4},
}


@pytest.fixture
def tiny_cfg():
    return ExperimentConfig.model_validate(TINY)


@pytest.fixture(params=["qp", "qcqp", "nonconvex"])
def family(request):
    return request.param


@pytest.fixture
def tiny_ds(tiny_cfg):
    d = tiny_cfg.data
    ds = datasets.generate(d.family, d.n_var, d.n_eq, d.n_ineq, d.n_instances, d.seed)
    return ds, datasets.split_indices(len(ds), d.val_frac, d.test_frac, d.split_seed)

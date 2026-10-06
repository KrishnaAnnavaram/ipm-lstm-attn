import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from ipm_lstm_attn import datasets
from ipm_lstm_attn.config import ExperimentConfig, load_config, resolve_device


def test_generation_is_seeded():
    a = datasets.generate("qp", 6, 3, 3, 10, seed=4)
    b = datasets.generate("qp", 6, 3, 3, 10, seed=4)
    c = datasets.generate("qp", 6, 3, 3, 10, seed=5)
    assert datasets.content_hash(a) == datasets.content_hash(b) != datasets.content_hash(c)


def test_every_generated_instance_is_strictly_feasible(family):
    ds = datasets.generate(family, 8, 3, 4, 20, seed=1)
    pinv = np.linalg.pinv(ds.A)
    for i in range(len(ds)):
        inst = ds.instance(i)
        assert np.all(inst.ineq(pinv @ inst.b) < 0)


def test_save_is_immutable(tmp_path):
    ds = datasets.generate("qp", 6, 3, 3, 10, seed=4)
    path = datasets.save(ds, tmp_path)
    before = path.read_bytes()
    assert datasets.save(ds, tmp_path) == path  # same content: no-op
    assert path.read_bytes() == before
    manifest = path.with_suffix(".json")
    meta = json.loads(manifest.read_text())
    meta["sha256"] = "0" * 64
    manifest.write_text(json.dumps(meta))
    with pytest.raises(datasets.ImmutableDataError):
        datasets.save(ds, tmp_path)


def test_load_validates_and_is_read_only(tmp_path):
    ds = datasets.generate("qcqp", 6, 3, 3, 10, seed=4)
    loaded = datasets.load(datasets.save(ds, tmp_path))
    assert np.array_equal(loaded.b, ds.b) and np.array_equal(loaded.H, ds.H)
    with pytest.raises(ValueError):
        loaded.b[0, 0] = 9.0


def test_load_accepts_legacy_rhs_name_without_rewriting(tmp_path):
    ds = datasets.generate("qp", 6, 3, 3, 10, seed=4)
    path = tmp_path / "legacy.npz"
    np.savez(path, Q=ds.Q, p=ds.p, A=ds.A, G=ds.G, c=ds.c, X=ds.b[..., None])
    path.with_suffix(".json").write_text(json.dumps(ds.meta))
    before = path.read_bytes()
    loaded = datasets.load(path)
    assert np.array_equal(loaded.b, ds.b)
    assert path.read_bytes() == before


def test_load_rejects_bad_schema(tmp_path):
    ds = datasets.generate("qp", 6, 3, 3, 10, seed=4)
    path = tmp_path / "bad.npz"
    np.savez(path, Q=ds.Q, p=ds.p, A=ds.A, G=ds.G[:2], c=ds.c, b=ds.b)
    path.with_suffix(".json").write_text(json.dumps(ds.meta))
    with pytest.raises(datasets.SchemaError):
        datasets.load(path)
    np.savez(path, Q=ds.Q, p=ds.p, A=ds.A, G=ds.G, c=ds.c)
    with pytest.raises(datasets.SchemaError):
        datasets.load(path)


def test_load_detects_tampering(tmp_path):
    ds = datasets.generate("qp", 6, 3, 3, 10, seed=4)
    path = datasets.save(ds, tmp_path)
    np.savez(path, Q=ds.Q, p=ds.p, A=ds.A, G=ds.G, c=ds.c, b=ds.b + 1.0)
    with pytest.raises(datasets.SchemaError):
        datasets.load(path)


def test_split_is_disjoint_and_complete():
    sp = datasets.split_indices(100, 0.1, 0.2, seed=3)
    allidx = np.concatenate([sp.train, sp.val, sp.test])
    assert len(set(allidx)) == 100 and sorted(allidx) == list(range(100))
    assert len(sp.test) == 20 and len(sp.val) == 10
    again = datasets.split_indices(100, 0.1, 0.2, seed=3)
    assert np.array_equal(sp.test, again.test)


def test_split_instance_uses_one_index_for_all_arrays():
    ds = datasets.generate("qp", 6, 3, 3, 10, seed=4)
    inst = ds.instance(7)
    assert np.array_equal(inst.b, ds.b[7])


def test_unknown_config_key_is_an_error(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[model]\nvariant = "lstm1"\nuse_self_attention = true\n')
    with pytest.raises(ValidationError):
        load_config(p)
    p.write_text('[ipm]\nsolver = "ipopt"\n')
    with pytest.raises(ValidationError):
        load_config(p)


def test_config_validation_rules():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"model": {"variant": "transformer"}})
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"data": {"n_var": 5, "n_eq": 5}})
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"model": {"d_model": 10, "heads": 4}})
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"ablation": {"variants": ["lstm2"], "baseline": "lstm1"}})


def test_shipped_configs_are_valid():
    paths = sorted((Path(__file__).parents[1] / "configs").glob("*.toml"))
    assert len(paths) >= 4
    for path in paths:
        assert load_config(path).name


def test_device_auto_never_hard_codes_a_gpu_index():
    assert resolve_device("auto") in ("cpu", "cuda")
    assert resolve_device("cpu") == "cpu"


def test_dotenv_loads_only_known_unset_variables(tmp_path, monkeypatch):
    from ipm_lstm_attn.config import load_dotenv, paths_from_env

    monkeypatch.delenv("IPM_LSTM_ATTN_DATA_DIR", raising=False)
    monkeypatch.setenv("IPM_LSTM_ATTN_DEVICE", "cpu")
    env = tmp_path / ".env"
    env.write_text("# comment\nIPM_LSTM_ATTN_DATA_DIR=/tmp/d\nIPM_LSTM_ATTN_DEVICE=cuda\nOTHER=1\n")
    assert load_dotenv(env) == ["IPM_LSTM_ATTN_DATA_DIR"]
    assert str(paths_from_env().data_dir).replace("\\", "/").endswith("/tmp/d")
    assert load_dotenv(tmp_path / "missing.env") == []

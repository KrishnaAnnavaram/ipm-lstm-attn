import json

import pytest

from ipm_lstm_attn import cli
from ipm_lstm_attn.baselines import ExactIPMBackend
from ipm_lstm_attn.bench import benchmark
from ipm_lstm_attn.evaluate import evaluate_solver
from ipm_lstm_attn.linsolve import CGNormalSolver, DirectSolver

CFG = """name = "clitest"
[data]
n_var = 6
n_eq = 3
n_ineq = 3
n_instances = 20
val_frac = 0.2
test_frac = 0.2
[ipm]
outer_iters = 3
[model]
inner_steps = 4
"""


def test_evaluate_classical_solvers(tiny_cfg, tiny_ds):
    ds, split = tiny_ds
    exact = evaluate_solver(ds, split.test, DirectSolver(), tiny_cfg.ipm)
    cg = evaluate_solver(ds, split.test, CGNormalSolver(2), tiny_cfg.ipm)
    assert exact.n_instances == len(split.test)
    assert len(exact.kkt_history) == tiny_cfg.ipm.outer_iters
    # same outer budget: the exact Newton direction ends closer to the optimum
    assert exact.summary["log10_kkt"]["mean"] < cg.summary["log10_kkt"]["mean"]
    assert exact.summary["warm_converged_rate"] == 1.0


def test_benchmark_keeps_modes_apart(tiny_cfg, tiny_ds):
    ds, split = tiny_ds
    out = benchmark(ds.instances(split.test), CGNormalSolver(3), ExactIPMBackend(), tiny_cfg.ipm, tiny_cfg.bench)
    modes = [(r["mode"], r["batch_size"]) for r in out["rows"]]
    assert modes == [("per_instance", 1), ("batched", 4), ("per_instance", 1), ("per_instance", 1)]
    assert out["per_instance_pipeline_s"] > 0 and out["environment"]["device"] == "cpu"


def test_cli_generate_check_and_baseline(tmp_path, monkeypatch):
    monkeypatch.setenv("IPM_LSTM_ATTN_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IPM_LSTM_ATTN_RESULTS_DIR", str(tmp_path / "results"))
    cfg = tmp_path / "c.toml"
    cfg.write_text(CFG)
    assert cli.main(["check-config", str(cfg)]) == 0
    assert cli.main(["generate", "--n-var", "5", "--n-eq", "2", "--n-ineq", "2", "--n-instances", "10"]) == 0
    assert cli.main(["baseline", "--config", str(cfg), "--solver", "cg", "--cg-iters", "3"]) == 0
    out = json.loads((tmp_path / "results" / "clitest_cg3_test.json").read_text())
    assert out["solver"] == "cg3" and out["n_instances"] == 4
    assert cli.main(["baseline", "--config", str(cfg)]) == 0  # second run reuses the immutable dataset


def test_cli_bench_without_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("IPM_LSTM_ATTN_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IPM_LSTM_ATTN_RESULTS_DIR", str(tmp_path / "results"))
    cfg = tmp_path / "c.toml"
    cfg.write_text(CFG + "[bench]\nwarmup = 0\nrepeats = 1\nmax_instances = 2\n")
    assert cli.main(["bench", "--config", str(cfg)]) == 0
    assert (tmp_path / "results" / "clitest_cg4_bench.json").exists()


def test_cli_rejects_unknown_flag():
    with pytest.raises(SystemExit):
        cli.main(["baseline", "--config", "x.toml", "--solver_typo", "ipopt"])


def test_cli_demo_without_torch(tmp_path, capsys):
    assert cli.main(["demo", "--no-torch", "--out-dir", str(tmp_path)]) == 0
    assert "cg8" in capsys.readouterr().out

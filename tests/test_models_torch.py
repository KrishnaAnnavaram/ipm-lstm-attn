"""Tests of the learned solvers. They skip when torch is not installed (the CI installs only the dev extra)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ipm_lstm_attn import kkt  # noqa: E402
from ipm_lstm_attn.config import ModelConfig  # noqa: E402
from ipm_lstm_attn.evaluate import evaluate_solver  # noqa: E402
from ipm_lstm_attn.models.adapter import LearnedSolver  # noqa: E402
from ipm_lstm_attn.models.nets import AttentionBlock, MessagePassingBlock, build_model, structure_mask  # noqa: E402
from ipm_lstm_attn.registry import VARIANTS  # noqa: E402
from ipm_lstm_attn.seeding import set_seed  # noqa: E402


def _system(B=3, N=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    J = torch.randn(B, N, N, generator=g) + 3 * torch.eye(N)
    F = torch.randn(B, N, generator=g)
    return J, F


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_each_variant_builds_what_its_name_says(variant):
    m = build_model(ModelConfig(variant=variant, d_model=8, heads=2, inner_steps=2))
    v = VARIANTS[variant]
    assert m.num_layers == v.num_layers
    has_attn = any(isinstance(x, AttentionBlock) for x in m.modules())
    has_gnn = any(isinstance(x, MessagePassingBlock) for x in m.modules())
    assert has_attn == (v.block in ("attention", "masked_attention"))
    assert has_gnn == (v.block == "gnn")
    dy, loss, rel = m(*_system())
    assert dy.shape == (3, 7) and loss.ndim == 0 and rel.shape == (3,)


def test_dropout_is_defined_once_and_used_between_layers():
    torch.manual_seed(0)
    m = build_model(ModelConfig(variant="lstm2", dropout=0.5, inner_steps=3))
    assert sum(isinstance(x, torch.nn.Dropout) for x in m.modules()) == 1
    J, F = _system()
    m.train()
    torch.manual_seed(1)
    a = m(J, F)[1]
    torch.manual_seed(2)
    b = m(J, F)[1]
    assert not torch.allclose(a, b)  # dropout is active in training mode
    m.eval()
    assert torch.allclose(m(J, F)[1], m(J, F)[1])


def test_seed_fixes_initial_weights():
    cfg = ModelConfig(variant="lstm2_attn", d_model=8, heads=2)
    set_seed(3)
    a = build_model(cfg).state_dict()
    set_seed(3)
    b = build_model(cfg).state_dict()
    set_seed(4)
    c = build_model(cfg).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not all(torch.equal(a[k], c[k]) for k in a)


def test_structure_mask_matches_numpy():
    J = np.zeros((1, 4, 4))
    J[0, 0, 0] = J[0, 1, 1] = J[0, 1, 2] = J[0, 3, 3] = 1.0
    t = structure_mask(torch.as_tensor(J))
    assert np.array_equal(t.numpy(), kkt.structure_mask(J))


def test_masked_attention_ignores_unconnected_tokens():
    torch.manual_seed(0)
    block = AttentionBlock(4, 2, masked=True).eval()
    h = torch.randn(1, 4, 4)
    allowed = torch.eye(4, dtype=torch.bool).unsqueeze(0)
    allowed[0, 0, 1] = allowed[0, 1, 0] = True
    out1 = block(h, allowed)
    h2 = h.clone()
    h2[0, 3] += 5.0  # token 3 is not connected to tokens 0 and 1
    out2 = block(h2, allowed)
    assert torch.allclose(out1[0, :2], out2[0, :2], atol=1e-6)


def test_solver_is_scale_equivariant():
    m = build_model(ModelConfig(variant="lstm1", inner_steps=4)).eval()
    J, F = _system()
    with torch.no_grad():
        a = m(J, F)[0]
        b = m(J, 1000.0 * F)[0]
    assert torch.allclose(1000.0 * a, b, rtol=1e-4, atol=1e-4)


def test_returned_iterate_is_never_worse_than_zero():
    m = build_model(ModelConfig(variant="lstm2_gnn", d_model=8, heads=2, inner_steps=3)).eval()
    _, _, rel = m(*_system())
    assert torch.all(rel <= 1.0 + 1e-6)


def test_training_saves_and_reloads_checkpoint(tiny_cfg, tiny_ds, tmp_path):
    from ipm_lstm_attn.models.train import load_checkpoint, train

    ds, split = tiny_ds
    res = train(tiny_cfg, ds, split, "cpu", tmp_path / "new" / "ckpt", log=lambda s: None)
    assert res.checkpoint is not None and res.checkpoint.exists()
    assert res.checkpoint.with_suffix(".json").exists()
    assert len(res.history) >= 1 and np.isfinite(res.best_val)
    again = load_checkpoint(res.checkpoint)
    J, F = _system(N=kkt.Layout.of(ds.instance(0)).size)
    with torch.no_grad():
        assert torch.allclose(res.model(J, F)[0], again(J, F)[0])


def test_training_is_reproducible(tiny_cfg, tiny_ds):
    from ipm_lstm_attn.models.train import train

    ds, split = tiny_ds
    a = train(tiny_cfg, ds, split, log=lambda s: None)
    b = train(tiny_cfg, ds, split, log=lambda s: None)
    assert [h["train_loss"] for h in a.history] == [h["train_loss"] for h in b.history]


def test_learned_solver_runs_in_the_ipm(tiny_cfg, tiny_ds):
    ds, split = tiny_ds
    model = build_model(tiny_cfg.model)
    rep = evaluate_solver(ds, split.test, LearnedSolver(model), tiny_cfg.ipm)
    assert rep.solver == "learned:lstm1" and rep.summary["warm_converged_rate"] == 1.0


def test_precision_report_measures_fp16(tiny_cfg, tiny_ds):
    from ipm_lstm_attn.models.precision import fp16_roundtrip, precision_report

    ds, split = tiny_ds
    model = build_model(tiny_cfg.model)
    half = fp16_roundtrip(model)
    assert all(torch.equal(p, p.half().float()) for p in half.parameters())
    out = precision_report(model, ds, split.test, tiny_cfg.ipm)
    assert out["bytes_fp16"] * 2 == out["bytes_fp32"]
    assert out["max_abs_weight_change"] >= 0 and "delta_log10_kkt_max_abs" in out


def test_ablation_reports_paired_differences(tiny_cfg, tiny_ds):
    from ipm_lstm_attn.models.ablation import run_ablation

    ds, split = tiny_ds
    cfg = tiny_cfg.model_copy(update={"train": tiny_cfg.train.model_copy(update={"epochs": 1})})
    out = run_ablation(cfg, ds, split, log=lambda s: None)
    assert set(out["variants"]) == {"lstm1", "lstm2", "cg3"}
    diff = out["variants"]["lstm2"]["vs_lstm1"]["warm_iters"]
    assert diff["ci_low"] <= diff["mean"] <= diff["ci_high"]
    assert "seed_std" in out["variants"]["lstm2"]

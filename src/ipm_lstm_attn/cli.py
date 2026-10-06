"""Command line interface: ``ipm-lstm-attn <command> ...``."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from . import datasets
from .baselines import make_backend
from .config import ExperimentConfig, device_from_env, load_config, load_dotenv, paths_from_env, resolve_device
from .evaluate import evaluate_solver
from .linsolve import CGNormalSolver, DirectSolver


def _need_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit("this command needs the torch extra: pip install -e \".[torch]\"")


def prepare_dataset(cfg: ExperimentConfig, data_dir: Path) -> tuple[datasets.Dataset, datasets.Split]:
    """Generate the dataset of the config once, then always read it back from disk with validation."""
    d = cfg.data
    ds = datasets.generate(d.family, d.n_var, d.n_eq, d.n_ineq, d.n_instances, d.seed)
    path = datasets.save(ds, data_dir)
    ds = datasets.load(path)
    return ds, datasets.split_indices(len(ds), d.val_frac, d.test_frac, d.split_seed)


def _write(obj: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return path


def _print_summary(rep) -> None:
    s = rep.summary
    print(f"solver {rep.solver}  backend {rep.backend}  instances {rep.n_instances}")
    for key in ("log10_kkt", "rel_gap", "eq_max", "ineq_max", "cold_iters", "warm_iters", "iters_saved"):
        print(f"  {key:12s} mean {s[key]['mean']:.4g}  median {s[key]['median']:.4g}")
    print(f"  warm solves converged: {s['warm_converged_rate']:.0%}")


def _cfg_with(cfg: ExperimentConfig, variant: str | None, seed: int | None) -> ExperimentConfig:
    upd = {}
    if variant:
        upd["model"] = cfg.model.model_copy(update={"variant": variant})
    if seed is not None:
        upd["train"] = cfg.train.model_copy(update={"seed": seed})
    return ExperimentConfig.model_validate({**cfg.model_dump(), **{k: v.model_dump() for k, v in upd.items()}})


def cmd_generate(a) -> None:
    ds = datasets.generate(a.family, a.n_var, a.n_eq, a.n_ineq, a.n_instances, a.seed)
    path = datasets.save(ds, a.out_dir or paths_from_env().data_dir)
    print(f"dataset {path}  sha256 {datasets.content_hash(ds)[:16]}")


def cmd_check_config(a) -> None:
    cfg = load_config(a.config)
    print(json.dumps(cfg.model_dump(), indent=2))


def cmd_baseline(a) -> None:
    cfg = load_config(a.config)
    paths = paths_from_env()
    ds, split = prepare_dataset(cfg, paths.data_dir)
    solver = DirectSolver() if a.solver == "direct" else CGNormalSolver(a.cg_iters or cfg.model.inner_steps)
    backend = make_backend(a.backend, cfg.ipm.warm_tol, cfg.ipm.reference_max_iter, cfg.ipm.sigma, cfg.ipm.tau)
    rep = evaluate_solver(ds, getattr(split, a.split), solver, cfg.ipm, backend)
    _print_summary(rep)
    _write(rep.as_dict(), paths.results_dir / f"{cfg.name}_{solver.name}_{a.split}.json")


def cmd_train(a) -> None:
    _need_torch()
    from .models.train import train

    cfg = _cfg_with(load_config(a.config), a.variant, a.seed)
    paths = paths_from_env()
    ds, split = prepare_dataset(cfg, paths.data_dir)
    device = resolve_device(a.device or device_from_env())
    res = train(cfg, ds, split, device, paths.results_dir / "checkpoints")
    print(f"best val log10 KKT {res.best_val:.3f} at epoch {res.best_epoch}; checkpoint {res.checkpoint}")


def cmd_evaluate(a) -> None:
    _need_torch()
    from .models.adapter import LearnedSolver
    from .models.train import load_checkpoint

    cfg = load_config(a.config)
    paths = paths_from_env()
    ds, split = prepare_dataset(cfg, paths.data_dir)
    device = resolve_device(a.device or device_from_env())
    model = load_checkpoint(a.checkpoint, device)
    backend = make_backend(a.backend, cfg.ipm.warm_tol, cfg.ipm.reference_max_iter, cfg.ipm.sigma, cfg.ipm.tau)
    rep = evaluate_solver(ds, getattr(split, a.split), LearnedSolver(model, device), cfg.ipm, backend)
    _print_summary(rep)
    _write(rep.as_dict(), paths.results_dir / f"{Path(a.checkpoint).stem}_{a.split}.json")


def cmd_ablate(a) -> None:
    _need_torch()
    from .models.ablation import run_ablation

    cfg = load_config(a.config)
    paths = paths_from_env()
    ds, split = prepare_dataset(cfg, paths.data_dir)
    device = resolve_device(a.device or device_from_env())
    out = run_ablation(cfg, ds, split, device, paths.results_dir / "checkpoints")
    for name, row in out["variants"].items():
        mean = row["mean"]
        line = f"{name:12s} warm_iters {mean['warm_iters']:.3f}  log10_kkt {mean['log10_kkt']:.3f}  rel_gap {mean['rel_gap']:.4f}"
        diff = row.get("vs_" + out["baseline"])
        if diff:
            w = diff["warm_iters"]
            line += f"  d_warm_iters {w['mean']:+.3f} [{w['ci_low']:+.3f}, {w['ci_high']:+.3f}]"
        print(line)
    _write(out, paths.results_dir / f"{cfg.name}_ablation.json")


def cmd_bench(a) -> None:
    from .bench import benchmark

    cfg = load_config(a.config)
    paths = paths_from_env()
    ds, split = prepare_dataset(cfg, paths.data_dir)
    device = "cpu"
    if a.checkpoint:
        _need_torch()
        from .models.adapter import LearnedSolver
        from .models.train import load_checkpoint

        device = resolve_device(a.device or device_from_env())
        solver = LearnedSolver(load_checkpoint(a.checkpoint, device), device)
    else:
        solver = CGNormalSolver(cfg.model.inner_steps)
    backend = make_backend(a.backend, cfg.ipm.warm_tol, cfg.ipm.reference_max_iter, cfg.ipm.sigma, cfg.ipm.tau)
    out = benchmark(ds.instances(split.test), solver, backend, cfg.ipm, cfg.bench, device)
    for row in out["rows"]:
        print(f"{row['label']:28s} {row['mode']:12s} batch {row['batch_size']:3d}  "
              f"median {1e3 * row['median_per_instance_s']:.3f} ms/instance")
    print(f"pipeline {1e3 * out['per_instance_pipeline_s']:.3f} ms vs cold {1e3 * out['per_instance_cold_s']:.3f} ms "
          f"per instance; warm iters {out['warm_iters_mean']:.2f} vs cold {out['cold_iters_mean']:.2f}")
    _write(out, paths.results_dir / f"{cfg.name}_{solver.name.replace(':', '_')}_bench.json")


def cmd_precision(a) -> None:
    _need_torch()
    from .models.precision import precision_report
    from .models.train import load_checkpoint

    cfg = load_config(a.config)
    paths = paths_from_env()
    ds, split = prepare_dataset(cfg, paths.data_dir)
    out = precision_report(load_checkpoint(a.checkpoint), ds, split.test, cfg.ipm)
    print(json.dumps(out, indent=2))
    _write(out, paths.results_dir / f"{Path(a.checkpoint).stem}_precision.json")


def cmd_plot(a) -> None:
    from .plots import plot_kkt_history

    print(f"wrote {plot_kkt_history([Path(p) for p in a.reports], Path(a.out))}")


DEMO_CONFIG = {
    "name": "demo",
    "data": {"family": "qp", "n_var": 10, "n_eq": 5, "n_ineq": 5, "n_instances": 120, "seed": 17},
    "ipm": {"outer_iters": 8},
    "model": {"inner_steps": 8, "hidden_dim": 24, "d_model": 8, "heads": 2},
    "train": {"epochs": 4, "batch_size": 16, "seed": 0},
}


def cmd_demo(a) -> None:
    """Offline demo on a small synthetic QP set. No download, no key, no network."""
    cfg = ExperimentConfig.model_validate(DEMO_CONFIG)
    work = Path(a.out_dir) if a.out_dir else Path(tempfile.mkdtemp(prefix="ipm_lstm_attn_demo_"))
    ds, split = prepare_dataset(cfg, work / "data")
    print(f"dataset {ds.name}: train {len(split.train)}, val {len(split.val)}, test {len(split.test)}")
    rows = []
    from .evaluate import reference_objectives

    f_star = reference_objectives(ds.instances(split.test), cfg.ipm)
    solvers = [CGNormalSolver(cfg.model.inner_steps)]
    if not a.no_torch:
        try:
            import torch  # noqa: F401
        except ImportError:
            print("torch is not installed: the demo shows the classical solver only")
        else:
            from .models.adapter import LearnedSolver
            from .models.train import train

            for variant in ("lstm1", "lstm2_mask"):
                vcfg = _cfg_with(cfg, variant, None)
                res = train(vcfg, ds, split, "cpu", None, log=lambda s: None)
                solvers.append(LearnedSolver(res.model, "cpu"))
    for solver in solvers:
        rep = evaluate_solver(ds, split.test, solver, cfg.ipm, f_star=f_star)
        s = rep.summary
        rows.append((solver.name, s["log10_kkt"]["mean"], s["rel_gap"]["mean"], s["cold_iters"]["mean"],
                     s["warm_iters"]["mean"]))
    print(f"{'solver':18s} {'log10 KKT':>10s} {'rel gap':>9s} {'cold it':>8s} {'warm it':>8s}")
    for name, k, g, c, w in rows:
        print(f"{name:18s} {k:10.3f} {g:9.4f} {c:8.2f} {w:8.2f}")
    print("synthetic data, one seed: these numbers show that the pipeline runs, not that a variant is better")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ipm-lstm-attn", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="write one synthetic dataset (immutable)")
    g.add_argument("--family", choices=["qp", "qcqp", "nonconvex"], default="qp")
    g.add_argument("--n-var", type=int, default=20)
    g.add_argument("--n-eq", type=int, default=10)
    g.add_argument("--n-ineq", type=int, default=10)
    g.add_argument("--n-instances", type=int, default=500)
    g.add_argument("--seed", type=int, default=17)
    g.add_argument("--out-dir")
    g.set_defaults(func=cmd_generate)

    c = sub.add_parser("check-config", help="validate a config file (unknown keys are errors)")
    c.add_argument("config")
    c.set_defaults(func=cmd_check_config)

    def with_cfg(name: str, help_: str, func, checkpoint: bool = False, optional_ckpt: bool = False):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--config", required=True)
        sp.add_argument("--device", help="auto, cpu or cuda (default: IPM_LSTM_ATTN_DEVICE or auto)")
        if checkpoint:
            sp.add_argument("--checkpoint", required=not optional_ckpt)
        sp.set_defaults(func=func)
        return sp

    b = with_cfg("baseline", "evaluate a classical Newton solver (no torch)", cmd_baseline)
    b.add_argument("--solver", choices=["direct", "cg"], default="cg")
    b.add_argument("--cg-iters", type=int)
    b.add_argument("--split", choices=["val", "test"], default="test")
    b.add_argument("--backend", choices=["exact_ipm", "ipopt"], default="exact_ipm")

    t = with_cfg("train", "train one variant with one seed", cmd_train)
    t.add_argument("--variant")
    t.add_argument("--seed", type=int)

    e = with_cfg("evaluate", "evaluate a checkpoint", cmd_evaluate, checkpoint=True)
    e.add_argument("--split", choices=["val", "test"], default="test")
    e.add_argument("--backend", choices=["exact_ipm", "ipopt"], default="exact_ipm")

    with_cfg("ablate", "train and compare all variants over all seeds", cmd_ablate)

    bn = with_cfg("bench", "timing protocol (checkpoint optional: CG is the default solver)", cmd_bench,
                  checkpoint=True, optional_ckpt=True)
    bn.add_argument("--backend", choices=["exact_ipm", "ipopt"], default="exact_ipm")

    with_cfg("precision-check", "measure the effect of fp16 weight storage", cmd_precision, checkpoint=True)

    pl = sub.add_parser("plot", help="plot KKT history of evaluate reports (needs the plot extra)")
    pl.add_argument("reports", nargs="+")
    pl.add_argument("--out", default="results/kkt_history.png")
    pl.set_defaults(func=cmd_plot)

    d = sub.add_parser("demo", help="offline demo on synthetic data")
    d.add_argument("--no-torch", action="store_true", help="show the classical solver only")
    d.add_argument("--out-dir")
    d.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv()
    np.set_printoptions(precision=4, suppress=True)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Seeded ablation: every variant x every seed on the same dataset and the same split (needs torch).

The comparison is paired. For each test instance the metric is first
averaged over the seeds of a variant. The difference against the baseline
variant then gets a bootstrap confidence interval over the test instances.
A classical solver with the same inner budget (CG on the normal equations)
is evaluated once as a non-learned reference.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from ..config import ExperimentConfig
from ..datasets import Dataset, Split
from ..evaluate import evaluate_solver, reference_objectives
from ..linsolve import CGNormalSolver
from ..metrics import paired_bootstrap
from .adapter import LearnedSolver
from .train import train

METRICS = ("warm_iters", "log10_kkt", "rel_gap")


def run_ablation(cfg: ExperimentConfig, ds: Dataset, split: Split, device: str = "cpu",
                 checkpoint_dir: Path | None = None, log: Callable[[str], None] = print) -> dict:
    test = split.test
    f_star = reference_objectives(ds.instances(test), cfg.ipm)
    per_variant: dict[str, dict[str, np.ndarray]] = {}
    seed_means: dict[str, dict[str, list[float]]] = {}
    for variant in cfg.ablation.variants:
        runs = {m: [] for m in METRICS}
        seed_means[variant] = {m: [] for m in METRICS}
        for seed in cfg.ablation.seeds:
            run_cfg = cfg.model_copy(update={
                "model": cfg.model.model_copy(update={"variant": variant}),
                "train": cfg.train.model_copy(update={"seed": seed}),
            })
            log(f"== {variant} seed {seed}")
            result = train(run_cfg, ds, split, device, checkpoint_dir, log=log)
            rep = evaluate_solver(ds, test, LearnedSolver(result.model, device), cfg.ipm, f_star=f_star)
            for m in METRICS:
                col = rep.column(m)
                runs[m].append(col)
                seed_means[variant][m].append(float(col.mean()))
        per_variant[variant] = {m: np.mean(np.stack(runs[m]), axis=0) for m in METRICS}

    cg = CGNormalSolver(cfg.model.inner_steps)
    cg_rep = evaluate_solver(ds, test, cg, cfg.ipm, f_star=f_star)
    per_variant[cg.name] = {m: cg_rep.column(m) for m in METRICS}

    base = cfg.ablation.baseline
    table = {}
    for name, cols in per_variant.items():
        row = {"mean": {m: float(cols[m].mean()) for m in METRICS}}
        if name in seed_means:
            row["seed_std"] = {m: float(np.std(seed_means[name][m], ddof=1)) if len(seed_means[name][m]) > 1 else 0.0
                               for m in METRICS}
        if name != base:
            row["vs_" + base] = {m: paired_bootstrap(cols[m], per_variant[base][m], cfg.ablation.bootstrap).as_dict()
                                 for m in METRICS}
        table[name] = row
    return {"baseline": base, "seeds": list(cfg.ablation.seeds), "test_instances": int(len(test)),
            "metrics": list(METRICS), "variants": table}

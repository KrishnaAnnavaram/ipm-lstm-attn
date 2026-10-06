"""Seeded training of a learned Newton solver inside the IPM loop (needs torch).

For each mini-batch the IPM runs ``outer_iters`` steps from the cold start.
At each step the model solves the Newton system, the loss is the mean
relative residual over the inner steps, and the optimiser takes one step. The
IPM then moves with the detached direction. Early stopping watches the mean
log10 KKT error on the validation split.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .. import ipm, kkt
from ..config import ExperimentConfig, ModelConfig
from ..datasets import Dataset, Split
from ..evaluate import approximate
from ..seeding import set_seed
from .adapter import LearnedSolver
from .nets import LearnedNewtonSolver, build_model


@dataclass
class TrainResult:
    model: LearnedNewtonSolver
    best_val: float
    best_epoch: int
    history: list[dict] = field(default_factory=list)
    checkpoint: Path | None = None


class EarlyStopping:
    """Keep the best weights in memory. Stop after ``patience`` epochs with no improvement."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best = float("inf")
        self.best_epoch = -1
        self.best_state: dict | None = None
        self.bad_epochs = 0

    def step(self, value: float, epoch: int, model: torch.nn.Module) -> bool:
        """Return True when training must stop."""
        if value < self.best:
            self.best, self.best_epoch, self.bad_epochs = value, epoch, 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def validation_score(model: LearnedNewtonSolver, ds: Dataset, idx: np.ndarray, cfg: ExperimentConfig,
                     device: str) -> float:
    """Mean log10 KKT error after the fixed outer budget. Lower is better."""
    res = approximate(ds.instances(idx), LearnedSolver(model, device), cfg.ipm)
    return float(np.mean(np.log10(np.maximum(res.errors, 1e-16))))


def train(cfg: ExperimentConfig, ds: Dataset, split: Split, device: str = "cpu",
          checkpoint_dir: str | os.PathLike | None = None, log: Callable[[str], None] = print) -> TrainResult:
    set_seed(cfg.train.seed)  # before the model exists, so the initial weights are reproducible
    model = build_model(cfg.model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    rng = np.random.default_rng(cfg.train.seed)
    stopper = EarlyStopping(cfg.train.patience)
    history: list[dict] = []
    for epoch in range(cfg.train.epochs):
        model.train()
        order = rng.permutation(split.train)
        losses = []
        for start in range(0, len(order), cfg.train.batch_size):
            insts = ds.instances(order[start : start + cfg.train.batch_size])
            y = np.stack([kkt.initial_point(i) for i in insts])
            for _ in range(cfg.ipm.outer_iters):
                systems = [ipm.newton_system(i, y[b], cfg.ipm.sigma, cfg.ipm.precondition) for b, i in enumerate(insts)]
                J = torch.as_tensor(np.stack([s[0] for s in systems]), dtype=torch.float32, device=device)
                F = torch.as_tensor(np.stack([s[1] for s in systems]), dtype=torch.float32, device=device)
                dy, loss, _ = model(J, F)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                opt.step()
                losses.append(float(loss.detach()))
                step = dy.detach().double().cpu().numpy()
                step[~np.isfinite(step)] = 0.0
                y = np.stack([ipm.take_step(i, y[b], step[b], cfg.ipm.tau) for b, i in enumerate(insts)])
        model.eval()
        val = validation_score(model, ds, split.val, cfg, device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_log10_kkt": val})
        log(f"epoch {epoch:3d}  train loss {np.mean(losses):.4e}  val log10 KKT {val:.3f}")
        if stopper.step(val, epoch, model):
            log(f"early stop at epoch {epoch}; best epoch {stopper.best_epoch}")
            break
    assert stopper.best_state is not None
    model.load_state_dict(stopper.best_state)
    model.eval()
    result = TrainResult(model, stopper.best, stopper.best_epoch, history)
    if checkpoint_dir is not None:
        result.checkpoint = save_checkpoint(model, cfg, Path(checkpoint_dir), history)
    return result


def checkpoint_name(cfg: ExperimentConfig) -> str:
    return f"{cfg.name}_{cfg.model.variant}_seed{cfg.train.seed}"


def save_checkpoint(model: LearnedNewtonSolver, cfg: ExperimentConfig, directory: Path, history: list[dict]) -> Path:
    os.makedirs(directory, exist_ok=True)
    path = directory / f"{checkpoint_name(cfg)}.pt"
    torch.save({"state_dict": model.state_dict(), "model_config": cfg.model.model_dump(),
                "train_seed": cfg.train.seed}, path)
    path.with_suffix(".json").write_text(
        json.dumps({"config": cfg.model_dump(), "history": history}, indent=2), encoding="utf-8")
    return path


def load_checkpoint(path: str | os.PathLike, device: str = "cpu") -> LearnedNewtonSolver:
    blob = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ModelConfig.model_validate(blob["model_config"])).to(device)
    model.load_state_dict(blob["state_dict"])
    return model.eval()

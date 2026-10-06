"""Strict experiment configuration.

A config file is TOML. Every section is a pydantic model with
``extra="forbid"``: an unknown or misspelt key is an error, never a silent
no-op. Paths and the device come from environment variables.
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .registry import VARIANTS


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Strict):
    family: Literal["qp", "qcqp", "nonconvex"] = "qp"
    n_var: int = Field(20, ge=2)
    n_eq: int = Field(10, ge=1)
    n_ineq: int = Field(10, ge=1)
    n_instances: int = Field(500, ge=10)
    seed: int = 17
    val_frac: float = Field(0.1, gt=0, lt=1)
    test_frac: float = Field(0.1, gt=0, lt=1)
    split_seed: int = 0

    @model_validator(mode="after")
    def _sizes(self) -> "DataConfig":
        if self.n_eq >= self.n_var:
            raise ValueError("n_eq must be smaller than n_var")
        if self.val_frac + self.test_frac >= 0.9:
            raise ValueError("val_frac + test_frac must be below 0.9")
        return self


class IPMConfig(_Strict):
    sigma: float = Field(0.1, gt=0, lt=1)
    tau: float = Field(0.995, gt=0, lt=1)
    outer_iters: int = Field(20, ge=1)  # fixed budget of the learned IPM
    precondition: bool = True
    reference_tol: float = Field(1e-8, gt=0)
    reference_max_iter: int = Field(200, ge=1)
    warm_tol: float = Field(1e-6, gt=0)  # tolerance of the warm-started solve


class ModelConfig(_Strict):
    variant: str = "lstm2_attn"
    hidden_dim: int = Field(32, ge=2)
    d_model: int = Field(16, ge=2)  # width of the attention or GNN block
    heads: int = Field(4, ge=1)
    dropout: float = Field(0.0, ge=0, lt=1)
    inner_steps: int = Field(10, ge=1)

    @model_validator(mode="after")
    def _variant(self) -> "ModelConfig":
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown variant {self.variant!r}; use one of {sorted(VARIANTS)}")
        if self.d_model % self.heads:
            raise ValueError("d_model must be a multiple of heads")
        return self


class TrainConfig(_Strict):
    seed: int = 0
    epochs: int = Field(20, ge=1)
    batch_size: int = Field(32, ge=1)
    lr: float = Field(1e-3, gt=0)
    weight_decay: float = Field(0.0, ge=0)
    patience: int = Field(5, ge=1)
    grad_clip: float = Field(1.0, gt=0)


class AblationConfig(_Strict):
    variants: list[str] = Field(default_factory=lambda: sorted(VARIANTS))
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    baseline: str = "lstm1"
    bootstrap: int = Field(2000, ge=100)

    @model_validator(mode="after")
    def _known(self) -> "AblationConfig":
        unknown = [v for v in [*self.variants, self.baseline] if v not in VARIANTS]
        if unknown:
            raise ValueError(f"unknown variants {unknown}")
        if self.baseline not in self.variants:
            raise ValueError("baseline must be one of the variants")
        return self


class BenchConfig(_Strict):
    warmup: int = Field(2, ge=0)
    repeats: int = Field(5, ge=1)
    batch_size: int = Field(32, ge=1)
    max_instances: int = Field(20, ge=1)


class ExperimentConfig(_Strict):
    name: str = "experiment"
    data: DataConfig = DataConfig()
    ipm: IPMConfig = IPMConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    ablation: AblationConfig = AblationConfig()
    bench: BenchConfig = BenchConfig()


def load_config(path: str | os.PathLike) -> ExperimentConfig:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    return ExperimentConfig.model_validate(raw)


ENV_VARS = ("IPM_LSTM_ATTN_DATA_DIR", "IPM_LSTM_ATTN_RESULTS_DIR", "IPM_LSTM_ATTN_DEVICE")


def load_dotenv(path: str | os.PathLike = ".env") -> list[str]:
    """Read ``KEY=VALUE`` lines of a local ``.env`` file. Variables that are already set win."""
    p = Path(path)
    if not p.is_file():
        return []
    loaded = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key in ENV_VARS and value and not os.environ.get(key):
            os.environ[key] = value.strip("\"'")
            loaded.append(key)
    return loaded


class Paths(_Strict):
    data_dir: Path
    results_dir: Path


def paths_from_env() -> Paths:
    return Paths(
        data_dir=Path(os.environ.get("IPM_LSTM_ATTN_DATA_DIR") or "data/generated"),
        results_dir=Path(os.environ.get("IPM_LSTM_ATTN_RESULTS_DIR") or "results"),
    )


def device_from_env(default: str = "auto") -> str:
    return os.environ.get("IPM_LSTM_ATTN_DEVICE") or default


def resolve_device(name: str = "auto") -> str:
    """``auto`` gives ``cuda`` when torch sees a GPU, else ``cpu``. Never a hard-coded GPU index."""
    if name != "auto":
        return name
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

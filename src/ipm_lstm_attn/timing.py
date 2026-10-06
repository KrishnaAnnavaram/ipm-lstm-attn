"""Timing protocol: warm-up runs, repeats, and per-instance or batched modes that are never mixed."""
from __future__ import annotations

import os
import platform
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class TimingStats:
    label: str
    mode: str  # "per_instance" or "batched"
    batch_size: int
    instances: int  # instances handled by one call
    warmup: int
    repeats: int
    samples_s: list[float] = field(default_factory=list)  # seconds per call

    @property
    def median_per_instance_s(self) -> float:
        return float(np.median(self.samples_s)) / self.instances

    @property
    def iqr_per_instance_s(self) -> float:
        q1, q3 = np.quantile(self.samples_s, [0.25, 0.75])
        return float(q3 - q1) / self.instances

    def as_dict(self) -> dict:
        d = asdict(self)
        d["median_per_instance_s"] = self.median_per_instance_s
        d["iqr_per_instance_s"] = self.iqr_per_instance_s
        return d


def measure(fn: Callable[[], object], label: str, mode: str, batch_size: int, instances: int,
            warmup: int = 2, repeats: int = 5, sync: Callable[[], None] | None = None) -> TimingStats:
    """Call ``fn`` ``warmup`` times without timing, then ``repeats`` times with timing.

    ``sync`` runs after each call before the clock stops (for example
    ``torch.cuda.synchronize`` on a GPU).
    """
    if mode not in ("per_instance", "batched"):
        raise ValueError("mode must be 'per_instance' or 'batched'")
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be >= 1 and warmup >= 0")
    for _ in range(warmup):
        fn()
        if sync:
            sync()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        samples.append(time.perf_counter() - t0)
    return TimingStats(label, mode, batch_size, instances, warmup, repeats, samples)


def environment() -> dict[str, str | int]:
    """The facts that a timing result depends on."""
    info: dict[str, str | int] = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count() or 0,
        "numpy": np.__version__,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        info["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except ImportError:
        info["torch"] = "not installed"
    return info

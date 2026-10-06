"""Plots from saved result files (needs the ``plot`` extra: matplotlib)."""
from __future__ import annotations

import json
from pathlib import Path


def plot_kkt_history(report_paths: list[Path], out: Path) -> Path:
    """Plot the mean log10 KKT error per outer IPM step for one or more ``evaluate`` reports."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for path in report_paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        hist = data["kkt_history"]
        ax.plot(range(1, len(hist) + 1), hist, marker="o", label=data["solver"])
    ax.set_xlabel("outer IPM step")
    ax.set_ylabel("mean log10 KKT error")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out

"""ipm_lstm_attn: a learned interior-point method with a seeded ablation of the Newton-step solver.

The core package (problems, KKT system, IPM, data, metrics, timing) needs only
numpy, scipy and pydantic. The learned solvers in ``ipm_lstm_attn.models``
need the ``torch`` extra.
"""

__version__ = "0.1.0"

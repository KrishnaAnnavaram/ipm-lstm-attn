# Third-party credit

## Method

This project studies the learned interior-point method from:

> "IPM-LSTM: A Learning-Based Interior Point Method for Solving Nonlinear Programs", NeurIPS 2024.

The code in this repository is a new implementation. It copies no source file of the upstream
IPM-LSTM repository. If you use the method, cite the paper above and respect the license of the
upstream repository.

## Benchmark recipe

The synthetic "RHS" instance recipe follows:

> P. L. Donti, D. Rolnick and J. Z. Kolter, "DC3: A learning method for optimization with hard
> constraints", ICLR 2021.

## Optional software

| Package | Use | License |
|---|---|---|
| PyTorch | learned solvers (`torch` extra) | BSD-3-Clause |
| cyipopt / IPOPT | optional warm-start back-end (`ipopt` extra) | EPL-2.0 (IPOPT) |
| matplotlib | plots (`plot` extra) | Matplotlib license (PSF-based) |

"""Synthetic instance generation, immutable storage, schema validation and splits.

The recipe follows the DC3 "RHS" benchmark: Q, p, A, G, c (and H for QCQP) are
shared, and each instance has its own equality right-hand side ``b``.

Files are written once. A second write with different content raises an
error, and the loader never writes. There is no tool that edits a dataset in
place.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .problems import FAMILIES, Instance

GENERATOR_VERSION = "1"
SHARED_KEYS = ("Q", "p", "A", "G", "c")
# Older files name the equality right-hand side "X". The loader reads both names.
RHS_ALIASES = ("b", "X")


@dataclass(frozen=True)
class Dataset:
    family: str
    Q: np.ndarray
    p: np.ndarray
    A: np.ndarray
    G: np.ndarray
    c: np.ndarray
    b: np.ndarray  # (N_instances, p_eq)
    H: np.ndarray | None
    meta: dict

    def __len__(self) -> int:
        return self.b.shape[0]

    def instance(self, i: int) -> Instance:
        return Instance(self.family, self.Q, self.p, self.A, self.b[i], self.G, self.c, self.H)

    def instances(self, idx: np.ndarray | list[int]) -> list[Instance]:
        return [self.instance(int(i)) for i in idx]

    @property
    def name(self) -> str:
        m = self.meta
        return f"{m['family']}_n{m['n_var']}_eq{m['n_eq']}_in{m['n_ineq']}_N{m['n_instances']}_s{m['seed']}"


def generate(family: str, n_var: int, n_eq: int, n_ineq: int, n_instances: int, seed: int) -> Dataset:
    """Make a dataset with a private ``numpy.random.Generator``. Same arguments give the same arrays."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    if not (0 < n_eq < n_var) or n_ineq < 1 or n_instances < 3:
        raise ValueError("need 0 < n_eq < n_var, n_ineq >= 1 and n_instances >= 3")
    rng = np.random.default_rng(seed)
    if family == "qcqp":
        Q = np.diag(rng.uniform(0.0, 0.5, n_var))
        p = rng.uniform(-1.0, 1.0, n_var)
        A = rng.uniform(-1.0, 1.0, (n_eq, n_var))
        G = rng.uniform(-1.0, 1.0, (n_ineq, n_var))
        H = np.stack([np.diag(rng.uniform(0.0, 0.1, n_var)) for _ in range(n_ineq)])
        b = rng.uniform(-0.5, 0.5, (n_instances, n_eq))
        b_max = 0.5
    else:
        Q = np.diag(rng.uniform(0.0, 1.0, n_var))
        p = rng.uniform(0.0, 1.0, n_var)
        A = rng.normal(0.0, 1.0, (n_eq, n_var))
        G = rng.normal(0.0, 1.0, (n_ineq, n_var))
        H = None
        b = rng.uniform(-1.0, 1.0, (n_instances, n_eq))
        b_max = 1.0
    # c makes x = pinv(A) b strictly feasible for every |b_j| <= b_max.
    A_pinv = np.linalg.pinv(A)
    c = b_max * np.sum(np.abs(G @ A_pinv), axis=1)
    if H is not None:
        x_norm_sq = (np.linalg.norm(A_pinv, 2) * b_max) ** 2 * n_eq
        c = c + 0.5 * x_norm_sq * np.array([np.max(np.diag(h)) for h in H])
    c = c + 1e-3
    meta = {
        "family": family,
        "n_var": n_var,
        "n_eq": n_eq,
        "n_ineq": n_ineq,
        "n_instances": n_instances,
        "seed": seed,
        "generator_version": GENERATOR_VERSION,
    }
    return Dataset(family, Q, p, A, G, c, b, H, meta)


def _arrays(ds: Dataset) -> dict[str, np.ndarray]:
    arrs = {"Q": ds.Q, "p": ds.p, "A": ds.A, "G": ds.G, "c": ds.c, "b": ds.b}
    if ds.H is not None:
        arrs["H"] = ds.H
    return arrs


def content_hash(ds: Dataset) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(ds.meta, sort_keys=True).encode())
    for key, arr in sorted(_arrays(ds).items()):
        h.update(key.encode())
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    return h.hexdigest()


class ImmutableDataError(RuntimeError):
    """A write tried to replace an existing dataset with different content."""


def save(ds: Dataset, directory: str | os.PathLike) -> Path:
    """Write ``<name>.npz`` and ``<name>.json`` once. Same content again is a no-op."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{ds.name}.npz"
    manifest = directory / f"{ds.name}.json"
    digest = content_hash(ds)
    if path.exists() or manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
        if old.get("sha256") == digest and path.exists():
            return path
        raise ImmutableDataError(f"{path} exists with different content; datasets are never rewritten")
    buf = io.BytesIO()
    np.savez(buf, **_arrays(ds))
    tmp = path.with_suffix(".npz.tmp")
    tmp.write_bytes(buf.getvalue())
    tmp.replace(path)
    manifest.write_text(json.dumps({**ds.meta, "sha256": digest}, indent=2, sort_keys=True), encoding="utf-8")
    return path


class SchemaError(ValueError):
    """A dataset file does not match the expected keys or shapes."""


def load(path: str | os.PathLike, verify_hash: bool = True) -> Dataset:
    """Read a dataset and check its keys, shapes and (optionally) its SHA-256 value."""
    path = Path(path)
    manifest = path.with_suffix(".json")
    if not manifest.exists():
        raise SchemaError(f"missing manifest {manifest}")
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    for key in ("family", "n_var", "n_eq", "n_ineq", "n_instances", "seed", "generator_version"):
        if key not in meta:
            raise SchemaError(f"manifest misses {key!r}")
    with np.load(path, allow_pickle=False) as z:
        files = set(z.files)
        missing = [k for k in SHARED_KEYS if k not in files]
        if missing:
            raise SchemaError(f"{path} misses arrays {missing}")
        rhs_key = next((k for k in RHS_ALIASES if k in files), None)
        if rhs_key is None:
            raise SchemaError(f"{path} has no equality right-hand side ('b' or 'X')")
        arrs = {k: np.array(z[k], dtype=np.float64) for k in SHARED_KEYS}
        b = np.array(z[rhs_key], dtype=np.float64)
        H = np.array(z["H"], dtype=np.float64) if "H" in files else None
    if b.ndim == 3 and b.shape[-1] == 1:
        b = b[..., 0]
    n, p_eq, m, N = meta["n_var"], meta["n_eq"], meta["n_ineq"], meta["n_instances"]
    expected = {"Q": (n, n), "p": (n,), "A": (p_eq, n), "G": (m, n), "c": (m,)}
    for key, shape in expected.items():
        if arrs[key].shape != shape:
            raise SchemaError(f"{key} has shape {arrs[key].shape}, expected {shape}")
    if b.shape != (N, p_eq):
        raise SchemaError(f"b has shape {b.shape}, expected {(N, p_eq)}")
    if meta["family"] == "qcqp" and (H is None or H.shape != (m, n, n)):
        raise SchemaError("qcqp dataset needs H with shape (m, n, n)")
    for arr in [*arrs.values(), b] + ([H] if H is not None else []):
        if not np.all(np.isfinite(arr)):
            raise SchemaError("dataset has non-finite values")
        arr.setflags(write=False)
    ds = Dataset(meta["family"], arrs["Q"], arrs["p"], arrs["A"], arrs["G"], arrs["c"], b, H,
                 {k: meta[k] for k in meta if k != "sha256"})
    if verify_hash and "sha256" in meta and content_hash(ds) != meta["sha256"]:
        raise SchemaError(f"{path} does not match the SHA-256 value in its manifest")
    return ds


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def split_indices(n_instances: int, val_frac: float, test_frac: float, seed: int) -> Split:
    """Disjoint train/val/test index sets from one seeded permutation.

    Every array of an instance comes from the same index, so no split can mix
    the inequality data of one part with the equality data of another part.
    """
    if not (0 < val_frac < 1 and 0 < test_frac < 1 and val_frac + test_frac < 1):
        raise ValueError("fractions must be in (0, 1) and sum to less than 1")
    perm = np.random.default_rng(seed).permutation(n_instances)
    n_test = max(1, int(round(n_instances * test_frac)))
    n_val = max(1, int(round(n_instances * val_frac)))
    if n_test + n_val >= n_instances:
        raise ValueError("too few instances for this split")
    test = np.sort(perm[:n_test])
    val = np.sort(perm[n_test : n_test + n_val])
    train = np.sort(perm[n_test + n_val :])
    return Split(train, val, test)

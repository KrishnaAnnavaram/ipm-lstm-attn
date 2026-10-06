"""Learned Newton solvers. The submodules import torch; this package file does not.

Install the ``torch`` extra to use them: ``pip install -e ".[torch]"``.
"""
from ..registry import VARIANTS, Variant

__all__ = ["VARIANTS", "Variant", "torch_available"]


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True

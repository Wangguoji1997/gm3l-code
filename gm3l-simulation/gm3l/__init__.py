"""Gaussian-mixture network simulation and COM, SMLE, and FINAL estimators."""

from importlib import import_module
import sys

from .strict_smle import FitConfig

__all__ = ["FitConfig", "get_backend"]


def get_backend(name="numpy"):
    """Select a numerical backend without loading a native library at import."""
    if name == "numpy":
        return import_module(".strict_smle", __name__)
    if name == "accelerate":
        if sys.platform != "darwin":
            raise RuntimeError("The accelerate backend requires macOS; use numpy instead")
        return import_module(".strict_smle_accelerate", __name__)
    raise ValueError("backend must be 'numpy' or 'accelerate'")

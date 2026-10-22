"""Shared input-identity, completion, and exported-score checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from scipy.special import logsumexp


INPUT_NAMES = ("X_train", "X_train_eval", "X_test", "A_train", "A_test_train", "A_test_test")
PROBABILITY_TYPES = {"probability"}
LOG_TYPES = {"log_probability", "logits", "log_likelihood", "log_score"}
NATIVE_TYPES = {"native", "decision", "negative_distance", "distance_score", "confidence", "vote_fraction"}
SUCCESS_STATUSES = {"complete", "completed", "ok", "success", "converged"}


def fingerprint(input_dir: str | Path) -> str:
    """Hash model inputs and config using the original run-record format.

    Evaluation labels are outside this allowlist and are never read here.
    """
    root = Path(input_dir)
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("protocol") not in {"inductive", "transductive"}:
        raise ValueError("Specify inductive or transductive protocol")
    names = INPUT_NAMES if cfg["protocol"] == "transductive" else INPUT_NAMES[:-1]
    digest = hashlib.sha256()
    for path in [root / "config.json"] + [root / (key + ".npy") for key in names]:
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def verify_input_identity(diagnostics: dict[str, Any], expected_sha256: str) -> None:
    """Reject unverified or stale results before computing evaluation scores."""
    recorded = diagnostics.get("input_sha256")
    if not isinstance(recorded, str) or re.fullmatch(r"[0-9a-f]{64}", recorded) is None:
        raise ValueError("Missing or invalid run input_sha256; input identity cannot be verified")
    if recorded != expected_sha256:
        raise ValueError("Run input_sha256 does not match the prepared model inputs/config")


def completion_error(diagnostics: dict[str, Any], *, require_completed: bool = False) -> str | None:
    """Explicit failure overrides completion; unknown convergence is not failure.

    Native runs need not have a completed flag. Legacy completed-only records
    remain readable by the evaluator, but external adapters must declare both
    completed=true and a successful status. Fixed-budget methods need not claim
    numerical convergence or certify a global optimum.
    """
    if "completed" in diagnostics and type(diagnostics["completed"]) is not bool:
        return "completed must be a boolean"
    if diagnostics.get("completed") is False:
        return "Run declares completed=false"
    convergence = diagnostics.get("convergence")
    if (diagnostics.get("converged") is False or convergence is False
            or (isinstance(convergence, dict) and convergence.get("converged") is False)):
        return "Run explicitly declares nonconvergence"
    if "status" in diagnostics:
        status = diagnostics["status"]
        if not isinstance(status, str) or status not in SUCCESS_STATUSES:
            return f"Run status is not successful: {status!r}"
    elif require_completed or diagnostics.get("completed") is not True:
        return "Run has no successful completion status"
    if require_completed and diagnostics.get("completed") is not True:
        return "External adapter must declare completed=true"
    return None


def validate_scores(values: Any, n: int, c: int, score_type: str, name: str) -> np.ndarray:
    """Validate exported scores without clipping or altering zero probabilities."""
    values = np.asarray(values, dtype=float)
    if values.shape != (n, c):
        raise ValueError(f"{name}: expected {(n, c)}, got {values.shape}")
    if not isinstance(score_type, str) or score_type not in PROBABILITY_TYPES | LOG_TYPES | NATIVE_TYPES:
        raise ValueError(f"Unsupported score_type: {score_type}")
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError(f"{name} has NaN or positive infinity")
    if score_type in LOG_TYPES:
        if not np.isfinite(logsumexp(values, axis=1)).all():
            raise ValueError(f"{name} contains a row with no finite log score")
    elif not np.isfinite(values).all():
        raise ValueError(f"{name} must have finite scores")
    if score_type in PROBABILITY_TYPES:
        if np.any(values < 0) or not np.allclose(values.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
            raise ValueError(f"{name} must contain nonnegative row-normalized probabilities/fractions")
    return values

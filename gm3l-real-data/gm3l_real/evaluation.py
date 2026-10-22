"""Label-isolated, training-aligned evaluation of exported clustering predictions.

This module never fits a model or selects a checkpoint, seed, or score definition.
Only this evaluation boundary may consume evaluation_only.npz.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

from .validation import (PROBABILITY_TYPES, LOG_TYPES, NATIVE_TYPES, SUCCESS_STATUSES,
                         completion_error, fingerprint, validate_scores, verify_input_identity)

POSTERIOR_TYPES = {"probability", "log_probability", "logits"}


class UndefinedPosteriorMSE(ValueError):
    """The exported scores cannot supply an aligned class-posterior MSE."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _integers(values: Any, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0 or not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"{name} must be a nonempty one-dimensional numeric array")
    if not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
        raise ValueError(f"{name} must contain finite integers")
    return raw.astype(np.int64)


def _labels(path: Path) -> np.ndarray:
    # Do not read as dtype=int: that can silently truncate malformed decimals.
    values = np.loadtxt(path, delimiter=",", ndmin=2)
    if values.shape[1] != 1:
        raise ValueError(f"{path.name} must have one column")
    return _integers(values[:, 0], path.name)


def _score_matrix(path: Path, n: int, c: int, score_type: str) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", ndmin=2)
    return validate_scores(values, n, c, score_type, path.name)


def log_probabilities(scores: np.ndarray, score_type: str) -> np.ndarray:
    """Normalize declared probabilistic/log scores; never reinterpret distances."""
    if score_type in PROBABILITY_TYPES:
        with np.errstate(divide="ignore"):
            return np.log(scores)
    if score_type in LOG_TYPES:
        return scores - logsumexp(scores, axis=1, keepdims=True)
    raise ValueError(f"Cannot convert {score_type!r} to probabilities without an explicit model")


def _log_posterior_mse(log_probability: np.ndarray, labels: np.ndarray) -> float:
    """Natural log of posterior MSE over samples and classes, after alignment.

    Use the sum of wrong-class probabilities for the true-class error, avoiding
    subtraction from a probability rounded to one. Keep squares and sums in the
    log domain. An all-zero error in the supplied scores gives -inf; probability
    tails already rounded to zero before export cannot be recovered here.
    """
    n, k = log_probability.shape
    log_p = log_probability - logsumexp(log_probability, axis=1, keepdims=True)
    wrong = log_p[np.arange(k)[None, :] != labels[:, None]].reshape(n, k - 1)
    log_true_error = logsumexp(wrong, axis=1)
    log_squared_errors = np.concatenate((2 * log_true_error, 2 * wrong.ravel()))
    return float(logsumexp(log_squared_errors) - np.log(n * k))


def training_mapping(
    train: np.ndarray, y_train: np.ndarray, columns: np.ndarray, K: int,
    *, alignment: str = "hard", mapping: str = "hungarian",
    train_probability: np.ndarray | None = None,
) -> dict[int, int]:
    """Construct mappings using training truth only; test truth is not accepted."""
    if alignment not in {"hard", "soft"} or mapping not in {"hungarian", "majority"}:
        raise ValueError("alignment must be hard/soft; mapping must be hungarian/majority")
    if alignment == "soft":
        if train_probability is None or train_probability.shape != (len(train), len(columns)):
            raise ValueError("Soft alignment requires native train_scores.csv with the same columns")
        overlap = train_probability.T @ np.eye(K, dtype=float)[y_train]
    else:
        overlap = np.asarray([np.bincount(y_train[train == c], minlength=K) for c in columns])
    if mapping == "hungarian":
        # Rectangular Hungarian is allowed for native variable-K partitions.
        # Unmatched score classes remain explicitly undefined, never zero-filled.
        rows, classes = linear_sum_assignment(-overlap)
        return {int(columns[r]): int(c) for r, c in zip(rows, classes)}
    if alignment != "hard":
        raise ValueError("Unknown-K majority mapping is defined from hard training allocations")
    return {int(columns[j]): int(overlap[j].argmax()) for j in range(len(columns)) if overlap[j].sum() > 0}


def _aligned_log_probability(
    log_probability: np.ndarray, columns: np.ndarray, mapping: dict[int, int], y_train: np.ndarray,
    K: int, meta: dict[str, Any],
) -> np.ndarray:
    """Apply the training mapping without leaving the log-probability domain."""
    unseen = meta.get("unseen_cluster_policy", "error")
    if unseen not in {"error", "training_prior"}:
        raise ValueError("unseen_cluster_policy must be error or training_prior")
    unmatched = [j for j, c in enumerate(columns) if int(c) not in mapping]
    by_class = [[j for j, c in enumerate(columns) if mapping.get(int(c)) == k] for k in range(K)]
    missing = [k for k, indices in enumerate(by_class) if not indices]
    if unmatched and unseen == "error":
        raise UndefinedPosteriorMSE(f"Unmapped score clusters: {[int(columns[j]) for j in unmatched]}")
    if missing and not (unmatched and unseen == "training_prior"):
        raise UndefinedPosteriorMSE(f"No mapped posterior probability for classes: {missing}; no constant-score imputation")
    if not unmatched and all(len(indices) == 1 for indices in by_class):
        return np.column_stack([log_probability[:, indices[0]] for indices in by_class])
    lp = log_probabilities(log_probability, "log_probability")
    result = np.full((len(lp), K), -np.inf)
    for k, indices in enumerate(by_class):
        if indices:
            result[:, k] = logsumexp(lp[:, indices], axis=1)
    if unmatched:
        prior = np.bincount(y_train, minlength=K) / len(y_train)
        result = np.logaddexp(result, logsumexp(lp[:, unmatched], axis=1)[:, None] + np.log(prior)[None, :])
    return result


def evaluate_one(
    output: str | Path, y_train: Any, y_test: Any, K: int, *, protocol: Any = None,
    expected_input_sha256: str | None = None,
) -> dict[str, Any]:
    """Evaluate one method/split without writing to or changing model outputs."""
    output = Path(output)
    diag = _json(output / "diagnostics.json")
    failure = completion_error(diag)
    if failure is not None:
        detail = diag.get("error", diag.get("reason"))
        return {"status": "failed", "reason": failure + (f": {detail}" if detail else ""),
                "runtime_seconds": diag.get("runtime_seconds", diag.get("elapsed_seconds"))}
    if expected_input_sha256 is not None:
        verify_input_identity(diag, expected_input_sha256)
    if protocol is not None and "protocol" in diag and diag["protocol"] != protocol:
        raise ValueError("Adapter and prepared split declare different protocols")
    if isinstance(K, bool) or not isinstance(K, (int, np.integer)) or K < 2:
        raise ValueError("K must be an integer >= 2")
    yt, ye = _integers(y_train, "y_train"), _integers(y_test, "y_test")
    if np.any(yt < 0) or np.any(yt >= K) or np.any(ye < 0) or np.any(ye >= K):
        raise ValueError("Evaluation labels must lie in 0..K-1")
    if set(yt) != set(range(K)):
        raise ValueError("Training labels do not contain all K classes; training alignment is undefined")
    train, test = _labels(output / "train_labels.csv"), _labels(output / "test_labels.csv")
    if len(train) != len(yt) or len(test) != len(ye):
        raise ValueError("Prediction lengths do not match the evaluation split")
    meta = _json(output / "score_columns.json")
    columns = _integers(meta["cluster_ids"], "cluster_ids")
    if len(np.unique(columns)) != len(columns):
        raise ValueError("Duplicate score columns")
    if not set(train).issubset(set(columns)) or not set(test).issubset(set(columns)):
        raise ValueError("Hard prediction cluster IDs are absent from score columns")
    score_type = str(meta["score_type"])
    scores = _score_matrix(output / "test_scores.csv", len(test), len(columns), score_type)
    alignment, rule = meta.get("alignment", "hard"), meta.get("mapping", "hungarian")
    train_probability = None
    if alignment == "soft":
        train_scores = _score_matrix(output / "train_scores.csv", len(train), len(columns), score_type)
        train_probability = np.exp(log_probabilities(train_scores, score_type))
    mapping = training_mapping(train, yt, columns, K, alignment=alignment, mapping=rule,
                               train_probability=train_probability)
    unseen = meta.get("unseen_cluster_policy", "error")
    if unseen not in {"error", "training_prior"}:
        raise ValueError("unseen_cluster_policy must be error or training_prior")
    fallback = int(np.bincount(yt, minlength=K).argmax()) if unseen == "training_prior" else -1
    prediction = np.asarray([mapping.get(int(c), fallback) for c in test])
    result: dict[str, Any] = {
        "status": "complete", "reason": "", "ACC": float(np.mean(prediction == ye)),
        "mapping": {str(c): k for c, k in mapping.items()},
        "mapping_rule": rule, "alignment": alignment, "score_type": score_type,
        "alignment_source": meta.get("alignment_source", "exported native training assignments/scores"),
        "unseen_cluster_policy": unseen,
        "score_aggregation": meta.get("score_aggregation"),
        "n_train": len(train), "n_test": len(test), "raw_train_clusters": len(np.unique(train)),
        "raw_test_clusters": len(np.unique(test)),
        "runtime_seconds": diag.get("runtime_seconds", diag.get("elapsed_seconds")),
        "adapter_convergence": diag.get("converged", diag.get("convergence")),
        "MSE_zeta": None, "log_MSE_zeta": None, "MSE_zeta_status": None,
    }
    try:
        if score_type not in POSTERIOR_TYPES:
            raise UndefinedPosteriorMSE(f"score_type={score_type!r} does not declare class posterior probabilities")
        log_probability = log_probabilities(scores, score_type)
        aligned = _aligned_log_probability(log_probability, columns, mapping, yt, K, meta)
        log_mse = _log_posterior_mse(aligned, ye)
        with np.errstate(under="ignore"):
            result["MSE_zeta"] = float(np.exp(log_mse))
        # JSON cannot represent -inf: null with zero_in_input denotes that
        # case, distinct from a finite log MSE whose ordinary MSE underflows.
        result["log_MSE_zeta"] = log_mse if np.isfinite(log_mse) else None
        result["MSE_zeta_status"] = (
            "zero_in_input" if np.isneginf(log_mse)
            else "underflow" if result["MSE_zeta"] == 0.0 else "positive"
        )
    except UndefinedPosteriorMSE as exc:
        result.update(status="mse_unavailable", reason=str(exc))
    return result


def normalize_split_id(value: Any) -> str:
    value = str(value)
    suffix = value[6:] if value.startswith("split_") else value
    if not suffix.isdigit():
        raise ValueError(f"Invalid split ID: {value}")
    return f"split_{int(suffix):03d}"


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def _has_posterior_mse(row: dict[str, Any]) -> bool:
    """Include finite log errors even when raw MSE underflows; retain input zeros."""
    if row.get("status") != "complete":
        return False
    mse, log_mse = row.get("MSE_zeta"), row.get("log_MSE_zeta")
    if not isinstance(mse, (float, int)) or not np.isfinite(mse) or not 0 <= mse <= 1:
        return False
    if row.get("MSE_zeta_status") == "zero_in_input":
        return mse == 0.0 and log_mse is None
    return (row.get("MSE_zeta_status") in {"positive", "underflow"}
            and isinstance(log_mse, (float, int)) and np.isfinite(log_mse) and log_mse <= 0)


def summarize_rows(rows: list[dict[str, Any]], methods: list[str], split_ids: list[str]) -> dict[str, Any]:
    """Use the same posterior-MSE-valid split intersection for every method."""
    index = {(row["split_id"], row["method"]): row for row in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate split/method records")
    common = [split for split in split_ids if all(
        _has_posterior_mse(index.get((split, method), {})) for method in methods)]
    summaries = []
    for method in methods:
        own = [row for row in rows if row["method"] == method]
        counts = Counter(row["status"] for row in own)
        item: dict[str, Any] = {"method": method, "requested_n": len(split_ids), "paired_n": len(common)}
        item.update({status + "_n": counts[status] for status in ("complete", "failed", "missing", "evaluation_failed", "mse_unavailable")})
        for metric in ("ACC", "log_MSE_zeta"):
            raw_values = [index[(split, method)].get(metric) for split in common]
            # An input zero corresponds to log MSE = -inf, not a finite boxplot
            # observation. Do not silently omit it or invent a plotting floor.
            complete_values = all(isinstance(v, (float, int)) and np.isfinite(v) for v in raw_values)
            values = np.asarray(raw_values if complete_values else [], dtype=float)
            item[metric + "_mean"] = float(values.mean()) if values.size else None
            item[metric + "_median"] = float(np.median(values)) if values.size else None
            item[metric + "_sd"] = float(values.std(ddof=1)) if values.size > 1 else None
            item[metric + "_q1"] = float(np.quantile(values, .25)) if values.size else None
            item[metric + "_q3"] = float(np.quantile(values, .75)) if values.size else None
        mse = [index[(split, method)].get("MSE_zeta") for split in common]
        item["MSE_zeta_available_n"] = sum(value is not None for value in mse)
        available_mse = np.asarray([value for value in mse if value is not None], dtype=float)
        # No method-specific favorable subset; retain the shared MSE cohort.
        full_mse = bool(len(common)) and len(available_mse) == len(common)
        item["MSE_zeta_mean"] = float(available_mse.mean()) if full_mse else None
        item["MSE_zeta_median"] = float(np.median(available_mse)) if full_mse else None
        item["MSE_zeta_sd"] = float(available_mse.std(ddof=1)) if full_mse and len(available_mse) > 1 else None
        mse_rows = [index[(split, method)] for split in common]
        item["MSE_zeta_underflow_n"] = sum(row.get("MSE_zeta_status") == "underflow" for row in mse_rows)
        item["MSE_zeta_zero_in_input_n"] = sum(row.get("MSE_zeta_status") == "zero_in_input" for row in mse_rows)
        log_mse = [(-np.inf if row.get("MSE_zeta_status") == "zero_in_input"
                    else row.get("log_MSE_zeta")) for row in mse_rows]
        # Log of the arithmetic mean MSE, NOT the arithmetic mean of log MSEs.
        # Retain the same common cohort and include input zeros without flooring.
        log_mean_mse = (float(logsumexp(log_mse) - np.log(len(common)))
                        if full_mse and all(value is not None for value in log_mse) else None)
        item["log_mean_MSE_zeta"] = (log_mean_mse if log_mean_mse is not None
                                     and np.isfinite(log_mean_mse) else None)
        summaries.append(item)
    excluded = [row for row in rows if row["split_id"] not in common]
    return {"methods": methods, "requested_split_ids": split_ids, "common_valid_split_ids": common,
            "paired_n": len(common), "summary": summaries, "excluded_split_records": excluded,
            "comparison_policy": "All summaries use the same posterior-MSE-valid split intersection across methods; ACC is auxiliary and does not select the cohort. Input zeros are retained, not floored or silently removed.",
            "uncertainty_note": "Spread across overlapping random partitions is not an independent-sample confidence interval."}


def evaluate_experiment(
    prepared_dir: str | Path, runs_dir: str | Path, output_dir: str | Path,
    methods: Iterable[str], split_ids: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Verify run/input identity, then score exported predictions against labels."""
    prepared, runs, output = Path(prepared_dir), Path(runs_dir), Path(output_dir)
    methods = list(methods)
    if not methods or len(set(methods)) != len(methods) or any(m in {"", ".", ".."} or Path(m).name != m for m in methods):
        raise ValueError("Supply a nonempty list of unique method directory names")
    ids = [normalize_split_id(x) for x in split_ids] if split_ids is not None else sorted(p.name for p in prepared.glob("split_*") if p.is_dir())
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("No requested splits, or duplicate split IDs")
    ids = sorted(ids, key=lambda x: int(x.split("_")[1]))
    rows: list[dict[str, Any]] = []
    protocols: set[str] = set()
    for split in ids:
        config = None
        try:
            config = _json(prepared / split / "input" / "config.json")
            for key in ("K", "seed", "protocol"):
                if key not in config:
                    raise ValueError(f"Prepared config missing {key}")
            protocols.add(json.dumps(config["protocol"], sort_keys=True))
            input_identity = fingerprint(prepared / split / "input")
            with np.load(prepared / split / "evaluation_only.npz", allow_pickle=False) as data:
                yt, ye = data["y_train"].copy(), data["y_test"].copy()
            split_error = None
        except Exception as exc:
            split_error = exc
        for method in methods:
            row: dict[str, Any] = {"split_id": split, "method": method, "status": "missing", "reason": ""}
            if config is not None:
                row.update({key: config.get(key) for key in ("K", "seed", "protocol")})
            try:
                if split_error is not None:
                    raise split_error
                row.update(evaluate_one(runs / split / method, yt, ye, config["K"],
                                        protocol=config["protocol"], expected_input_sha256=input_identity))
            except FileNotFoundError as exc:
                row.update(status="missing", reason=f"Missing required file: {Path(exc.filename).name if exc.filename else str(exc)}")
            except Exception as exc:
                row.update(status="evaluation_failed", reason=f"{type(exc).__name__}: {exc}")
            rows.append(row)
    if len(protocols) > 1:
        # Retain all records for audit but never aggregate different protocols.
        for row in rows:
            if row["status"] == "complete":
                row.update(status="evaluation_failed", reason="Mixed protocols in one comparison; evaluate each protocol separately")
    # Changing score definitions across seeds invalidates that method's cohort.
    semantics = ("score_type", "alignment", "alignment_source", "mapping_rule", "unseen_cluster_policy", "score_aggregation")
    for method in methods:
        own = [row for row in rows if row["method"] == method and row["status"] in {"complete", "mse_unavailable"}]
        signatures = {tuple(row.get(key) for key in semantics) for row in own}
        if len(signatures) > 1:
            for row in own:
                row.update(status="evaluation_failed", reason="Score/alignment definition changes across splits")
    payload = summarize_rows(rows, methods, ids)
    payload["protocols"] = [json.loads(p) for p in sorted(protocols)]
    output.mkdir(parents=True, exist_ok=True)
    fields = ["split_id", "seed", "protocol", "K", "method", "status", "reason", "ACC",
              "MSE_zeta", "log_MSE_zeta", "MSE_zeta_status",
              "score_type", "alignment", "alignment_source", "mapping_rule", "unseen_cluster_policy", "score_aggregation",
              "runtime_seconds", "n_train", "n_test", "raw_train_clusters", "raw_test_clusters", "adapter_convergence", "mapping"]
    _write_csv(output / "metrics.csv", rows, fields)
    _write_csv(output / "summary.csv", payload["summary"], list(payload["summary"][0]))
    _write_json(output / "summary.json", payload)
    return payload

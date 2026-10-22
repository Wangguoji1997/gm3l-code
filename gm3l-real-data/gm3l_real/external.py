"""Portable, opt-in command adapter for user-installed external methods.

Only explicit model arrays are exported. This is an interface, not a bundled
implementation or an operating-system sandbox for untrusted external code.
"""
from __future__ import annotations

import json
from numbers import Integral
from pathlib import Path
import re
import subprocess
import tempfile
import time

import numpy as np

from .validation import completion_error, validate_scores


MODEL_INPUT_KEYS = frozenset({"X_train", "X_test", "A_train", "A_test_train", "A_test_test"})
REQUIRED_OUTPUTS = ("train_labels.csv", "test_labels.csv", "test_scores.csv",
                    "score_columns.json", "diagnostics.json")


class ExternalAdapterError(RuntimeError):
    def __init__(self, status, message, diagnostics=None):
        super().__init__(message)
        self.status = status
        self.diagnostics = dict(diagnostics or {}, status=status, completed=False, error=message)


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_model_inputs(inputs, *, transductive=False):
    """Copy only allowed arrays; never silently filter a supplied y field."""
    if not isinstance(inputs, dict):
        raise TypeError("inputs must be a mapping of model arrays")
    unknown = set(inputs) - MODEL_INPUT_KEYS
    if unknown:
        raise ValueError(f"Prohibited model-input keys (labels are never accepted): {sorted(unknown)}")
    if not {"X_train", "X_test"}.issubset(inputs):
        raise ValueError("X_train and X_test are required")
    result = {}
    for key, value in inputs.items():
        array = np.asarray(value)
        if array.ndim != 2 or array.dtype.kind not in "biuf" or not np.isfinite(array).all():
            raise ValueError(f"{key} must be a finite, real, two-dimensional array")
        result[key] = np.array(array, dtype=np.float64, copy=True)
    train, test = result["X_train"], result["X_test"]
    if len(train) < 1 or len(test) < 1 or train.shape[1] < 1 or test.shape[1] != train.shape[1]:
        raise ValueError("External protocol requires nonempty train/test features with matching columns")
    if ("A_train" in result) != ("A_test_train" in result):
        raise ValueError("Supply A_train and A_test_train together")
    if "A_test_test" in result and (not transductive or "A_train" not in result):
        raise ValueError("A_test_test requires graph inputs and explicit transductive=True")
    sizes = {"A_train": (len(train), len(train)), "A_test_train": (len(test), len(train)),
             "A_test_test": (len(test), len(test))}
    for key in (set(result) - {"X_train", "X_test"}):
        graph = result[key]
        if graph.shape != sizes[key] or not np.isin(graph, [0, 1]).all():
            raise ValueError(f"{key} has invalid binary graph dimensions or values")
        if key != "A_test_train" and (not np.array_equal(graph, graph.T) or np.any(np.diag(graph))):
            raise ValueError(f"{key} must be symmetric with a zero diagonal")
    return result


def load_model_input_directory(directory, *, transductive=False):
    """Read an explicitly provided label-free CSV directory, rejecting extras."""
    directory = Path(directory)
    allowed_names = {name + ".csv" for name in MODEL_INPUT_KEYS}
    entries = list(directory.iterdir())
    if any(not path.is_file() or path.name not in allowed_names for path in entries):
        raise ValueError("Input directory may contain only the five named model CSV files; no labels/config/results")
    return validate_model_inputs({path.stem: np.loadtxt(path, delimiter=",", ndmin=2)
                                 for path in entries}, transductive=transductive)


def _integer_vector(path, count, name):
    value = np.loadtxt(path, delimiter=",", ndmin=1)
    if value.shape != (count,) or not np.isfinite(value).all() or not np.equal(value, np.floor(value)).all():
        raise ValueError(f"{name} must contain exactly {count} finite integer labels")
    return value.astype(np.int64)


def _load_outputs(output, n_train, n_test):
    missing = [name for name in REQUIRED_OUTPUTS if not (output / name).is_file()]
    if missing:
        raise ValueError(f"Missing required adapter outputs: {missing}")
    metadata = json.loads((output / "score_columns.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((output / "diagnostics.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or not isinstance(diagnostics, dict):
        raise ValueError("Output metadata and diagnostics must be JSON objects")
    error = completion_error(diagnostics, require_completed=True)
    if error is not None:
        raise ValueError(error)
    if not {"cluster_ids", "score_type"}.issubset(metadata):
        raise ValueError("score_columns.json requires cluster_ids and score_type")
    cluster_ids = np.asarray(metadata["cluster_ids"])
    if (cluster_ids.ndim != 1 or not len(cluster_ids) or cluster_ids.dtype.kind not in "iu"
            or len(np.unique(cluster_ids)) != len(cluster_ids)):
        raise ValueError("cluster_ids must be a nonempty list of unique integer IDs")
    if not isinstance(metadata["score_type"], str) or not metadata["score_type"]:
        raise ValueError("score_type must be a nonempty string")
    scores = validate_scores(np.loadtxt(output / "test_scores.csv", delimiter=",", ndmin=2),
                             n_test, len(cluster_ids), metadata["score_type"], "test_scores.csv")
    result = dict(train_labels=_integer_vector(output / "train_labels.csv", n_train, "train_labels"),
                  test_labels=_integer_vector(output / "test_labels.csv", n_test, "test_labels"),
                  test_scores=scores, cluster_ids=cluster_ids.astype(np.int64),
                  score_type=metadata["score_type"], score_metadata=metadata, diagnostics=diagnostics)
    if metadata.get("alignment", "hard") == "soft":
        path = output / "train_scores.csv"
        if not path.is_file():
            raise ValueError("Soft alignment requires train_scores.csv")
        result["train_scores"] = validate_scores(np.loadtxt(path, delimiter=",", ndmin=2),
                                                n_train, len(cluster_ids), metadata["score_type"],
                                                "train_scores.csv")
    return result


def _expand_command(command, input_directory, output_directory, seed):
    if not isinstance(command, (list, tuple)) or not command or any(not isinstance(arg, str) or not arg for arg in command):
        raise ExternalAdapterError("missing_dependency", "Configure a nonempty command argument list for a user-installed external method")
    if not all(any(marker in arg for arg in command) for marker in ("{input}", "{output}")):
        raise ValueError("External command must contain {input} and {output} placeholders")
    substitutions = {"input": str(input_directory), "output": str(output_directory), "seed": str(seed)}
    resolved = []
    for argument in command:
        unknown = set(re.findall(r"\{([A-Za-z_][A-Za-z_0-9]*)\}", argument)) - set(substitutions)
        if unknown:
            raise ValueError(f"Unsupported command placeholders: {sorted(unknown)}")
        for key, value in substitutions.items():
            argument = argument.replace("{" + key + "}", value)
        resolved.append(argument)
    return resolved


def run_external(command, inputs, output_dir, seed, *, timeout=3600, cwd=None, transductive=False):
    """Run an explicit argv (shell=False), validate outputs, and never pass y.

    output_dir must be new or empty; existing experiment artifacts are never
    reused/overwritten. Inputs are generated in an isolated temporary directory
    containing only model CSVs. The supplied external process is trusted code:
    this export boundary is not an OS security sandbox or a no-leakage proof.
    """
    arrays = validate_model_inputs(inputs, transductive=transductive)
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if isinstance(timeout, bool) or not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a positive number of seconds")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite or reuse nonempty output directory: {output}")
    started = time.monotonic()
    bridge = dict(seed=int(seed), exported_keys=sorted(arrays), ground_truth_labels_exported=False,
                   transductive=bool(transductive), shell=False, numerical_reproduction_verified=False)
    try:
        with tempfile.TemporaryDirectory(prefix="gm3l-model-input-") as temporary:
            input_directory = Path(temporary)
            for name, array in arrays.items():
                np.savetxt(input_directory / (name + ".csv"), array, delimiter=",", fmt="%.17g")
            resolved = _expand_command(command, input_directory, output, int(seed))
            bridge["command"] = resolved
            try:
                process = subprocess.run(resolved, cwd=cwd, shell=False, capture_output=True,
                                         text=True, timeout=float(timeout), check=False)
            except FileNotFoundError as exc:
                raise ExternalAdapterError("missing_dependency", f"External executable or working directory unavailable: {exc}") from exc
            except subprocess.TimeoutExpired as exc:
                raise ExternalAdapterError("timeout", f"External command exceeded {timeout} seconds") from exc
            (output / "stdout.log").write_text(process.stdout or "", encoding="utf-8")
            (output / "stderr.log").write_text(process.stderr or "", encoding="utf-8")
            bridge["returncode"] = int(process.returncode)
            adapter_diagnostics = {}
            path = output / "diagnostics.json"
            if path.is_file():
                try:
                    adapter_diagnostics = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(adapter_diagnostics, dict):
                        adapter_diagnostics = {}
                except (OSError, ValueError):
                    pass
            completion_failure = completion_error(adapter_diagnostics, require_completed=True)
            if process.returncode != 0 or (adapter_diagnostics and completion_failure is not None):
                dependency_text = (process.stderr or "") + (process.stdout or "")
                is_missing = adapter_diagnostics.get("status") == "missing_dependency" or any(
                    text in dependency_text for text in ("ModuleNotFoundError", "No module named", "there is no package called",
                        "can't open file", "Fatal error: cannot open file", "not found in current path"))
                status = "missing_dependency" if is_missing else "failed"
                detail = f": {completion_failure}" if completion_failure is not None else ""
                raise ExternalAdapterError(status, f"External adapter did not complete (return code {process.returncode}){detail}; see stdout.log/stderr.log",
                                           {"adapter_diagnostics": adapter_diagnostics})
            try:
                result = _load_outputs(output, len(arrays["X_train"]), len(arrays["X_test"]))
            except (ValueError, OSError, TypeError, KeyError) as exc:
                raise ExternalAdapterError("invalid_output", str(exc),
                                           {"adapter_diagnostics": adapter_diagnostics}) from exc
            result["diagnostics"]["external_bridge"] = {**bridge, "runtime_seconds": time.monotonic() - started}
            # Do not claim convergence or change any native score/label.
            _json(output / "diagnostics.json", result["diagnostics"])
            return result
    except ExternalAdapterError as exc:
        exc.diagnostics["external_bridge"] = {**bridge, "runtime_seconds": time.monotonic() - started}
        _json(output / "diagnostics.json", exc.diagnostics)
        raise

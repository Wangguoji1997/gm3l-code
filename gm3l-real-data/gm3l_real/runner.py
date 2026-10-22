"""Run estimators using only exported feature/graph inputs, never true labels."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import sys
import numpy as np

from .model import (FitConfig, initialize_theta, fit_smle, fit_gmm, estep_log,
                    setup_adjacency, final_from_zeta, split_posteriors, gaussian_log_posterior)
from .validation import INPUT_NAMES, fingerprint


NATIVE = {"SMLE", "FINAL", "GMM", "GMM_tied"}


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def load_inputs(input_dir):
    """Explicit allowlist: no evaluation_only archive or label files are opened."""
    input_dir = Path(input_dir)
    cfg = json.loads((input_dir/"config.json").read_text(encoding="utf-8"))
    if cfg.get("protocol") not in {"inductive", "transductive"}:
        raise ValueError("Specify inductive or transductive protocol")
    required = INPUT_NAMES if cfg["protocol"] == "transductive" else INPUT_NAMES[:-1]
    inputs = {name: np.load(input_dir/(name+".npy"), allow_pickle=False) for name in required}
    X, Xt = inputs["X_train"], inputs["X_test"]
    if X.ndim != 2 or Xt.ndim != 2 or X.shape[1] != Xt.shape[1]:
        raise ValueError("Invalid train/test features")
    for name in ("X_train", "X_train_eval", "X_test"):
        if not np.isfinite(inputs[name]).all():
            raise ValueError("Features must be finite")
    if inputs["X_train_eval"].shape != X.shape:
        raise ValueError("Invalid X_train_eval shape")
    if inputs["A_train"].shape != (len(X), len(X)):
        raise ValueError("Invalid A_train shape")
    setup_adjacency(inputs["A_train"])
    across = inputs["A_test_train"]
    if across.shape != (len(Xt), len(X)) or not np.isin(across, [0, 1]).all():
        raise ValueError("Invalid A_test_train")
    if cfg["protocol"] == "transductive":
        if inputs["A_test_test"].shape != (len(Xt), len(Xt)):
            raise ValueError("Invalid A_test_test shape")
        setup_adjacency(inputs["A_test_test"])
    if not isinstance(cfg.get("K"), int) or not 2 <= cfg["K"] < len(X):
        raise ValueError("An explicit 2 <= K < n_train is required")
    return inputs, cfg


def source_fingerprint():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def export_result(output_dir, result, diagnostics):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for key in ("train_labels", "test_labels"):
        np.savetxt(output_dir/(key+".csv"), np.asarray(result[key], dtype=int), fmt="%d", delimiter=",")
    np.savetxt(output_dir/"test_scores.csv", result["test_scores"], delimiter=",", fmt="%.17g")
    if "train_scores" in result:
        np.savetxt(output_dir/"train_scores.csv", result["train_scores"], delimiter=",", fmt="%.17g")
    metadata = result.get("score_metadata", {})
    if not metadata:
        metadata = dict(cluster_ids=result["cluster_ids"], score_type=result["score_type"])
    write_json(output_dir/"score_columns.json", metadata)
    if result.get("model_arrays"):
        np.savez_compressed(output_dir/"model.npz", **result["model_arrays"])
    write_json(output_dir/"diagnostics.json", diagnostics)


def native_result(theta, train, test, *, alignment_source):
    return dict(train_labels=train.argmax(axis=1), test_labels=test.argmax(axis=1),
                train_scores=train, test_scores=test,
                model_arrays={key: value for key, value in theta.items() if isinstance(value, np.ndarray)},
                score_metadata=dict(cluster_ids=list(range(len(theta["alpha"]))), score_type="log_probability",
                                    alignment="soft", mapping="hungarian",
                                    alignment_source=alignment_source))


def fit_options(options):
    return FitConfig(**{key: options[key] for key in asdict(FitConfig()) if key in options})


def run_native(inputs, cfg, methods):
    """Return successful results and diagnostics for every requested native method."""
    X, A, seed, K = inputs["X_train"], inputs["A_train"], cfg["seed"], cfg["K"]
    results, reports = {}, {}
    common = dict(protocol=cfg["protocol"], seed=seed, fit_uses_labels=False, fit_uses_test=False)
    if {"SMLE", "FINAL"} & set(methods):
        opts = cfg.get("smle", {})
        config = fit_options(opts)
        attempts, candidates = [], []
        started = time.perf_counter()
        for offset in opts.get("seed_offsets", [0, 100003]):
            for initializer in opts.get("initializers", ["spectral", "x_kmeans"]):
                info = dict(initializer=initializer, seed_offset=offset)
                try:
                    initial = initialize_theta(X, A, K, seed+offset, config, initializer)
                    fit = fit_smle(X, A, initial, config)
                    info.update({key: fit[key] for key in ("converged", "failure", "iterations", "residual", "objective", "monotonicity_violation")})
                    if fit["converged"] and fit["failure"] is None:
                        candidates.append((fit["objective"], fit, initializer, offset))
                except Exception as exc:
                    info.update(converged=False, failure=f"{type(exc).__name__}: {exc}")
                attempts.append(info)
        base = dict(common, attempts=attempts, hyperparameters=dict(asdict(config),
                    seed_offsets=opts.get("seed_offsets", [0, 100003]), initializers=opts.get("initializers", ["spectral", "x_kmeans"])),
                    prediction_uses_test_test=cfg["protocol"] == "transductive")
        if not candidates:
            for method in {"SMLE", "FINAL"} & set(methods):
                reports[method] = dict(base, method=method, status="failed", reason="No converged SMLE start", runtime_seconds=time.perf_counter()-started)
        else:
            _, smle, initializer, offset = max(candidates, key=lambda item: item[0])
            base.update(selected_initializer=initializer, selected_seed_offset=offset)
            B, D = setup_adjacency(A)
            post = estep_log(X, B, D, smle)
            train_mapping_scores = post["log_zeta"]
            if "SMLE" in methods:
                try:
                    _, test = split_posteriors(inputs, smle, cfg["protocol"])
                    results["SMLE"] = native_result(smle, train_mapping_scores, test, alignment_source="SMLE training posterior")
                    reports["SMLE"] = dict(base, method="SMLE", status="complete", converged=True, runtime_seconds=time.perf_counter()-started)
                except Exception as exc:
                    reports["SMLE"] = dict(base, method="SMLE", status="failed", reason=f"{type(exc).__name__}: {exc}")
            if "FINAL" in methods:
                try:
                    final = final_from_zeta(X, A, post["zeta"])
                    _, test = split_posteriors(inputs, final, cfg["protocol"])
                    # FINAL retains the SMLE component order; mapping is evaluated later.
                    results["FINAL"] = native_result(final, train_mapping_scores, test, alignment_source="SMLE training posterior (shared with FINAL)")
                    reports["FINAL"] = dict(base, method="FINAL", status="complete", converged=True,
                                            construction="One closed-form update from SMLE training posterior", runtime_seconds=time.perf_counter()-started)
                except Exception as exc:
                    reports["FINAL"] = dict(base, method="FINAL", status="failed", reason=f"{type(exc).__name__}: {exc}")
    for method in ("GMM", "GMM_tied"):
        if method not in methods:
            continue
        opts = cfg.get("gmm_tied" if method == "GMM_tied" else "gmm", {})
        config = fit_options(opts)
        attempts, candidates = [], []
        started = time.perf_counter()
        for offset in opts.get("seed_offsets", [0, 100000]):
            info = dict(seed_offset=offset)
            try:
                fit = fit_gmm(X, K, seed+offset, config, tied=method == "GMM_tied")
                info.update({key: fit[key] for key in ("converged", "failure", "iterations", "residual", "objective")})
                if fit["converged"] and fit["failure"] is None:
                    candidates.append((fit["objective"], fit, offset))
            except Exception as exc:
                info.update(converged=False, failure=f"{type(exc).__name__}: {exc}")
            attempts.append(info)
        report = dict(common, method=method, attempts=attempts, hyperparameters=dict(asdict(config), seed_offsets=opts.get("seed_offsets", [0, 100000])), prediction_uses_test_test=False)
        if not candidates:
            reports[method] = dict(report, status="failed", reason="No converged GMM start", runtime_seconds=time.perf_counter()-started)
            continue
        _, theta, offset = max(candidates, key=lambda item: item[0])
        train, _ = gaussian_log_posterior(X, theta)
        test, _ = gaussian_log_posterior(inputs["X_test"], theta)
        results[method] = native_result(theta, train, test, alignment_source="GMM training posterior")
        reports[method] = dict(report, status="complete", converged=True, selected_seed_offset=offset, runtime_seconds=time.perf_counter()-started)
    return results, reports


def run_split(input_dir, output_dir, methods):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Specify unique method names")
    for method in methods:
        if not method or Path(method).name != method or method in {".", ".."}:
            raise ValueError("Method names must be directory basenames")
        if (output_dir/method).exists():
            raise FileExistsError(f"Output already exists: {output_dir/method}; use a new run directory")
    inputs, cfg = load_inputs(input_dir)
    identity = fingerprint(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results, reports = {}, {}
    for group in ({"SMLE", "FINAL"}, {"GMM"}, {"GMM_tied"}):
        requested = set(methods) & group
        if not requested:
            continue
        try:
            group_results, group_reports = run_native(inputs, cfg, requested)
            results.update(group_results)
            reports.update(group_reports)
        except Exception as exc:
            for method in requested:
                reports[method] = dict(method=method, status="failed", protocol=cfg["protocol"],
                                       reason=f"{type(exc).__name__}: {exc}")
    for method in methods:
        if method in NATIVE:
            continue
        started = time.perf_counter()
        effective_seed = cfg["seed"]
        try:
            if method in cfg.get("external", {}):
                from .external import run_external
                opts = cfg["external"][method]
                input_keys = opts.get("input_keys", [key for key in inputs if key != "X_train_eval"])
                if not isinstance(input_keys, list) or not {"X_train", "X_test"}.issubset(input_keys) or not set(input_keys).issubset(set(inputs)-{"X_train_eval"}):
                    raise ValueError("External input_keys must name only available model X/A arrays")
                external_inputs = {key: inputs[key] for key in input_keys}
                effective_seed += opts.get("seed_offset", 0)
                command = [part.replace("{K}",str(cfg["K"])).replace("{python}",sys.executable)
                           .replace("{package}",str(Path(__file__).resolve().parents[1])) for part in opts["command"]]
                result = run_external(command, external_inputs, output_dir/method,
                                      effective_seed,
                                      timeout=opts.get("timeout", 3600), cwd=opts.get("cwd"),
                                      transductive=cfg["protocol"] == "transductive")
            else:
                from .baselines import fit_baseline
                result = fit_baseline(method, inputs["X_train"], inputs["X_test"], cfg["K"], cfg["seed"], cfg.get("baselines", {}).get(method, {}))
            results[method] = result
            diagnostic = result.get("diagnostics", {})
            adapter_status = diagnostic.get("status", "complete")
            status = "complete" if adapter_status in {"complete", "completed", "ok", "success", "converged"} else adapter_status
            reports[method] = dict(diagnostic, method=method, status=status, adapter_status=adapter_status, protocol=cfg["protocol"],
                                   seed=effective_seed, runtime_seconds=time.perf_counter()-started)
        except Exception as exc:
            reports[method] = dict(getattr(exc, "diagnostics", {}), method=method,
                                   status=getattr(exc, "status", "failed"), protocol=cfg["protocol"], seed=effective_seed,
                                   reason=f"{type(exc).__name__}: {exc}", runtime_seconds=time.perf_counter()-started)
    for method in methods:
        report = dict(reports[method], input_sha256=identity, source_sha256=source_fingerprint())
        destination = output_dir/method
        if method in results:
            export_result(destination, results[method], report)
        else:
            destination.mkdir(parents=True, exist_ok=True)
            write_json(destination/"diagnostics.json", report)
        reports[method] = report
    return reports

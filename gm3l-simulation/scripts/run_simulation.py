#!/usr/bin/env python3
"""Generate paired simulations and fit SMLE, ORACLE, FINAL, and GMM."""
from __future__ import annotations

import os

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

from gm3l import get_backend, strict_smle
from gm3l.clustering import fit_gmm, star_log_mse
from gm3l.complete_data import complete_data_estimate
from gm3l.data import make_parameters, simulate_data
from gm3l.exact_spectral_init import initialize_spectral_theta
from gm3l.final_estimator import final_from_zeta

KEYS = ("alpha", "mu", "sigma", "pi")
METHODS = ("SMLE", "ORACLE", "FINAL", "GMM")


class FitFailure(FloatingPointError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_config(path):
    cfg = json.loads(path.read_text())
    required = {"name", "K", "p", "n_grid", "replications", "data_seed", "init_seed",
                "tolerance", "max_iter", "chunk_iterations"}
    if set(cfg) != required:
        raise ValueError(f"Configuration keys must be {sorted(required)}")
    if (cfg["K"], cfg["p"]) != (3, 5):
        raise ValueError("This study uses the fixed K=3, p=5 manuscript design")
    integers = ("replications", "data_seed", "init_seed", "max_iter", "chunk_iterations")
    if any(type(cfg[key]) is not int or cfg[key] <= 0 for key in integers):
        raise ValueError("Seeds, replication counts, and iteration limits must be positive integers")
    ns = cfg["n_grid"]
    if not ns or any(type(n) is not int or n <= cfg["p"] + cfg["K"] for n in ns):
        raise ValueError("Sample sizes must be integers larger than p+K")
    if ns != sorted(set(ns)):
        raise ValueError("n_grid must be strictly increasing")
    if not np.isfinite(cfg["tolerance"]) or cfg["tolerance"] <= 0:
        raise ValueError("tolerance must be positive and finite")
    return cfg


def alignment(labels, zeta):
    k = zeta.shape[1]
    counts = np.zeros((k, k), dtype=int)
    np.add.at(counts, (labels, zeta.argmax(axis=1)), 1)
    rows, columns = linear_sum_assignment(-counts)
    order = np.empty(k, dtype=int)
    order[rows] = columns
    return order


def parameter_mse(truth, theta, order=None):
    if order is not None:
        theta = {"alpha": theta["alpha"][order], "mu": theta["mu"][order],
                 "sigma": theta["sigma"][order], "pi": theta["pi"][np.ix_(order, order)]}
    return {"mse_" + key: float(np.mean((theta[key] - truth[key]) ** 2)) for key in KEYS}


def fit_smle(x, adjacency, cfg, seed, engine):
    config = strict_smle.FitConfig(max_iter=cfg["chunk_iterations"], tolerance=cfg["tolerance"])
    theta, initialization = initialize_spectral_theta(
        x, adjacency, cfg["K"], seed, config, return_diagnostics=True)
    initial = {key: value.copy() for key, value in theta.items()}
    iterations, history = 0, []
    status = "max_iter"
    while iterations < cfg["max_iter"]:
        chunk = strict_smle.FitConfig(
            max_iter=min(cfg["chunk_iterations"], cfg["max_iter"] - iterations),
            tolerance=cfg["tolerance"])
        fit = engine.fit_smle(x, adjacency, theta, chunk)
        theta = {key: fit[key] for key in KEYS}
        iterations += int(fit["iterations"])
        history.extend(fit["objective_history"] if not history else fit["objective_history"][1:])
        if fit["failure"]:
            raise FitFailure(fit["failure"], dict(
                iterations=iterations, initialization=initialization,
                objective_history=history, last_parameters=theta))
        posterior = engine.estep(x, adjacency, theta)
        residual = float(np.linalg.norm(
            strict_smle.parameter_vector(engine.mstep(x, posterior))
            - strict_smle.parameter_vector(theta)))
        if fit["converged"] and residual <= cfg["tolerance"]:
            status = "ok"
            break
        if int(fit["iterations"]) == 0:
            raise FloatingPointError("SMLE made no iteration progress")
    info = dict(status=status, iterations=iterations, fixed_point_residual=residual,
                objective=posterior["objective"], initialization=initialization,
                monotonicity_violation=max(0.0, float(-np.min(np.diff(history)))) if len(history) > 1 else 0.0)
    return theta, posterior, info, initial


def fit_one(labels, x, adjacency, truth, rep, n, cfg):
    common = dict(replication=rep, n=n)
    rows = {name: dict(**common, estimator=name, status="failed") for name in METHODS}
    estimates = {}
    b = None
    try:
        theta, diagnostics = complete_data_estimate(labels, x, adjacency, cfg["K"])
        rows["ORACLE"].update(status="ok" if diagnostics["global_likelihood_feasible"] else "infeasible",
                              **parameter_mse(truth, theta), diagnostics=diagnostics)
        estimates.update({"ORACLE_" + key: theta[key] for key in KEYS})
    except Exception as exc:
        rows["ORACLE"]["error"] = f"{type(exc).__name__}: {exc}"

    try:
        engine = get_backend(cfg["backend"])
        b = strict_smle._off_diagonal(adjacency)
        product = np.matmul
        if cfg["backend"] == "accelerate":
            from gm3l.accelerate_blas import dgemm
            product = dgemm
        theta, posterior, info, initial = fit_smle(
            x, b, cfg, cfg["init_seed"] + 1009 * rep + 17 * n, engine)
        order = alignment(labels, posterior["zeta"])
        rows["SMLE"].update(**info, **parameter_mse(truth, theta, order),
                            **strict_smle.parameter_diagnostics(theta),
                            log_mse_zeta=star_log_mse(x, b, theta, labels, order, backend=cfg["backend"]),
                            evaluation_permutation=order.tolist())
        estimates.update({"SMLE_" + key: theta[key] for key in KEYS})
        estimates.update({"initial_" + key: initial[key] for key in KEYS})
        estimates["evaluation_permutation"] = order
    except Exception as exc:
        rows["SMLE"].update(status="failed", error=f"{type(exc).__name__}: {exc}",
                            **getattr(exc, "diagnostics", {}))
        rows["FINAL"]["status"] = "source_smle_failed"
    else:
        try:
            # FINAL uses soft SMLE weights; true labels enter evaluation only.
            theta_final, diagnostics = final_from_zeta(x, b, posterior["zeta"], product)
            rows["FINAL"].update(
                status="ok" if info["status"] == "ok" else "source_smle_not_converged",
                **parameter_mse(truth, theta_final, order),
                **strict_smle.parameter_diagnostics(theta_final), diagnostics=diagnostics,
                log_mse_zeta=star_log_mse(x, b, theta_final, labels, order, backend=cfg["backend"]),
                evaluation_permutation=order.tolist())
            estimates.update({"FINAL_" + key: theta_final[key] for key in KEYS})
        except Exception as exc:
            rows["FINAL"].update(status="failed", error=f"{type(exc).__name__}: {exc}")
    del b
    try:
        rows["GMM"].update(fit_gmm(x, labels, rep, n))
    except Exception as exc:
        rows["GMM"].update(status="failed", error=f"{type(exc).__name__}: {exc}")
    return list(rows.values()), estimates


def work(task):
    rep, cfg, directory = task
    out = Path(directory)
    files = [out / "replicates" / f"rep_{rep:04d}_n{n}.json" for n in cfg["n_grid"]]
    if all(path.exists() for path in files):
        return rep
    truth = make_parameters()
    seed = [cfg["data_seed"], cfg["K"], cfg["p"], rep]
    with threadpool_limits(limits=1):
        labels, x, adjacency = simulate_data(max(cfg["n_grid"]), truth, np.random.default_rng(seed))
        write_json(out / "generation" / f"rep_{rep:04d}.json", dict(
            replication=rep, seed=seed, n=max(cfg["n_grid"]),
            adjacency_sha256=hashlib.sha256(memoryview(adjacency)).hexdigest(),
            class_counts=np.bincount(labels, minlength=cfg["K"]).tolist()))
        for n, path in zip(cfg["n_grid"], files):
            if path.exists():
                continue
            started = time.perf_counter()
            rows, estimates = fit_one(labels[:n], x[:n], adjacency[:n, :n], truth, rep, n, cfg)
            if cfg["save_estimates"]:
                np.savez_compressed(path.with_suffix(".npz"), **estimates)
            write_json(path, dict(replication=rep, n=n, seconds=time.perf_counter() - started, rows=rows))
            print(json.dumps(dict(replication=rep, n=n,
                                  statuses={row["estimator"]: row["status"] for row in rows})), flush=True)
    return rep


def summarize(out, cfg):
    rows = []
    for rep in range(1, cfg["replications"] + 1):
        for n in cfg["n_grid"]:
            path = out / "replicates" / f"rep_{rep:04d}_n{n}.json"
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            if (record["replication"], record["n"]) != (rep, n):
                raise ValueError(f"Result identity mismatch: {path.name}")
            if [row["estimator"] for row in record["rows"]] != list(METHODS):
                raise ValueError(f"Result methods mismatch: {path.name}")
            for row in record["rows"]:
                rows.append({**{key: value for key, value in row.items()
                                if not isinstance(value, (list, dict))}, "fit_seconds": record["seconds"]})
    if rows:
        fields = sorted(set().union(*(row.keys() for row in rows)))
        temporary = out / "all_results.csv.tmp"
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(out / "all_results.csv")
    expected = cfg["replications"] * len(cfg["n_grid"]) * len(METHODS)
    failures = [{key: row.get(key) for key in ("replication", "n", "estimator", "status", "error")}
                for row in rows if row["status"] != "ok"]
    report = dict(complete=len(rows) == expected, expected_records=expected,
                  recorded=len(rows), non_ok=failures)
    write_json(out / "status.json", report)
    return report


def prepare(out, cfg):
    sources = [*sorted((ROOT / "gm3l").glob("*.py")), Path(__file__)]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    environment = dict(python=platform.python_version(), platform=platform.platform(),
                       packages={name: version(name) for name in
                                 ("numpy", "scipy", "scikit-learn", "matplotlib", "threadpoolctl")})
    if out.exists() and any(out.iterdir()):
        for name, expected in (("configuration.json", cfg), ("source_hashes.json", hashes),
                               ("environment.json", environment)):
            path = out / name
            if not path.exists() or json.loads(path.read_text()) != expected:
                raise ValueError(f"Existing output has a different {name}; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("replicates", "generation"):
        (out / name).mkdir(exist_ok=True)
    for name, value in (("configuration.json", cfg), ("source_hashes.json", hashes),
                        ("environment.json", environment), ("truth.json", make_parameters())):
        write_json(out / name, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "smoke.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "smoke")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--backend", choices=("numpy", "accelerate"), default="numpy")
    parser.add_argument("--replications", type=int, help="Override the replication count for a trial run")
    parser.add_argument("--save-estimates", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.workers < 1 or (args.replications is not None and args.replications < 1):
        parser.error("workers and replications must be positive")
    if args.backend == "accelerate" and sys.platform != "darwin":
        parser.error("Accelerate is available only on macOS; use --backend numpy")
    if args.replications is not None:
        cfg["replications"] = args.replications
    cfg.update(backend=args.backend, save_estimates=args.save_estimates)
    out = args.output.resolve()
    prepare(out, cfg)
    tasks = [(rep, cfg, str(out)) for rep in range(1, cfg["replications"] + 1)]
    started = time.perf_counter()
    try:
        if args.workers == 1:
            for task in tasks:
                work(task)
                summarize(out, cfg)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for future in as_completed(pool.submit(work, task) for task in tasks):
                    future.result()
                    summarize(out, cfg)
    except BaseException as exc:
        report = summarize(out, cfg)
        report.update(interrupted=True, error=f"{type(exc).__name__}: {exc}",
                      elapsed_seconds=time.perf_counter() - started)
        write_json(out / "completion.json", report)
        raise
    report = summarize(out, cfg)
    report["elapsed_seconds"] = time.perf_counter() - started
    write_json(out / "completion.json", report)
    print(json.dumps(report), flush=True)
    return 0 if report["complete"] and not report["non_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

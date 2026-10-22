#!/usr/bin/env python3
"""CLSBM full-graph spectral representation with training-only centroids.

This is the existing split-evaluation adapter, not a training-only representation
fit or an author-native out-of-sample predictor. It requires all five model CSVs,
including A_test_test, and never opens a labels archive or experiment config.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(input_dir, output_dir, seed, K):
    import numpy as np
    from scipy.linalg import eigh
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits
    from gm3l_real.external import load_model_input_directory

    started = time.monotonic()
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    inputs = load_model_input_directory(input_dir, transductive=True)
    required = {"X_train", "X_test", "A_train", "A_test_train", "A_test_test"}
    if set(inputs) != required:
        raise ValueError("CLSBM is transductive only and requires all five model CSVs, including A_test_test")
    xt, xe = inputs["X_train"], inputs["X_test"]
    if not isinstance(K, int) or isinstance(K, bool) or not 1 <= K <= len(xt):
        raise ValueError("K must be an explicit integer with 1 <= K <= n_train")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("Output directory must be new or empty; existing results are never overwritten")
    x = np.vstack([xt, xe])
    a = np.block([[inputs["A_train"], inputs["A_test_train"].T],
                  [inputs["A_test_train"], inputs["A_test_test"]]])
    # Preserve the source adapter equations, random draw, eigenspace selection
    # and K-means settings. Do not normalize or regularize this representation.
    weight = float(np.random.default_rng(seed).uniform())
    with threadpool_limits(limits=1):
        s = weight * a + x @ x.T / len(x)
        values, vectors = eigh(s, check_finite=False)
        order = np.argsort(np.abs(values))[::-1][:K]
        basis = vectors[:, order]
        embedding = s @ basis
        km = KMeans(n_clusters=K, n_init=30, random_state=seed).fit(embedding[:len(xt)])
        scores = -np.sum((embedding[len(xt):, None] - km.cluster_centers_[None]) ** 2, axis=2)
    if not np.isfinite(scores).all():
        raise FloatingPointError("CLSBM produced nonfinite distance scores")
    np.savez_compressed(output_dir / "model.npz", weight=weight, eigenvalues=values[order],
                        U_full=basis, centroids=km.cluster_centers_, train_labels=km.labels_ + 1)
    np.savetxt(output_dir / "train_labels.csv", km.labels_ + 1, fmt="%d", delimiter=",")
    np.savetxt(output_dir / "test_labels.csv", scores.argmax(axis=1) + 1, fmt="%d", delimiter=",")
    np.savetxt(output_dir / "test_scores.csv", scores, fmt="%.17g", delimiter=",")
    metadata = dict(cluster_ids=list(range(1, K + 1)), score_type="decision",
                    alignment="hard", mapping="hungarian",
                    score_definition="negative squared distance to training centroids in full-graph CLSBM spectral coordinates")
    write_json(output_dir / "score_columns.json", metadata)
    diagnostics = dict(method="CLSBM2025", status="complete", completed=True,
                       runtime_seconds=time.monotonic() - started, seed=seed, K=K, p=x.shape[1],
                       convergence=dict(kmeans_iterations=int(km.n_iter_), global_optimum_certified=False),
                       prediction_extension=dict(uses_test_test=True, representation_fit="full unlabeled graph",
                                                 centroid_fit="training rows only", training_only_representation=False),
                       original_equation="S=w*A+X*X^T/n_full, U=top K eigenvectors by absolute eigenvalue, Q=S*U",
                       original_weight=weight, ground_truth_labels_read=False,
                       warnings=["Explicit transductive representation exception: spectral basis uses test observations.",
                                 "Centroid fitting uses training nodes only; this extends native full-graph clustering to a split evaluation.",
                                 "Distance confidence is not a calibrated posterior probability."])
    write_json(output_dir / "diagnostics.json", diagnostics)
    return diagnostics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Isolated directory of five model CSVs only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--K", type=int, required=True, help="Explicit component count; never inferred from labels")
    args = parser.parse_args(argv)
    # Do not overwrite a previous diagnostics file even when validation fails.
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.exit(2, "Output exists and is not an empty directory; choose a new output directory.\n")
    try:
        run(args.input, args.output, args.seed, args.K)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        status = "missing_dependency" if isinstance(exc, ImportError) else "failed"
        write_json(args.output / "diagnostics.json", dict(method="CLSBM2025", status=status,
                   completed=False, reason=f"{type(exc).__name__}: {exc}"))
        print(f"CLSBM adapter failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"Completed CLSBM split adapter: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

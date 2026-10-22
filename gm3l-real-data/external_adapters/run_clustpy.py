#!/usr/bin/env python3
"""Portable DEC/DipEncoder CSV bridge; optional clustpy==0.0.3, no bundled library.

Source basis (local adapter code only, not third-party implementation):
  Xonly_realdata_comparison_20261009/adapters/dec_dip.py
  Xonly_realdata_comparison_20261009/auc_adapters/dip_scores.py

Preserves the original native estimator calls, fixed default training budget,
native DEC Student-t q, and DipEncoder threshold-vote scoring. Historical
progress-observation hooks are omitted; no loss, optimizer, label update, or
stopping rule is replaced. This bridge does not read old datasets/results or
import those history-dependent adapters. --help needs only the standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import random
import sys
import time


SUPPORTED_VERSION = "0.0.3"
NATIVE_DIP_SOURCE_SHA256 = "6e633c5c28e1cde6ee9d3519664b76bcee5d3c956dce1cd54be00cf9d7703500"
HISTORICAL_ADAPTER_SHA256 = "d1a8fdba41bdaf4d378dd49c539e0005b03cd0f24fed9987bcf47591699f4e7d"
SCORE_RULE_VERSION = "dipencoder_native_pairwise_vote_fraction_v1"
ALLOWED_INPUT_NAMES = frozenset(name + ".csv" for name in
    ("X_train", "X_test", "A_train", "A_test_train", "A_test_test"))


class DependencyError(RuntimeError):
    status = "missing_dependency"


def _source_hash(module):
    return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()


def _dependencies(method):
    """Check the original inspected version before opening any model-input CSV."""
    try:
        installed = version("clustpy")
    except PackageNotFoundError as exc:
        raise DependencyError("Install optional clustpy==0.0.3 and its PyTorch dependencies; no fallback is used") from exc
    if installed != SUPPORTED_VERSION:
        raise DependencyError(f"This preserved adapter requires clustpy=={SUPPORTED_VERSION}, found {installed}")
    try:
        import numpy as np
        import torch
        from clustpy.deep import DEC, DipEncoder
        from clustpy.deep.neural_networks import FeedforwardAutoencoder
        from threadpoolctl import threadpool_limits
        native = importlib.import_module("clustpy.deep.dec" if method == "DEC" else "clustpy.deep.dipencoder")
    except ImportError as exc:
        raise DependencyError(f"Missing optional ClustPy/PyTorch dependency: {exc}") from exc
    if method == "DipEncoder" and _source_hash(native) != NATIVE_DIP_SOURCE_SHA256:
        raise RuntimeError("Unsupported DipEncoder source: installed module differs from the originally inspected native vote-rule implementation")
    return np, torch, DEC, DipEncoder, FeedforwardAutoencoder, threadpool_limits, native


def _read_features(directory, np):
    """Only open two named feature CSVs; never open graph or label files."""
    directory = Path(directory)
    files = list(directory.iterdir())
    if any(not path.is_file() or path.name not in ALLOWED_INPUT_NAMES for path in files):
        raise ValueError("Input directory may contain only model CSV files; label/config/result files are prohibited")
    arrays = []
    for name in ("X_train", "X_test"):
        value = np.loadtxt(directory / (name + ".csv"), delimiter=",", ndmin=2)
        with np.errstate(over="ignore", invalid="ignore"):
            value = np.array(value, dtype=np.float32, order="C", copy=True)
        if value.ndim != 2 or value.shape[1] == 0 or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain finite real features after float32 conversion")
        arrays.append(value)
    train, test = arrays
    if len(train) < 2 or len(test) < 1 or train.shape[1] != test.shape[1]:
        raise ValueError("Require at least two training rows, a nonempty test set, and matching feature columns")
    return train, test


def _native_vote_counts(model, X, native, np):
    """Same pair order, projections, strict comparisons and ties as old supplement."""
    K = int(model.n_clusters_out_)
    if not np.isfinite(model.projection_axes_).all():
        raise FloatingPointError("Nonfinite native DipEncoder projection axes")
    if any(not np.isfinite(item[0]) for item in model.projection_thresholds_):
        raise FloatingPointError("Nonfinite native DipEncoder projection thresholds")
    embedded = model.transform(X)
    if not np.isfinite(embedded).all():
        raise FloatingPointError("Nonfinite native DipEncoder embedding")
    votes = np.zeros((len(X), K), dtype=np.float64)
    for m in range(K - 1):
        for n in range(m + 1, K):
            index = model.index_dict_[(m, n)]
            projection_vector = model.projection_axes_[index]
            threshold, cluster_tuple = model.projection_thresholds_[index]
            if m not in cluster_tuple or n not in cluster_tuple:
                raise RuntimeError("Invalid fitted native cluster-pair threshold")
            projected = np.matmul(embedded, projection_vector)
            votes[projected < threshold, cluster_tuple[0]] += 1
            votes[projected >= threshold, cluster_tuple[1]] += 1
    native_labels = native._predict_using_thresholds(embedded, model.projection_axes_,
        model.projection_thresholds_, K, model.index_dict_)
    if not np.array_equal(votes.argmax(axis=1), native_labels):
        raise RuntimeError("Exposed vote argmax differs from native DipEncoder predictor")
    if not np.all(votes.sum(axis=1) == K * (K - 1) // 2):
        raise RuntimeError("A native pair did not contribute exactly one vote")
    if np.any(votes < 0) or np.any(votes > K - 1):
        raise RuntimeError("Native vote count outside its possible range")
    return votes


def fit_clustpy(method, X_train, X_test, K, seed, dependencies):
    """Label-free training using unchanged source-default estimator arguments."""
    np, torch, DEC, DipEncoder, Autoencoder, threadpool_limits, native = dependencies
    if not 2 <= K <= len(X_train) or not 0 <= seed < 2**32:
        raise ValueError("Require 2 <= K <= n_train and 0 <= seed < 2**32")
    layers = [X_train.shape[1], 64, 64, 10]
    batch_size = 256 if method == "DEC" else 0
    kwargs = dict(n_clusters=K, batch_size=batch_size, pretrain_epochs=100,
        clustering_epochs=150, embedding_size=10, optimizer_class=torch.optim.Adam,
        pretrain_optimizer_params={"lr": 1e-3}, clustering_optimizer_params={"lr": 1e-4},
        clustering_loss_weight=1.0, device=torch.device("cpu"), random_state=seed,
        neural_network=(Autoencoder, {"layers": layers}),
        augmentation_invariance=False, custom_dataloaders=None)
    if method == "DEC":
        model = DEC(**kwargs, alpha=1.0)
    else:
        model = DipEncoder(**kwargs, ssl_loss_weight=None, max_cluster_size_diff_factor=3.0)
    old_threads, numpy_rng, python_rng = torch.get_num_threads(), np.random.get_state(), random.getstate()
    started = time.perf_counter()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]), threadpool_limits(limits=1):
            model.fit(X_train, y=None)
            fit_seconds = time.perf_counter() - started
            network = model.neural_network_trained_
            network.eval()
            before = {name: tensor.detach().cpu().clone() for name, tensor in network.state_dict().items()}
            with torch.no_grad():
                if method == "DEC":
                    centers = torch.from_numpy(model.dec_cluster_centers_)
                    def q(X):
                        embedded = torch.from_numpy(model.transform(X))
                        return native._dec_predict(centers, embedded, model.alpha, weights=None).cpu().numpy()
                    train_scores, test_scores = q(X_train), q(X_test)
                    train_labels = train_scores.argmax(axis=1).astype(np.int64)
                    test_labels = test_scores.argmax(axis=1).astype(np.int64)
                    if not np.array_equal(train_labels, model.dec_labels_):
                        raise RuntimeError("Native DEC q argmax differs from dec_labels_")
                    for scores in (train_scores, test_scores):
                        if not np.isfinite(scores).all() or not np.allclose(scores.sum(1), 1, atol=1e-5):
                            raise FloatingPointError("Invalid native DEC Student-t probabilities")
                    score_type = "probability"
                    scoring = dict(definition="Native DEC Student-t q from _dec_predict with dec_cluster_centers_",
                        probabilities_available=True,
                        extra_final_kmeans_not_used=True,
                        native_vs_extra_final_kmeans_disagreement=float(np.mean(train_labels != model.labels_)))
                else:
                    if model.n_clusters_out_ != K:
                        raise RuntimeError("Native fitted DipEncoder cluster count differs from requested K")
                    train_labels = np.asarray(model.labels_, dtype=np.int64)
                    test_labels = np.asarray(model.predict(X_test), dtype=np.int64)
                    train_votes = _native_vote_counts(model, X_train, native, np)
                    test_votes = _native_vote_counts(model, X_test, native, np)
                    for label, votes in ((train_labels, train_votes), (test_labels, test_votes)):
                        if not np.array_equal(votes.argmax(axis=1), label):
                            raise RuntimeError("Native vote argmax differs from retained fitted/predicted labels")
                    train_scores, test_scores = train_votes / (K - 1), test_votes / (K - 1)
                    score_type = "vote_fraction"
                    scoring = dict(definition="Native pairwise threshold votes for cluster c divided by K-1",
                        score_rule_version=SCORE_RULE_VERSION, probabilities_available=False,
                        score_range=[0.0, 1.0], per_row_score_sum=K / 2.0,
                        ties="threshold equality votes right; argmax ties lowest cluster ID",
                        native_vote_argmax_matches_retained_labels=True)
            after = network.state_dict()
            if any(not torch.equal(tensor, after[name].detach().cpu()) for name, tensor in before.items()):
                raise RuntimeError("Scoring changed fitted network state")
            arrays = {"network_state__" + name: tensor.detach().cpu().numpy().copy()
                      for name, tensor in after.items()}
            arrays.update(encoder_layers=np.asarray(layers, dtype=np.int64),
                          decoder_layers=np.asarray(layers[::-1], dtype=np.int64))
            if method == "DEC":
                arrays["dec_cluster_centers"] = np.array(model.dec_cluster_centers_, copy=True)
            else:
                ordered_pairs = [pair for pair, _ in sorted(model.index_dict_.items(), key=lambda item: item[1])]
                arrays.update(projection_axes=np.array(model.projection_axes_, copy=True),
                    projection_index_pairs=np.asarray(ordered_pairs, dtype=np.int64),
                    projection_threshold_values=np.asarray([item[0] for item in model.projection_thresholds_], dtype=np.float64),
                    projection_threshold_cluster_pairs=np.asarray([item[1] for item in model.projection_thresholds_], dtype=np.int64))
    finally:
        torch.set_num_threads(old_threads)
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)
    for labels, scores, count in ((train_labels, train_scores, len(X_train)), (test_labels, test_scores, len(X_test))):
        if labels.shape != (count,) or labels.min() < 0 or labels.max() >= K:
            raise RuntimeError("Invalid native cluster labels")
        if scores.shape != (count, K) or not np.isfinite(scores).all():
            raise RuntimeError("Invalid native score matrix")
    if any(array.dtype.hasobject for array in arrays.values()):
        raise RuntimeError("Model arrays unexpectedly require pickle")
    dependency_versions = {}
    for name in ("clustpy", "torch", "torchvision", "numpy", "scikit-learn", "scipy", "tqdm", "threadpoolctl"):
        try:
            dependency_versions[name] = version(name)
        except PackageNotFoundError:
            dependency_versions[name] = "not installed"
    metadata = dict(cluster_ids=list(range(K)), score_type=score_type, alignment="hard",
                    mapping="hungarian")
    diagnostics = dict(method=method, completed=True, status="completed", training_budget_completed=True,
        seed=seed, K=K, dependencies=dependency_versions, fit_input="X_train_only", fit_y=None,
        fit_uses_test=False, graph_files_read=False, ground_truth_labels_received=False,
        configuration=dict(pretrain_epochs=100, clustering_epochs=150, embedding_size=10,
            encoder_layers=layers, decoder_layers=layers[::-1], batch_size_requested=batch_size,
            batch_size_effective=256 if method == "DEC" else 25 * K,
            optimizer="Adam", pretrain_optimizer_params={"lr": 1e-3},
            clustering_optimizer_params={"lr": 1e-4}, clustering_loss_weight=1.0,
            device="cpu", num_threads=1, augmentation_invariance=False,
            alpha=1.0 if method == "DEC" else None,
            ssl_loss_weight=None if method == "DipEncoder" else "not applicable",
            max_cluster_size_diff_factor=3.0 if method == "DipEncoder" else None),
        preprocessing="none inside bridge; caller must fit preprocessing on training data only",
        convergence=dict(converged=None, stopping_rule="native fixed 100/150 epoch budgets",
            claim="Not established; historical observation-only loss/progress hooks were not copied"),
        scoring=scoring, fitted_network_unchanged_by_scoring=True,
        training_source_adapter="Xonly_realdata_comparison_20261009/adapters/dec_dip.py",
        scoring_source_adapter="Xonly_realdata_comparison_20261009/auc_adapters/dip_scores.py" if method == "DipEncoder" else "dec_dip.py native Student-t q branch",
        historical_training_adapter_sha256=HISTORICAL_ADAPTER_SHA256,
        installed_native_source_sha256=_source_hash(native),
        bridge_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_saved_predictions_compared=False, numerical_reproduction_verified=False,
        train_cluster_counts=np.bincount(train_labels, minlength=K).tolist(),
        test_cluster_counts=np.bincount(test_labels, minlength=K).tolist(),
        empty_train_cluster_ids=np.flatnonzero(np.bincount(train_labels, minlength=K) == 0).tolist(),
        fit_seconds=fit_seconds, total_seconds=time.perf_counter() - started)
    return dict(train_labels=train_labels, test_labels=test_labels, train_scores=train_scores,
                test_scores=test_scores, score_metadata=metadata, diagnostics=diagnostics, model_arrays=arrays)


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Label-free model CSV directory; only X_train/X_test are opened")
    parser.add_argument("--output", type=Path, required=True, help="New or empty output directory")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--K", type=int, required=True, help="Explicit known cluster-count prior; never inferred from labels")
    parser.add_argument("--method", choices=("DEC", "DipEncoder"), required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("Output directory must be empty; existing artifacts are preserved")
    try:
        dependencies = _dependencies(args.method)
        np = dependencies[0]
        train, test = _read_features(args.input, np)
        result = fit_clustpy(args.method, train, test, args.K, args.seed, dependencies)
        for name in ("train_labels", "test_labels", "train_scores", "test_scores"):
            np.savetxt(output / (name + ".csv"), result[name], delimiter=",",
                       fmt="%d" if name.endswith("labels") else "%.17g")
        np.savez_compressed(output / "model.npz", **result["model_arrays"])
        _write_json(output / "score_columns.json", result["score_metadata"])
        _write_json(output / "diagnostics.json", result["diagnostics"])
    except Exception as exc:
        status = getattr(exc, "status", "failed")
        _write_json(output / "diagnostics.json", dict(method=args.method, completed=False, status=status,
            seed=args.seed, K=args.K, error=f"{type(exc).__name__}: {exc}",
            third_party_implementation_bundled=False, numerical_reproduction_verified=False))
        print(f"{status}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""New label-independent splits and training-only PCA; no cached split reuse."""
from __future__ import annotations

from pathlib import Path
import json
import math
import numpy as np
from sklearn.decomposition import PCA

from .data_io import Dataset, validate_ids, validate_features, validate_adjacency, validate_labels, save_archive


def split_indices(n, *, seed, train_fraction):
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(train_fraction, (int, float)) or isinstance(train_fraction, bool) or not 0 < train_fraction < 1:
        raise ValueError("train_fraction must lie strictly between 0 and 1")
    n_train = math.floor(n * train_fraction)
    if n_train < 1 or n_train >= n:
        raise ValueError("The split must contain at least one training and one test record")
    permutation = np.random.default_rng(seed).permutation(n)
    return permutation[:n_train], permutation[n_train:]


def prepare_split(dataset: Dataset, output_dir, *, seed, train_fraction=0.5,
                  pca_components=None, K=None, protocol="transductive",
                  method_options=None, replication=0):
    """Write one NEW split directory and return its path.

    output_dir is the split directory itself, e.g. output/split_000. X_train
    intentionally retains PCA.fit_transform coordinates from the original
    training workflow. X_train_eval is PCA.transform(X_train_raw), used for
    full-graph transductive evaluation. Training labels stay outside input/.
    """
    ids = validate_ids(dataset.ids)
    X = validate_features(dataset.X, ids)
    A = validate_adjacency(dataset.A, len(ids))
    y = validate_labels(dataset.y, len(ids))
    train, test = split_indices(len(ids), seed=seed, train_fraction=train_fraction)
    if not isinstance(K, (int, np.integer)) or isinstance(K, bool) or not 2 <= K < len(train):
        raise ValueError("K must be an explicit integer with 2 <= K < n_train; never inferred from labels")
    if np.any((y < 0) | (y >= K)):
        raise ValueError("Evaluation labels must be prespecified class codes 0,...,K-1")
    if protocol not in {"inductive", "transductive"}:
        raise ValueError("protocol must be inductive or transductive")
    if not isinstance(replication, (int, np.integer)) or isinstance(replication, bool) or replication < 0:
        raise ValueError("replication must be a nonnegative integer")
    options = {} if method_options is None else dict(method_options)
    reserved = {"replication", "seed", "K", "p", "n_train", "n_test", "protocol", "shape", "pca", "method_options", "train_fraction"}
    if reserved & options.keys():
        raise ValueError("method_options cannot override split/protocol fields")
    # Validate serializability before creating any output.
    json.dumps(options, allow_nan=False)
    pca = None
    if pca_components is not None:
        if not isinstance(pca_components, (int, np.integer)) or isinstance(pca_components, bool) or not 1 <= pca_components <= min(len(train), X.shape[1]):
            raise ValueError("pca_components must be null or an integer <= min(n_train, feature count)")
        pca = PCA(n_components=int(pca_components), svd_solver="full")
        X_train = pca.fit_transform(X[train])
        X_train_eval = pca.transform(X[train])
        X_test = pca.transform(X[test])
    else:
        X_train, X_train_eval, X_test = X[train].copy(), X[train].copy(), X[test].copy()
    arrays = dict(X_train=X_train, X_train_eval=X_train_eval, X_test=X_test,
                  A_train=A[np.ix_(train, train)], A_test_train=A[np.ix_(test, train)])
    if protocol == "transductive":
        arrays["A_test_test"] = A[np.ix_(test, test)]
    cfg = dict(options, replication=int(replication), seed=int(seed), K=int(K),
               p=int(X_train.shape[1]), n_train=len(train), n_test=len(test),
               protocol=protocol, train_fraction=float(train_fraction),
               pca={"components": int(pca_components) if pca_components is not None else None, "solver": "full" if pca else None,
                    "fit": "training_features_only", "whiten": False},
               method_options=options, shape={key: list(value.shape) for key, value in arrays.items()})
    output_dir = Path(output_dir)
    # exist_ok=False also rejects stale partial runs: no seed replacement/reuse.
    output_dir.mkdir(parents=True, exist_ok=False)
    inp = output_dir / "input"
    inp.mkdir()
    for name, values in arrays.items():
        np.save(inp / (name + ".npy"), values, allow_pickle=False)
        np.savetxt(inp / (name + ".csv"), values, delimiter=",", fmt="%d" if name.startswith("A_") else "%.17g")
    save_archive(output_dir / "evaluation_only.npz", train_indices=train, test_indices=test,
                 y_train=y[train], y_test=y[test], ids_train=ids[train], ids_test=ids[test],
                 chat_id_train=ids[train], chat_id_test=ids[test])
    if pca is not None:
        save_archive(output_dir / "pca_training_only.npz", mean=pca.mean_, components=pca.components_,
                     explained_variance=pca.explained_variance_, explained_variance_ratio=pca.explained_variance_ratio_)
    (inp / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output_dir


def prepare_splits(dataset, output_dir, config):
    """Prepare exactly replications consecutive seeds, with no label selection."""
    seed_base, replications = config.get("seed_base"), config.get("replications")
    for name, value, lower in (("seed_base", seed_base, 0), ("replications", replications, 1)):
        if not isinstance(value, int) or isinstance(value, bool) or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")
    for field in ("train_fraction", "pca_components", "K"):
        if field not in config:
            raise ValueError(f"Configuration must explicitly include {field}")
    paths = [Path(output_dir) / f"split_{rep:03d}" for rep in range(replications)]
    if any(path.exists() for path in paths):
        raise FileExistsError("A target split directory exists; use a new output directory (no cached splits)")
    return [prepare_split(dataset, path, seed=seed_base + rep, replication=rep,
                          train_fraction=config["train_fraction"], pca_components=config["pca_components"],
                          K=config["K"], protocol=config.get("protocol", "transductive"),
                          method_options=config.get("method_options", {})) for rep, path in enumerate(paths)]

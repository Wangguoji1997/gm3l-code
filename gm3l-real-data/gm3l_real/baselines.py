"""Label-free classical adapters, extracted from the local comparison adapters.

Third-party estimators are imported from user-installed packages, not bundled.
Full/tied GMM intentionally live in the native implementation, not sklearn.
"""
from __future__ import annotations

from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral
import random
import warnings

import numpy as np


FINCH_CONFIG = dict(distance="cosine", ensure_early_exit=True, ann_threshold=20_000,
                    faiss_threshold=5_000_000, faiss_use_gpu=False,
                    return_meta_labels=False, verbose=False)
CLASSIX_CONFIG = dict(sorting="pca", metric="euclidean", radius=0.5, minPts=1,
                      group_merging="distance", mergeScale=1.5, post_alloc=True,
                      mergeTinyGroups=True, verbose=0, short_log_form=True)


class MissingDependencyError(ImportError):
    status = "missing_dependency"


@contextmanager
def _isolated_seed(seed):
    """CLASSIX may use process-global RNG; use serial calls/process workers."""
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        np.random.seed(seed)
        random.seed(seed)
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def _features(value, name, *, empty=False):
    value = np.asarray(value)
    if value.ndim != 2 or value.shape[1] == 0 or (not empty and len(value) == 0):
        raise ValueError(f"{name} must be a nonempty-feature, two-dimensional matrix")
    if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite real numeric features")
    return np.array(value, dtype=np.float64, order="C", copy=True)


def _labels(value, n):
    value = np.asarray(value).reshape(-1)
    if value.shape != (n,) or not np.isfinite(value).all() or not np.equal(value, np.floor(value)).all():
        raise ValueError("Estimator returned invalid integer cluster labels")
    return value.astype(np.int64)


def _fixed(defaults, options):
    params = dict(defaults)
    unknown = set(options) - set(params)
    if unknown:
        raise ValueError(f"Unknown options: {sorted(unknown)}")
    changed = [name for name in options if options[name] != params[name]]
    if changed:
        raise ValueError(f"This named preset fixes {changed}")
    return params


def _versions(*names):
    result = {}
    for name in ("numpy", "scipy", "scikit-learn", *names):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not installed"
    return result


def _classix_known_k(module, train, K, seed, params):
    """Original training-only radius grid and tie rule; do not force labels to K."""
    from sklearn.metrics import silhouette_score
    records, assignments = [], []
    for radius in np.geomspace(0.02, 1.0, 25):
        with _isolated_seed(seed):
            model = module.CLASSIX(**{**params, "radius": float(radius)})
            model.fit(train)
        labels = _labels(model.labels_, len(train))
        count = int(np.sum(np.unique(labels) >= 0))
        records.append(dict(radius=float(radius), n_clusters_train=count,
                            absolute_K_error=abs(count - K), training_silhouette=None))
        assignments.append(labels)
    error = min(row["absolute_K_error"] for row in records)
    tied = [i for i, row in enumerate(records) if row["absolute_K_error"] == error]
    cache = {}
    for i in tied:
        if len(tied) == 1 or not 2 <= records[i]["n_clusters_train"] < len(train):
            continue
        key = np.unique(assignments[i], return_inverse=True)[1].tobytes()
        if key not in cache:
            try:
                score = silhouette_score(train, assignments[i], metric="euclidean",
                    sample_size=2000 if len(train) > 2000 else None, random_state=seed)
                cache[key] = float(score) if np.isfinite(score) else None
            except ValueError:
                cache[key] = None
        records[i]["training_silhouette"] = cache[key]
    def tie_key(i):
        score = records[i]["training_silhouette"]
        return (-score if score is not None else float("inf"), records[i]["radius"])
    selected = min(tied, key=tie_key)
    params = {**params, "radius": records[selected]["radius"]}
    with _isolated_seed(seed):
        model = module.CLASSIX(**params)
        model.fit(train)
    if not np.array_equal(_labels(model.labels_, len(train)), assignments[selected]):
        raise RuntimeError("CLASSIX selected-radius refit changed its native partition")
    return model, params, dict(policy="known_K", candidates=records,
        selected_candidate_index=selected, selected_radius=params["radius"],
        objective="min abs(native training cluster count - K)",
        tie_breaker="max training Euclidean silhouette, then smaller radius",
        native_partition_modified=False)


def _classix_scores(model, train_labels, test):
    """Original supplement: nearest native starting point within each cluster."""
    from scipy.spatial.distance import cdist
    if model.metric != "euclidean" or model.group_merging != "distance":
        raise RuntimeError("CLASSIX scoring requires the original Euclidean distance path")
    groups = np.asarray(model.groups_, dtype=np.int64)
    inverse = np.asarray(getattr(model, "inverse_ind", np.argsort(model.ind)), dtype=np.int64)
    mapping = dict(zip(groups[inverse], train_labels))
    starts = np.asarray(model.splist_)[:, 0].astype(np.int64)
    group_ids = groups[starts]
    representative_ids = np.asarray([mapping[int(g)] for g in group_ids], dtype=np.int64)
    if hasattr(model, "label_change"):
        official = np.asarray([model.label_change.get(int(g), -1) for g in group_ids])
        if not np.array_equal(representative_ids, official):
            raise RuntimeError("CLASSIX native starting-point mapping changed")
    representatives = np.asarray(model.sp_data_pts, dtype=np.float64)
    cluster_ids = np.unique(train_labels)
    masks = [representative_ids == cluster for cluster in cluster_ids]
    if any(not mask.any() for mask in masks):
        raise RuntimeError("CLASSIX cluster has no native starting-point representative")
    scores = np.empty((len(test), len(cluster_ids)), dtype=float)
    processed = model.preprocessing(test) if len(test) else test
    for first in range(0, len(test), 1024):
        distances = cdist(processed[first:first + 1024], representatives, "sqeuclidean")
        for column, mask in enumerate(masks):
            scores[first:first + 1024, column] = -distances[:, mask].min(axis=1)
    arrays = dict(representatives=representatives.copy(), representative_cluster_ids=representative_ids,
                  training_mean=np.array(model.mu_, copy=True),
                  training_scale=np.asarray(model.dataScale_),
                  startingpoint_train_indices=np.asarray(model.ind)[starts])
    return scores, cluster_ids, arrays


def fit_baseline(method, X_train, X_test, K, seed, options=None):
    """Fit without y or graph inputs and return retained hard labels plus scores."""
    name = str(method).lower().replace("-", "").replace("_", "").replace(" ", "")
    if name not in {"finch", "classix", "classixknownk"}:
        raise ValueError(f"Unknown classical baseline {method!r}; native GMM is a separate implementation")
    if isinstance(K, bool) or not isinstance(K, Integral) or K < 1:
        raise ValueError("K must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    train, test = _features(X_train, "X_train"), _features(X_test, "X_test", empty=True)
    if train.shape[1] != test.shape[1] or (not name.startswith("classix") and K > len(train)):
        raise ValueError("Feature dimensions differ or K exceeds training rows")
    if options is not None and not isinstance(options, dict):
        raise TypeError("options must be a dictionary or None")
    options = dict(options or {})
    radius_selection = options.pop("radius_selection", None) if name.startswith("classix") else None
    if radius_selection not in {None, "known_K"}:
        raise ValueError("CLASSIX radius_selection supports only 'known_K'")
    known_k = name == "classixknownk" or radius_selection == "known_K"
    diagnostics = dict(method=str(method), seed=int(seed), K_requested=int(K),
        fit_scope="X_train only", heldout_used_for_fit=False, ground_truth_labels_received=False,
        scores_are_probabilities=False, hard_labels_replaced_by_score_argmax=False)
    try:
        from sklearn.metrics import pairwise_distances, pairwise_distances_argmin
        with _isolated_seed(int(seed)), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if name == "finch":
                from finch import FINCH
                if len(train) < 2:
                    raise ValueError("FINCH requires at least two training rows")
                params = _fixed(FINCH_CONFIG, options)
                partitions, counts, requested = FINCH(train, req_clust=int(K), random_state=int(seed), **params)
                if requested is None:
                    raise RuntimeError("FINCH did not return native requested-cluster labels")
                train_labels = _labels(requested, len(train))
                cluster_ids = np.unique(train_labels)
                centers = np.vstack([train[train_labels == c].mean(axis=0) for c in cluster_ids])
                nearest = pairwise_distances_argmin(test, centers, metric="cosine") if len(test) else np.empty(0, dtype=int)
                test_labels = cluster_ids[nearest]
                scores = -pairwise_distances(test, centers, metric="cosine") if len(test) else np.empty((0, len(cluster_ids)))
                arrays = dict(centers=centers, center_cluster_ids=cluster_ids.copy())
                diagnostics.update(config=params, heldout_is_adapter_extension=True,
                    heldout_assignment="nearest frozen training-cluster arithmetic mean (cosine)",
                    score_formula="negative cosine distance to frozen training-cluster mean",
                    hierarchy_cluster_counts=[int(c) for c in counts],
                    hierarchy_partition_count=int(np.asarray(partitions).shape[1]),
                    requested_K_achieved=len(cluster_ids) == K, library_versions=_versions("finch-clust"))
            else:
                import classix
                params = _fixed(CLASSIX_CONFIG, options)
                if known_k:
                    model, params, selection = _classix_known_k(classix, train, K, seed, params)
                    diagnostics["radius_selection"] = selection
                else:
                    model = classix.CLASSIX(**params)
                    model.fit(train)
                train_labels = _labels(model.labels_, len(train))
                test_labels = model.predict(test) if len(test) else np.empty(0, dtype=np.int64)
                scores, cluster_ids, arrays = _classix_scores(model, train_labels, test)
                diagnostics.update(config=params, K_ignored=not known_k,
                    heldout_assignment="official CLASSIX.predict; native variable cluster count retained",
                    score_formula="negative minimum squared distance to native cluster starting points in fitted preprocessing",
                    library_versions=_versions("classixclustering"))
            diagnostics["warnings"] = [f"{w.category.__name__}: {w.message}" for w in caught]
    except ImportError as exc:
        raise MissingDependencyError(f"missing_dependency for {method}: {exc}; install the original package, no fallback is used") from exc
    test_labels = _labels(test_labels, len(test))
    if scores.shape != (len(test), len(cluster_ids)) or not np.isfinite(scores).all():
        raise RuntimeError("Invalid finite test-score matrix")
    if not set(train_labels).issubset(set(cluster_ids)) or not set(test_labels).issubset(set(cluster_ids)):
        raise RuntimeError("Native label is missing from score columns")
    diagnostics.update(status="completed", completed=True, n_clusters_train=int(len(np.unique(train_labels))),
        n_clusters_test=int(len(np.unique(test_labels))), score_type="decision")
    return dict(train_labels=train_labels, test_labels=test_labels, test_scores=scores,
                cluster_ids=cluster_ids, score_type="decision", diagnostics=diagnostics, model_arrays=arrays,
                score_metadata=dict(cluster_ids=cluster_ids.tolist(), score_type="decision",
                    alignment="hard", mapping="hungarian", score_aggregation="max"))

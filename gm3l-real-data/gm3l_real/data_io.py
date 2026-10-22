"""Validated, pickle-free inputs supplied explicitly by the user.

Only requested arrays are read; text, source metadata and other archive members
are never forwarded to a model. Partial pairwise judgements are not negatives.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray
    A: np.ndarray
    ids: np.ndarray
    y: np.ndarray


def validate_ids(values, *, name="ids"):
    values = np.asarray(values)
    if values.ndim != 1 or values.size == 0 or values.dtype.kind not in "iuUS":
        raise ValueError(f"{name} must be a nonempty 1D integer or string array (no pickle)")
    if values.dtype.kind == "S":
        values = np.char.decode(values, "utf-8")
    result = values.astype(str)
    if any(not value.strip() for value in result):
        raise ValueError(f"{name} contains empty IDs")
    if len(set(result.tolist())) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def archive_ids(archive):
    keys = [key for key in ("ids", "chat_id") if key in archive.files]
    if not keys:
        raise ValueError("NPZ must contain unique ids or chat_id")
    result = validate_ids(archive[keys[0]])
    if len(keys) == 2 and not np.array_equal(result, validate_ids(archive[keys[1]])):
        raise ValueError("ids and chat_id disagree")
    return result


def validate_features(X, ids):
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] != len(ids) or X.shape[1] == 0:
        raise ValueError("X must have shape (number of IDs, positive feature count)")
    if X.dtype.kind not in "fiu" or not np.isfinite(X).all():
        raise ValueError("X must contain finite real numbers")
    return X.astype(np.float64, copy=False)


def validate_adjacency(A, n):
    A = np.asarray(A)
    if A.shape != (n, n) or A.dtype.kind not in "bifu":
        raise ValueError("A must be a numeric square matrix aligned to its IDs")
    if not np.isfinite(A).all() or not np.isin(A, [0, 1]).all():
        raise ValueError("A must be binary, with no unknown or missing entries")
    if not np.array_equal(A, A.T):
        raise ValueError("A must be symmetric")
    if np.any(np.diag(A)):
        raise ValueError("A diagonal must be zero; self-links are not modelled")
    return A.astype(np.uint8, copy=False)


def validate_labels(y, n):
    y = np.asarray(y)
    if y.shape != (n,) or y.dtype.kind not in "biuf" or not np.isfinite(y).all() or not np.equal(y, np.floor(y)).all():
        raise ValueError("y must be a 1D finite integer-coded array; encode reference classes as 0,...,K-1 before import")
    return y.astype(np.int64, copy=True)


def _complete_adjacency(archive, n):
    if "complete" in archive.files:
        complete = np.asarray(archive["complete"])
        if complete.shape != () or complete.dtype.kind not in "biu" or complete.item() != 1:
            raise ValueError("Partial adjacency is unsupported: unknown pairs cannot be treated as No")
    if "observed" in archive.files:
        observed = np.asarray(archive["observed"])
        if observed.shape != (n, n) or observed.dtype.kind not in "biu" or not np.isin(observed, [0, 1]).all():
            raise ValueError("Invalid observed mask")
        if not observed[~np.eye(n, dtype=bool)].all():
            raise ValueError("Partial adjacency is unsupported: every off-diagonal pair must be observed")


def alignment_order(source_ids, target_ids):
    source_ids, target_ids = validate_ids(source_ids), validate_ids(target_ids)
    if set(source_ids.tolist()) != set(target_ids.tolist()):
        raise ValueError("ID sets differ; implicit intersection, dropping and positional matching are forbidden")
    positions = {value: i for i, value in enumerate(source_ids)}
    return np.asarray([positions[value] for value in target_ids], dtype=np.int64)


def load_embeddings(path):
    """Read X and IDs only; labels/texts in legacy archives are not returned."""
    with np.load(Path(path), allow_pickle=False) as archive:
        ids = archive_ids(archive)
        X = validate_features(archive["X"], ids).copy()
    return X, ids


def load_adjacency(path):
    with np.load(Path(path), allow_pickle=False) as archive:
        ids = archive_ids(archive)
        _complete_adjacency(archive, len(ids))
        A = validate_adjacency(archive["A"], len(ids)).copy()
    return A, ids


def load_dataset(dataset_path=None, *, embeddings_path=None, adjacency_path=None, evaluation_path=None):
    """Return aligned Dataset from one archive or separate user-supplied NPZs.

    Separate embeddings and adjacency each require their own IDs. Evaluation
    labels may be in embeddings.npz or in evaluation_path (which requires IDs).
    K is never inferred here. There is no automatic dataset discovery.
    """
    if dataset_path is not None:
        if any(p is not None for p in (embeddings_path, adjacency_path, evaluation_path)):
            raise ValueError("Use either dataset_path or separate archives, not both")
        with np.load(Path(dataset_path), allow_pickle=False) as archive:
            ids = archive_ids(archive)
            X = validate_features(archive["X"], ids).copy()
            _complete_adjacency(archive, len(ids))
            A = validate_adjacency(archive["A"], len(ids)).copy()
            y = validate_labels(archive["y"], len(ids))
    else:
        if embeddings_path is None or adjacency_path is None:
            raise ValueError("Provide dataset_path or both embeddings_path and adjacency_path")
        X, ids = load_embeddings(embeddings_path)
        A, adjacency_ids = load_adjacency(adjacency_path)
        order = alignment_order(adjacency_ids, ids)
        A = A[np.ix_(order, order)]
        label_path = embeddings_path if evaluation_path is None else evaluation_path
        with np.load(Path(label_path), allow_pickle=False) as archive:
            label_ids = archive_ids(archive)
            y = validate_labels(archive["y"], len(label_ids))
            y = y[alignment_order(label_ids, ids)]
    return Dataset(X=X, A=A, ids=ids, y=y)


def save_archive(path, **arrays):
    """Write a new pickle-free NPZ without silently overwriting an earlier run."""
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("Output filename must end in .npz")
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ValueError("Object arrays are forbidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return path


def adjacency_from_pairs(path, ids, *, left_column="id_i", right_column="id_j", label_column="must_link"):
    """Import exhaustive pair labels (0/1 or Yes/No); do not infer missing pairs.

    IDs rather than array row numbers are required. Every unordered pair must be
    explicitly labelled, in either orientation, exactly once. No closure is added.
    """
    ids = validate_ids(ids)
    positions = {value: i for i, value in enumerate(ids)}
    A = np.zeros((len(ids), len(ids)), dtype=np.uint8)
    seen = set()
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        if not {left_column, right_column, label_column}.issubset(rows.fieldnames or []):
            raise ValueError("Pair CSV lacks required ID/label columns")
        for line, row in enumerate(rows, start=2):
            if row[left_column] not in positions or row[right_column] not in positions:
                raise ValueError(f"Unknown pair ID at CSV line {line}")
            i, j = positions[row[left_column]], positions[row[right_column]]
            if i == j:
                raise ValueError(f"Self pair at CSV line {line}")
            pair = (min(i, j), max(i, j))
            if pair in seen:
                raise ValueError(f"Duplicate unordered pair at CSV line {line}")
            value = {"0": 0, "1": 1, "Yes": 1, "No": 0}.get(row[label_column])
            if value is None:
                raise ValueError(f"Pair labels must be 0/1 or Yes/No at CSV line {line}")
            A[i, j] = A[j, i] = value
            seen.add(pair)
    if len(seen) != len(ids) * (len(ids) - 1) // 2:
        raise ValueError("Pair CSV is incomplete; absent pairs are unknown, not No")
    return A

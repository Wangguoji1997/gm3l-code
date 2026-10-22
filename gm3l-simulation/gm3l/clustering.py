"""Posterior-error evaluation and the fixed X-only Gaussian-mixture baseline.

The natural logarithm of posterior MSE is evaluated in the log domain, with
denominator n*K. Labels are used for evaluation only, never for GMM fitting,
initialization, or selection among starts.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from .strict_smle import log_gaussian_density


K = 3
N_STARTS = 10
INIT_SEED_NAMESPACE = 2026101401
MAX_ITER = 20000
TOL = 1e-6
BLOCK_ROWS = 256


def _validated_labels(labels, n, k):
    labels = np.asarray(labels)
    if labels.shape != (n,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Labels must be an integer vector with length n")
    if np.any((labels < 0) | (labels >= k)):
        raise ValueError("Labels must be in 0, ..., K-1")
    return labels


def log_mse_from_logits(logits, labels):
    """Return log(mean((posterior - one_hot(labels))**2)) stably.

    Columns must already be aligned to the true label encoding. The true-class
    error is the sum of wrong-class probabilities, not ``1 - rounded_p_true``.
    Neither probabilities nor squared errors need to be representable as
    nonzero float64 values, so errors of order exp(-10000) remain finite here.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] < 2 or not len(logits):
        raise ValueError("A nonempty n by K logits array with K >= 2 is required")
    if not np.isfinite(logits).all():
        raise ValueError("Logits must be finite")
    n, k = logits.shape
    labels = _validated_labels(labels, n, k)
    log_p = logits - logsumexp(logits, axis=1)[:, None]
    wrong_mask = np.arange(k)[None, :] != labels[:, None]
    wrong_log_p = log_p[wrong_mask].reshape(n, k - 1)
    log_true_class_error = logsumexp(wrong_log_p, axis=1)
    log_squared_errors = np.concatenate(
        (2 * log_true_class_error, 2 * wrong_log_p.ravel())
    )
    return float(logsumexp(log_squared_errors) - np.log(n * k))


def star_log_mse(X, A, theta, labels, order, backend="numpy"):
    """Evaluate a fitted star posterior without probability underflow.

    ``order`` is the already-selected evaluation permutation. SMLE and FINAL
    must each be evaluated using their own parameters and that same order.
    No fitting or new alignment is performed. Graph diagonal entries are
    excluded, whether the caller stores them as zeros or ones.
    """
    X = np.asarray(X, dtype=np.float64)
    A = np.asarray(A)
    if X.ndim != 2 or not len(X) or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty finite matrix")
    if A.shape != (len(X), len(X)):
        raise ValueError("A must be an n by n adjacency matrix")
    if backend == "numpy":
        product = np.matmul
    elif backend == "accelerate":
        from .accelerate_blas import dgemm
        product = dgemm
    else:
        raise ValueError("backend must be 'numpy' or 'accelerate'")
    alpha, mu, sigma, pi = (
        np.asarray(theta[key], dtype=np.float64)
        for key in ("alpha", "mu", "sigma", "pi")
    )
    k = len(alpha)
    order = np.asarray(order)
    if (order.shape != (k,) or not np.issubdtype(order.dtype, np.integer)
            or not np.array_equal(np.sort(order), np.arange(k))):
        raise ValueError("order must be a permutation of 0, ..., K-1")
    labels = _validated_labels(labels, len(X), k)
    if not np.all((alpha > 0) & (alpha < 1)):
        raise ValueError("Mixing weights lie outside the parameter space")
    if not np.all((pi > 0) & (pi < 1)):
        raise ValueError("Edge probabilities lie outside the parameter space")
    with threadpool_limits(limits=1):
        log_node = np.log(alpha)[None, :] + log_gaussian_density(X, mu, sigma)
        log_r = log_node - logsumexp(log_node, axis=1)[:, None]
        r = np.exp(log_r)
        log_h0 = np.log(r @ (1 - pi).T)
        log_h1 = np.log(r @ pi.T)
        difference = log_h1 - log_h0
        logits = log_r + log_h0.sum(axis=0)[None, :] - log_h0
        # Keep the original graph compact: convert only a row block for GEMM.
        for start in range(0, len(X), BLOCK_ROWS):
            stop = min(start + BLOCK_ROWS, len(X))
            edge_terms = product(
                np.asarray(A[start:stop], dtype=np.float64, order="C"), difference
            )
            edge_terms -= np.diag(A)[start:stop, None] * difference[start:stop]
            logits[start:stop] += edge_terms
    return log_mse_from_logits(logits[:, order], labels)


def _gmm_log_posterior(X, model):
    weighted = np.empty((len(X), K), dtype=np.float64)
    for k in range(K):
        chol = np.linalg.cholesky(model.covariances_[k])
        standardized = solve_triangular(
            chol, (X - model.means_[k]).T, lower=True, check_finite=False
        )
        weighted[:, k] = np.log(model.weights_[k]) - 0.5 * (
            X.shape[1] * np.log(2 * np.pi)
            + 2 * np.log(np.diag(chol)).sum()
            + np.square(standardized).sum(axis=0)
        )
    log_density = logsumexp(weighted, axis=1)
    return weighted - log_density[:, None], float(log_density.mean())


def fit_gmm(X, labels, rep, n):
    """Fit the fixed ten-start full-covariance GMM, then align and evaluate.

    Each start uses SeedSequence([2026101401, rep, n, start]). The selected
    model maximizes finite final model.score(X) among converged starts; ties
    retain the earlier start. Failed starts and an all-failed fit are explicit.
    All labels are accessed only after the model has been selected.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    if (isinstance(rep, (bool, np.bool_)) or not isinstance(rep, (int, np.integer))
            or rep < 1):
        raise ValueError("rep must be a positive integer")
    if (isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
            or n < K):
        raise ValueError("n must be an integer at least K")
    if X.ndim != 2 or len(X) != n or not np.isfinite(X).all():
        raise ValueError("X must have n rows and finite values")
    starts, best_model, best_start = [], None, None
    best_score = -np.inf
    with threadpool_limits(limits=1):
        for start in range(N_STARTS):
            seed = int(np.random.SeedSequence(
                [INIT_SEED_NAMESPACE, int(rep), int(n), start]
            ).generate_state(1)[0])
            attempt = {"start": start, "seed": seed}
            try:
                model = GaussianMixture(
                    n_components=K, covariance_type="full", tol=TOL,
                    reg_covar=0.0, max_iter=MAX_ITER, n_init=1,
                    init_params="kmeans", random_state=seed,
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(X)
                score = float(model.score(X))
                min_eigenvalue = float(np.linalg.eigvalsh(model.covariances_).min())
                if not np.isfinite(score) or min_eigenvalue <= 0:
                    raise ValueError("Non-finite likelihood or non-positive covariance")
                attempt.update(
                    status="ok", converged=bool(model.converged_),
                    n_iter=int(model.n_iter_), observed_mean_log_likelihood=score,
                    observed_log_likelihood=score * n,
                    lower_bound=float(model.lower_bound_),
                    min_covariance_eigenvalue=min_eigenvalue,
                    min_weight=float(model.weights_.min()),
                    warnings=[str(w.message) for w in caught],
                )
                if model.converged_ and score > best_score:
                    best_model, best_score, best_start = model, score, start
            except (ValueError, np.linalg.LinAlgError) as exc:
                attempt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            starts.append(attempt)

        diagnostics = {"attempts": starts, "selected_start": best_start}
        if best_model is None:
            return dict(status="failed", log_mse_zeta=None, iterations=0,
                        failure=f"No converged finite GMM fit among {N_STARTS} starts",
                        diagnostics=diagnostics)

        labels = _validated_labels(labels, n, K)
        log_posterior, check_score = _gmm_log_posterior(X, best_model)
        if not np.allclose(check_score, best_score, rtol=1e-11, atol=1e-11):
            raise AssertionError("GMM posterior likelihood mismatch")
        predicted = np.argmax(log_posterior, axis=1)
        counts = np.zeros((K, K), dtype=np.int64)
        np.add.at(counts, (labels, predicted), 1)
        rows, columns = linear_sum_assignment(-counts)
        permutation = np.empty(K, dtype=np.int64)
        permutation[rows] = columns
        aligned = log_posterior[:, permutation]
        log_mse = log_mse_from_logits(aligned, labels)
        direct_error = np.exp(aligned)
        direct_error[np.arange(n), labels] -= 1
        direct_mse = float(np.square(direct_error).mean())
        diagnostics.update(
            evaluation_permutation=permutation.tolist(),
            observed_mean_log_likelihood=best_score,
            observed_log_likelihood=best_score * n,
            posterior_likelihood_check_abs=abs(check_score - best_score),
            seed=starts[best_start]["seed"],
            min_covariance_eigenvalue=starts[best_start]["min_covariance_eigenvalue"],
        )
        return dict(status="ok", log_mse_zeta=log_mse,
                    iterations=int(best_model.n_iter_), mse_zeta=direct_mse,
                    clustering_accuracy=float(np.mean(np.argmax(aligned, axis=1) == labels)),
                    diagnostics=diagnostics)

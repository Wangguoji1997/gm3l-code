"""Closed-form complete-data estimates using observed simulated labels.

These estimates are an oracle comparison, never an SMLE initializer.  Each
undirected edge is counted once and A's diagonal is ignored.  The class MLE
covariance uses n_k, not n_k-1.  Singular covariances remain valid closed-form
moment estimates for squared-error comparisons; diagnostics explicitly state
that they are not feasible Gaussian parameters in the positive-definite model.
No ridge, clipping, covariance pooling, or regularization is performed.
"""
from __future__ import annotations

import numpy as np


def complete_data_estimate(Y, X, A, K):
    """Return ``(theta, diagnostics)`` for known labels 0,...,K-1.

    ``theta`` has alpha (K,), mu (K,p), sigma (K,p,p), and pi (K,K).
    Singular positive-semidefinite sigma is returned, not marked undefined.
    Undefined estimators (empty classes or a class with fewer than two nodes,
    giving a zero within-class edge denominator) raise an explicit ValueError.
    Invalid dimensions, nonfinite X, or nonbinary/asymmetric off-diagonal
    adjacency also raise ValueError.  Values on A's diagonal are never used.

    ``pd_covariance`` and ``global_likelihood_feasible`` are conservative
    numerical feasibility diagnostics: the former requires n_k>p, positive
    eigenvalues above roundoff, and successful Cholesky for every component;
    the latter additionally requires strictly interior alpha and pi.  This
    does not claim that a latent-label Gaussian-mixture likelihood has a
    finite global maximum.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y)
    A = np.asarray(A)
    if not isinstance(K, (int, np.integer)) or K < 1:
        raise ValueError('K must be a positive integer')
    K = int(K)
    if X.ndim != 2 or min(X.shape) < 1:
        raise ValueError('X must have shape (n,p), with n and p positive')
    n, p = X.shape
    if Y.shape != (n,) or A.shape != (n, n):
        raise ValueError('Expected Y.shape=(n,) and A.shape=(n,n)')
    if not np.isfinite(X).all():
        raise ValueError('X contains a nonfinite value')
    if not np.issubdtype(Y.dtype, np.integer):
        if not np.issubdtype(Y.dtype, np.number) or not np.isfinite(Y).all() or not np.equal(Y, np.floor(Y)).all():
            raise ValueError('Y must contain integer class labels')
        Y = Y.astype(np.int64)
    if np.any((Y < 0) | (Y >= K)):
        raise ValueError('Y labels must be in 0,...,K-1')

    groups = [np.flatnonzero(Y == k) for k in range(K)]
    counts = np.asarray([len(group) for group in groups], dtype=np.int64)
    empty = np.flatnonzero(counts == 0)
    if len(empty):
        raise ValueError(f'Undefined class mean/covariance and edge estimators: empty classes (0-based) {empty.tolist()}; counts={counts.tolist()}')
    singleton = np.flatnonzero(counts < 2)
    if len(singleton):
        raise ValueError(f'Undefined within-class Pi: zero edge denominator in classes (0-based) {singleton.tolist()}; counts={counts.tolist()}')

    alpha = counts / float(n)
    mu = np.asarray([X[group].mean(axis=0) for group in groups])
    sigma = np.empty((K, p, p), dtype=float)
    for k, group in enumerate(groups):
        residual = X[group] - mu[k]
        covariance = residual.T @ residual / counts[k]
        sigma[k] = (covariance + covariance.T) * 0.5

    pi = np.empty((K, K), dtype=float)
    edge_denominators = np.empty((K, K), dtype=np.int64)
    edge_successes = np.empty((K, K), dtype=float)
    directed_difference = 0.0
    for k, first in enumerate(groups):
        for ell in range(k, K):
            second = groups[ell]
            block = A[np.ix_(first, second)]
            if k == ell:
                upper = np.triu_indices(counts[k], k=1)
                edges = block[upper]
                reverse = block.T[upper]
                denominator = int(counts[k] * (counts[k] - 1) // 2)
            else:
                edges = block.ravel()
                reverse = A[np.ix_(second, first)].T.ravel()
                denominator = int(counts[k] * counts[ell])
            if not np.array_equal(edges, reverse):
                raise ValueError(f'Adjacency is not symmetric off its diagonal in block ({k},{ell})')
            if not np.all((edges == 0) | (edges == 1)):
                raise ValueError(f'Nonbinary/nonfinite off-diagonal adjacency in block ({k},{ell})')
            if denominator <= 0:
                raise ValueError(f'Undefined Pi: zero edge denominator in block ({k},{ell})')
            success = float(np.sum(edges, dtype=np.float64))
            value = success / denominator
            # Check the ordered-pair version directly using both orientations.
            ordered_success = success + float(np.sum(reverse, dtype=np.float64))
            ordered_value = ordered_success / (2 * denominator)
            directed_difference = max(directed_difference, abs(value - ordered_value))
            pi[k, ell] = pi[ell, k] = value
            edge_denominators[k, ell] = edge_denominators[ell, k] = denominator
            edge_successes[k, ell] = edge_successes[ell, k] = success

    eigenvalues = np.linalg.eigvalsh(sigma)
    eigen_min = eigenvalues[:, 0]
    tolerance = p * np.finfo(float).eps * np.maximum(1.0, np.max(np.abs(eigenvalues), axis=1))
    ranks = [int(np.linalg.matrix_rank(covariance)) for covariance in sigma]
    cholesky_ok = []
    for covariance in sigma:
        try:
            np.linalg.cholesky(covariance)
            cholesky_ok.append(True)
        except np.linalg.LinAlgError:
            cholesky_ok.append(False)
    pd_each = (counts > p) & (eigen_min > tolerance) & np.asarray(cholesky_ok)
    psd_each = eigen_min >= -tolerance
    interior_alpha = bool(np.all(alpha > 0) and (K == 1 or np.all(alpha < 1)))
    interior_pi = bool(np.all((pi > 0) & (pi < 1)))
    classes_le_p = np.flatnonzero(counts <= p).tolist()
    boundary_reasons = []
    if not np.all(pd_each):
        boundary_reasons.append('At least one closed-form covariance is singular or numerically not positive definite')
    if not interior_alpha:
        boundary_reasons.append('Mixing proportions are outside the open simplex')
    if not interior_pi:
        boundary_reasons.append('At least one edge estimate equals 0 or 1, outside the open Bernoulli parameter space')
    diagnostics = dict(
        n=int(n), p=int(p), K=K, counts=counts.tolist(),
        closed_form_estimates_defined=True,
        pd_covariance=bool(np.all(pd_each)), psd_covariance=bool(np.all(psd_each)),
        min_eigenvalue=float(eigen_min.min()),
        min_covariance_eigenvalue=float(eigen_min.min()),
        covariance_min_eigenvalues=eigen_min.tolist(), covariance_ranks=ranks,
        covariance_pd_by_class=pd_each.tolist(), covariance_cholesky_by_class=cholesky_ok,
        covariance_numerical_tolerances=tolerance.tolist(),
        classes_n_le_p=classes_le_p, classes_n_le_p_count=len(classes_le_p),
        class_index_convention='zero-based',
        alpha_interior=interior_alpha, pi_interior=interior_pi,
        global_likelihood_feasible=bool(np.all(pd_each) and interior_alpha and interior_pi),
        global_likelihood_feasibility_reasons=boundary_reasons,
        directed_vs_undirected_max_abs_diff=float(directed_difference),
        independent_edge_denominators=edge_denominators.tolist(),
        independent_edge_successes=edge_successes.tolist(),
        min_alpha=float(alpha.min()), min_pi=float(pi.min()), max_pi=float(pi.max()),
    )
    return dict(alpha=alpha, mu=mu, sigma=sigma, pi=pi), diagnostics


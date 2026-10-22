"""Gaussian, star-composite SMLE, and closed-form FINAL estimation.

Adapted from the original strict_smle.py, stable_em.py and real-data driver.
No covariance ridge, probability clipping, true-label initialization or
pseudocount is introduced. Numerical and boundary failures remain explicit.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from scipy.linalg import cholesky, solve_triangular, eigh
from scipy.special import expit, logsumexp
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class FitConfig:
    max_iter: int = 500
    tolerance: float = 1e-7
    kmeans_n_init: int = 10
    pi_init_diagonal: float = 0.70
    pi_init_off_diagonal: float = 0.30

    def __post_init__(self):
        if self.max_iter < 1 or self.kmeans_n_init < 1 or self.tolerance <= 0:
            raise ValueError("Positive iteration counts and tolerance are required")
        if not (0 < self.pi_init_diagonal < 1 and 0 < self.pi_init_off_diagonal < 1):
            raise ValueError("Initial edge probabilities must be interior")


def setup_adjacency(A):
    B = np.asarray(A, dtype=float).copy()
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError("A must be square")
    if not np.isfinite(B).all() or not np.isin(B, [0, 1]).all() or not np.array_equal(B, B.T):
        raise ValueError("A must be a complete symmetric binary matrix")
    if np.any(np.diag(B) != 0):
        raise ValueError("The input adjacency diagonal must be zero (no self-links)")
    np.fill_diagonal(B, 0)
    D = 1 - B
    np.fill_diagonal(D, 0)
    return B, D


def initialize_theta(X, A, K, seed, config=FitConfig(), method="spectral"):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not np.isfinite(X).all() or not 2 <= K < len(X):
        raise ValueError("Finite 2D features and 2 <= K < n_train are required")
    n = len(X)
    if method == "spectral":
        B, _ = setup_adjacency(A)
        _, eigenvectors = eigh(B, subset_by_index=[n-K, n-1], check_finite=False)
        norms = np.linalg.norm(eigenvectors, axis=1)
        if np.any(norms == 0):
            raise FloatingPointError("Zero spectral embedding row")
        features = eigenvectors / norms[:, None]
    elif method == "x_kmeans":
        features = X
    else:
        raise ValueError("Unknown initializer: " + method)
    labels = KMeans(n_clusters=K, n_init=config.kmeans_n_init, max_iter=300,
                    random_state=int(seed % (2**31-1))).fit_predict(features)
    counts = np.bincount(labels, minlength=K)
    if np.any(counts == 0):
        raise FloatingPointError("Empty initializer class")
    residual = X - X.mean(axis=0)
    covariance = residual.T @ residual / n
    pi = np.full((K, K), config.pi_init_off_diagonal)
    np.fill_diagonal(pi, config.pi_init_diagonal)
    return dict(alpha=counts/n, mu=np.asarray([X[labels == k].mean(axis=0) for k in range(K)]),
                sigma=np.repeat(covariance[None, :, :], K, axis=0), pi=pi)


def parameter_vector(theta, *, edges=True):
    K, p = theta["mu"].shape
    rows, cols = np.triu_indices(p)
    blocks = [theta["alpha"][:-1], theta["mu"].ravel(), theta["sigma"][:, rows, cols].ravel()]
    if edges:
        blocks.append(theta["pi"][np.triu_indices(K)])
    return np.concatenate(blocks)


def log_gaussian_density(X, mu, sigma):
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    result = np.empty((n, len(mu)))
    for k in range(len(mu)):
        factor = cholesky(sigma[k], lower=True, check_finite=False)
        transformed = solve_triangular(factor, (X-mu[k]).T, lower=True, check_finite=False)
        result[:, k] = -0.5 * (p*np.log(2*np.pi) + 2*np.log(np.diag(factor)).sum()
                              + np.sum(transformed**2, axis=0))
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite Gaussian log density")
    return result


def gaussian_log_posterior(X, theta):
    if not np.all((theta["alpha"] > 0) & (theta["alpha"] < 1)):
        raise FloatingPointError("Mixing weight reached the parameter-space boundary")
    node = np.log(theta["alpha"])[None, :] + log_gaussian_density(X, theta["mu"], theta["sigma"])
    normalizer = logsumexp(node, axis=1)
    return node-normalizer[:, None], float(normalizer.sum())


def log_edge_probabilities(theta):
    if "pi_logit" in theta:
        t = theta["pi_logit"]
        if not np.isfinite(t).all():
            raise FloatingPointError("Nonfinite edge logit")
        return -np.logaddexp(0, -t), -np.logaddexp(0, t)
    pi = theta["pi"]
    if not np.isfinite(pi).all() or np.any((pi < 0) | (pi > 1)):
        raise FloatingPointError("Invalid edge probability")
    with np.errstate(divide="ignore"):
        return np.log(pi), np.log1p(-pi)


def log_aggregate(M, Z, logZ):
    counts = M.T @ Z
    with np.errstate(divide="ignore"):
        result = np.log(counts)
    # Recover posterior tails lost on exponentiation; absent neighborhoods stay -inf.
    for j in np.flatnonzero(np.any(counts == 0, axis=1)):
        columns = np.flatnonzero(counts[j] == 0)
        neighbors = np.flatnonzero(M[:, j])
        if len(neighbors):
            result[j, columns] = logsumexp(logZ[np.ix_(neighbors, columns)], axis=0)
    return result


def estep_log(X, B, D, theta):
    lp1, lp0 = log_edge_probabilities(theta)
    logR, log_feature_likelihood = gaussian_log_posterior(X, theta)
    H1 = logsumexp(logR[:, None, :] + lp1[None, :, :], axis=2)
    H0 = logsumexp(logR[:, None, :] + lp0[None, :, :], axis=2)
    if not np.isfinite(H1).all() or not np.isfinite(H0).all():
        raise FloatingPointError("Undefined boundary star likelihood")
    logits = logR + B @ H1 + D @ H0
    norm = logsumexp(logits, axis=1)
    logZ = logits - norm[:, None]
    Z = np.exp(logZ)
    logU1 = log_aggregate(B, Z, logZ) - H1
    logU0 = log_aggregate(D, Z, logZ) - H0
    a = logsumexp(logU1[:, :, None] + lp1[None, :, :], axis=1)
    b = logsumexp(logU0[:, :, None] + lp0[None, :, :], axis=1)
    weights = Z + np.exp(logR + np.logaddexp(a, b))
    if not np.isclose(weights.sum(), len(X)**2, rtol=1e-10, atol=1e-7):
        raise FloatingPointError("E-step weights do not sum to n^2")
    J1 = lp1 + logsumexp(logU1[:, :, None] + logR[:, None, :], axis=0)
    J0 = lp0 + logsumexp(logU0[:, :, None] + logR[:, None, :], axis=0)
    # Symmetrization multiplies diagonal success and failure equally.
    next_t = np.logaddexp(J1, J1.T) - np.logaddexp(J0, J0.T)
    return dict(zeta=Z, log_zeta=logZ, weights=weights, next_pi_logit=next_t,
                objective=float(len(X)*log_feature_likelihood + norm.sum()))


def gaussian_update(X, weights, *, composite=False, tied=False):
    counts = weights.sum(axis=0)
    if np.any(counts <= 0) or not np.isfinite(counts).all():
        raise FloatingPointError("Empty M-step class")
    alpha = counts / (len(X)**2 if composite else len(X))
    if composite:
        alpha[-1] = 1-alpha[:-1].sum()
    mu = (weights.T @ X) / counts[:, None]
    K, p = mu.shape
    sigma = np.empty((K, p, p))
    if tied:
        pooled = np.zeros((p, p))
        for k in range(K):
            residual = X-mu[k]
            pooled += (residual.T*weights[:, k]) @ residual
        pooled /= len(X)
        sigma[:] = (pooled+pooled.T)*0.5
    else:
        for k in range(K):
            residual = X-mu[k]
            covariance = (residual.T*weights[:, k]) @ residual / counts[k]
            sigma[k] = (covariance+covariance.T)*0.5
    return dict(alpha=alpha, mu=mu, sigma=sigma)


def fit_smle(X, A, initial, config=FitConfig()):
    theta = {k: np.array(initial[k], dtype=float, copy=True) for k in ("alpha", "mu", "sigma", "pi")}
    theta["pi_logit"] = np.array(initial.get("pi_logit", np.log(theta["pi"])-np.log1p(-theta["pi"])), copy=True)
    B, D = setup_adjacency(A)
    history, residual, failure, converged, iteration = [], float("inf"), None, False, 0
    started = time.perf_counter()
    try:
        post = estep_log(X, B, D, theta)
        history.append(post["objective"])
        for iteration in range(1, config.max_iter+1):
            updated = gaussian_update(X, post["weights"], composite=True)
            updated["pi_logit"] = post["next_pi_logit"]
            updated["pi"] = expit(updated["pi_logit"])
            residual = float(np.linalg.norm(parameter_vector(updated)-parameter_vector(theta)))
            theta = updated
            post = estep_log(X, B, D, theta)
            history.append(post["objective"])
            if residual < config.tolerance:
                converged = True
                break
    except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
        failure = f"{type(exc).__name__}: {exc}"
    return dict(**theta, converged=converged, failure=failure, residual=residual, iterations=iteration,
                objective=history[-1] if history else float("nan"), objective_history=history,
                monotonicity_violation=max(0.0, -float(np.min(np.diff(history)))) if len(history)>1 else 0.0,
                elapsed_seconds=time.perf_counter()-started)


def fit_gmm(X, K, seed, config=FitConfig(), *, tied=False):
    """Original X-only EM; full covariance unless explicitly requested tied."""
    initial = initialize_theta(X, np.zeros((len(X), len(X))), K, seed, config, "x_kmeans")
    theta = {k: initial[k] for k in ("alpha", "mu", "sigma")}
    history, residual, failure, converged, iteration = [], float("inf"), None, False, 0
    started = time.perf_counter()
    try:
        logp, objective = gaussian_log_posterior(X, theta)
        history.append(objective)
        for iteration in range(1, config.max_iter+1):
            updated = gaussian_update(X, np.exp(logp), tied=tied)
            residual = float(np.linalg.norm(parameter_vector(updated, edges=False)-parameter_vector(theta, edges=False)))
            theta = updated
            logp, objective = gaussian_log_posterior(X, theta)
            history.append(objective)
            if residual < config.tolerance:
                converged = True
                break
    except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
        failure = f"{type(exc).__name__}: {exc}"
    return dict(**theta, converged=converged, failure=failure, residual=residual, iterations=iteration,
                objective=history[-1] if history else float("nan"), objective_history=history,
                elapsed_seconds=time.perf_counter()-started)


def final_from_zeta(X, A, zeta):
    """One closed-form FINAL update from SMLE training posteriors."""
    theta = gaussian_update(X, zeta)
    B, D = setup_adjacency(A)
    success = zeta.T @ (B @ zeta)
    failure = zeta.T @ (D @ zeta)
    success = (success+success.T)*0.5
    failure = (failure+failure.T)*0.5
    denominator = success+failure
    if np.any(denominator <= 0):
        raise FloatingPointError("Undefined FINAL edge denominator")
    theta["pi"] = success/denominator
    if not np.isfinite(theta["pi"]).all():
        raise FloatingPointError("Undefined FINAL edge probability")
    return theta


def star_log_posterior(Xquery, Xreference, Aquery, theta, *, same_nodes=False):
    """Integrate neighbor classes with parameters frozen at the training fit."""
    lp1, lp0 = log_edge_probabilities(theta)
    logR, _ = gaussian_log_posterior(Xreference, theta)
    H1 = logsumexp(logR[:, None, :] + lp1[None, :, :], axis=2)
    H0 = logsumexp(logR[:, None, :] + lp0[None, :, :], axis=2)
    if not np.isfinite(H1).all() or not np.isfinite(H0).all():
        raise FloatingPointError("Undefined boundary star likelihood")
    B = np.asarray(Aquery, dtype=float).copy()
    if B.shape != (len(Xquery), len(Xreference)) or not np.isin(B, [0, 1]).all():
        raise ValueError("Prediction adjacency has invalid shape or non-binary entries")
    D = 1-B
    if same_nodes:
        if B.shape[0] != B.shape[1]:
            raise ValueError("same_nodes requires square adjacency")
        np.fill_diagonal(B, 0)
        np.fill_diagonal(D, 0)
    query, _ = gaussian_log_posterior(Xquery, theta)
    logits = query+B@H1+D@H0
    logp = logits-logsumexp(logits, axis=1, keepdims=True)
    if not np.isfinite(logp).all():
        raise FloatingPointError("Nonfinite posterior score")
    return logp


def split_posteriors(inputs, theta, protocol):
    X, Xt, A = inputs["X_train"], inputs["X_test"], inputs["A_train"]
    train = star_log_posterior(X, X, A, theta, same_nodes=True)
    if protocol == "inductive":
        test = star_log_posterior(Xt, X, inputs["A_test_train"], theta)
    elif protocol == "transductive":
        # Keep training fit_transform and evaluation transform coordinates distinct.
        Xall = np.concatenate((inputs["X_train_eval"], Xt), axis=0)
        across = inputs["A_test_train"]
        Aall = np.block([[A, across.T], [across, inputs["A_test_test"]]])
        test = star_log_posterior(Xall, Xall, Aall, theta, same_nodes=True)[len(X):]
    else:
        raise ValueError("Unknown prediction protocol")
    return train, test

"""General-K soft-posterior FINAL estimator, with all self-pairs excluded."""
import numpy as np
from scipy.special import xlogy


def final_from_zeta(X, adjacency, zeta, product=None):
    X = np.asarray(X, dtype=float)
    zeta = np.asarray(zeta, dtype=float)
    n, p = X.shape
    if zeta.ndim != 2 or len(zeta) != n or not np.isfinite(zeta).all() or np.any(zeta < 0):
        raise ValueError('Invalid soft posterior matrix')
    K = zeta.shape[1]
    row_error = float(np.max(np.abs(zeta.sum(axis=1) - 1)))
    if row_error > 1e-9:
        raise FloatingPointError('Posterior rows do not sum to one')
    counts = zeta.sum(axis=0)
    if np.any(counts <= 0):
        raise FloatingPointError('Empty soft class')
    mu = (zeta.T @ X) / counts[:, None]
    sigma = np.empty((K, p, p))
    for k in range(K):
        residual = X - mu[k]
        covariance = (residual.T * zeta[:, k]) @ residual / counts[k]
        sigma[k] = (covariance + covariance.T) / 2
    if product is None:
        B = np.asarray(adjacency, dtype=float).copy()
        np.fill_diagonal(B, 0.)
        neighbor_sum = B @ zeta
    else:
        # The caller supplies an audited product with diagonal-free adjacency.
        neighbor_sum = product(adjacency, zeta)
    numerator = zeta.T @ neighbor_sum
    denominator = np.outer(counts, counts) - zeta.T @ zeta
    if np.any(denominator <= 0):
        raise FloatingPointError('Nonpositive FINAL pair denominator')
    numerator = (numerator + numerator.T) / 2
    denominator = (denominator + denominator.T) / 2
    theta = dict(alpha=counts/n, mu=mu, sigma=sigma, pi=numerator/denominator)
    diagnostics = dict(posterior_row_sum_max_error=row_error,
        soft_counts=counts.tolist(), mean_posterior_entropy=float(-xlogy(zeta,zeta).sum()/n),
        mean_max_posterior=float(zeta.max(axis=1).mean()),
        mean_posterior_softness=float(np.mean(1-np.sum(zeta*zeta,axis=1))))
    return theta, diagnostics

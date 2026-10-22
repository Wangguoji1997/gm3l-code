"""Fixed K=3, p=5 model and exact memory-bounded simulation data generation."""

import numpy as np

DATA_SEED = 2026101101
K = 3
P = 5
N_GRID = (3200, 6400, 12800, 25600)


def make_parameters():
    """Return fresh arrays for the fixed model used by the manuscript study."""
    mu = np.array([[-.3, -1.1, -.7, -.1, .1],
                   [.1, -.5, .8, .2, .6],
                   [.2, 1.6, -.1, -.1, -.7]])
    distance = np.abs(np.arange(P)[:, None] - np.arange(P)[None, :])
    rho = np.array([.3, .5, .7])
    pi = np.full((K, K), .15)
    np.fill_diagonal(pi, .70)
    return dict(alpha=np.array([11/30, 1/3, 3/10]), mu=mu,
                sigma=rho[:, None, None]**distance[None, :, :], pi=pi)


def make_true_parameters(K=3, p=5, seed=None, eta=None):
    """Compatibility interface; this study's truth is fixed, not seed-generated."""
    if (K, p) != (3, 5):
        raise ValueError("This study requires K=3 and p=5")
    return make_parameters()


def simulate_data(n, parameters, rng):
    """Generate uint8 adjacency in the original upper-triangle RNG order.

    Generate once at the largest requested n, then slice all three arrays for
    paired, nested samples. Generating smaller n separately changes the sample.
    """
    if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n < 1:
        raise ValueError("n must be a positive integer")
    alpha, mu, sigma, pi = (parameters[k] for k in ('alpha', 'mu', 'sigma', 'pi'))
    labels = rng.choice(len(alpha), n, p=alpha)
    X = np.empty((n, mu.shape[1]))
    for k in range(len(alpha)):
        selected = np.flatnonzero(labels == k)
        if len(selected):
            X[selected] = rng.multivariate_normal(mu[k], sigma[k], len(selected))
    A = np.zeros((n, n), dtype=np.uint8)
    np.fill_diagonal(A, 1)
    for i in range(n-1):
        edges = rng.binomial(1, pi[labels[i], labels[i+1:]])
        A[i, i+1:] = edges
        A[i+1:, i] = edges
    return labels, X, A

#!/usr/bin/env python3
"""Exact Algorithm 1 star-composite SMLE using NumPy matrix multiplication.

No ridge, covariance floor, probability clipping, true-label initialization, or
likelihood-based stopping is used. Dense aggregation only rearranges the exact
sums in the manuscript. Boundary/numerical failures are exposed in fit results.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
from dataclasses import dataclass
import time
import numpy as np
from scipy.linalg import cholesky, solve_triangular, eigh
from scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from .data import make_true_parameters, simulate_data


@dataclass(frozen=True)
class FitConfig:
    max_iter: int = 200
    tolerance: float = 1e-5
    kmeans_n_init: int = 10
    pi_init_diagonal: float = 0.70
    pi_init_off_diagonal: float = 0.30

def initial_pi(K: int, config: FitConfig):
    pi = np.full((K, K), config.pi_init_off_diagonal)
    np.fill_diagonal(pi, config.pi_init_diagonal)
    return pi


def initialize_theta(X, A, K, seed, config=FitConfig(), method='spectral'):
    """Observed-data-only initializer; covariance pooled, Pi fixed 0.7/0.3.

    Spectral uses largest K eigenvectors of the symmetric observed adjacency
    (diagonal removed), row-normalized, then KMeans(n_init=10). X-only KMeans
    provides the manuscript's example and an optional sensitivity initializer.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if method == 'spectral':
        B = np.asarray(A, dtype=float).copy()
        np.fill_diagonal(B, 0.0)
        _, eigenvectors = eigh(B, subset_by_index=[n-K, n-1], check_finite=False)
        norms = np.linalg.norm(eigenvectors, axis=1)
        if np.any(norms == 0):
            raise FloatingPointError('Zero spectral embedding row')
        features = eigenvectors / norms[:, None]
    elif method == 'x_kmeans':
        features = X
    else:
        raise ValueError("method must be 'spectral' or 'x_kmeans'")
    labels = KMeans(n_clusters=K, n_init=config.kmeans_n_init, max_iter=300,
                    random_state=int(seed % (2**31-1))).fit_predict(features)
    counts = np.bincount(labels, minlength=K)
    if np.any(counts == 0):
        raise FloatingPointError('Empty initializer class')
    alpha = counts / n
    mu = np.asarray([X[labels == k].mean(axis=0) for k in range(K)])
    residual = X - X.mean(axis=0)
    covariance = residual.T @ residual / n
    sigma = np.repeat(covariance[None, :, :], K, axis=0)
    return dict(alpha=alpha, mu=mu, sigma=sigma, pi=initial_pi(K,config))


def parameter_vector(theta):
    K, p = theta['mu'].shape
    sigma_upper = np.triu_indices(p)
    pi_upper = np.triu_indices(K)
    return np.concatenate((theta['alpha'][:-1], theta['mu'].ravel(),
        theta['sigma'][:,sigma_upper[0],sigma_upper[1]].ravel(),
        theta['pi'][pi_upper]))


def log_gaussian_density(X, mu, sigma):
    n, p = X.shape
    result = np.empty((n, len(mu)))
    for k in range(len(mu)):
        factor = cholesky(sigma[k], lower=True, check_finite=False)
        transformed = solve_triangular(factor, (X-mu[k]).T, lower=True, check_finite=False)
        result[:,k] = -0.5*(p*np.log(2*np.pi) + 2*np.log(np.diag(factor)).sum()
                               + np.sum(transformed**2,axis=0))
    return result


def _off_diagonal(A):
    B = np.asarray(A,dtype=float)
    if np.any(np.diag(B) != 0.0):
        B = B.copy()
        np.fill_diagonal(B,0.0)
    return B


def estep(X, A, theta):
    """Exact posterior sufficient statistics; no n*n*K*K tensors."""
    X = np.asarray(X,dtype=float)
    B = _off_diagonal(A)
    alpha, mu, sigma, pi = (theta[key] for key in ('alpha','mu','sigma','pi'))
    if not np.all((alpha > 0)&(alpha < 1)):
        raise FloatingPointError('Mixing weight reached parameter-space boundary')
    if not np.all((pi > 0)&(pi < 1)):
        raise FloatingPointError('Pi reached parameter-space boundary')
    log_node = np.log(alpha)[None,:] + log_gaussian_density(X,mu,sigma)
    log_f = logsumexp(log_node,axis=1)
    log_R = log_node-log_f[:,None]
    R = np.exp(log_R)
    H1 = R @ pi.T
    H0 = R @ (1-pi).T
    log_H0, log_H1 = np.log(H0), np.log(H1)
    logits = log_R + log_H0.sum(axis=0)[None,:] - log_H0 + B @ (log_H1-log_H0)
    normalizer = logsumexp(logits,axis=1)
    zeta = np.exp(logits-normalizer[:,None])
    C1 = B.T @ zeta
    C0 = zeta.sum(axis=0)[None,:]-zeta-C1
    # Zero is exact for an absent central class; matrix subtraction can cause
    # at most floating-point-sized negatives. This corrects arithmetic only.
    min_c0 = float(C0.min())
    if min_c0 < -1e-10 * len(X):
        raise FloatingPointError('Invalid negative aggregated posterior count')
    C0 = np.maximum(C0,0.0)
    U1, U0 = C1/H1, C0/H0
    weights = zeta + R*(U1@pi+U0@(1-pi))
    J1 = (U1.T@R)*pi
    J0 = (U0.T@R)*(1-pi)
    numerator = J1+J1.T
    denominator = J1+J0+(J1+J0).T
    # For k=k, xi_[kk] contains only one term, not two. The quotient agrees,
    # but keep the exact manuscript statistics for auditability.
    d = np.diag_indices(len(alpha))
    numerator[d] *= 0.5
    denominator[d] *= 0.5
    if not np.isclose(weights.sum(),len(X)**2,rtol=1e-10,atol=1e-7):
        raise FloatingPointError('E-step weights do not sum to n^2')
    return dict(zeta=zeta, weights=weights,pi_numerator=numerator,
        pi_denominator=denominator, objective=float(len(X)*log_f.sum()+normalizer.sum()))


def mstep(X, posterior):
    weights = posterior['weights']
    counts = weights.sum(axis=0)
    if np.any(counts <= 0):
        raise FloatingPointError('Zero M-step class weight')
    alpha = counts/(len(X)**2)
    # Algorithm 1 defines alpha_K as the sum-to-one complement.
    alpha[-1] = 1.0-alpha[:-1].sum()
    mu = (weights.T@X)/counts[:,None]
    K, p = mu.shape
    sigma = np.empty((K,p,p))
    for k in range(K):
        difference = X-mu[k]
        covariance = (difference.T*weights[:,k])@difference/counts[k]
        sigma[k] = (covariance+covariance.T)*0.5
    if np.any(posterior['pi_denominator'] <= 0):
        raise FloatingPointError('Zero M-step Pi denominator')
    pi = posterior['pi_numerator']/posterior['pi_denominator']
    return dict(alpha=alpha,mu=mu,sigma=sigma,pi=pi)


def parameter_diagnostics(theta):
    eigvals = np.linalg.eigvalsh(theta['sigma'])
    pi = theta['pi']
    return dict(min_alpha=float(theta['alpha'].min()),min_covariance_eigenvalue=float(eigvals.min()),
        min_pi=float(pi.min()),max_pi=float(pi.max()),
        near_covariance_boundary=bool(eigvals.min()<1e-8),
        near_probability_boundary=bool(min(theta['alpha'].min(),pi.min(),1-pi.max())<1e-8))


def fit_smle(X, A, initial, config=FitConfig()):
    """One start of Algorithm 1. All failures/convergence diagnostics explicit."""
    start=time.perf_counter()
    theta={key:np.array(initial[key],copy=True,dtype=float) for key in ('alpha','mu','sigma','pi')}
    B=_off_diagonal(A)
    history=[]
    residual=float('inf')
    converged=False
    failure=None
    iteration=0
    try:
        posterior=estep(X,B,theta)
        history.append(posterior['objective'])
        for iteration in range(1,config.max_iter+1):
            updated=mstep(X,posterior)
            residual=float(np.linalg.norm(parameter_vector(updated)-parameter_vector(theta)))
            theta=updated
            posterior=estep(X,B,theta)
            history.append(posterior['objective'])
            if residual<config.tolerance:
                converged=True
                break
    except (np.linalg.LinAlgError,FloatingPointError,ValueError) as error:
        failure=f'{type(error).__name__}: {error}'
    try:
        diagnostics=parameter_diagnostics(theta)
    except (np.linalg.LinAlgError,ValueError):
        diagnostics={}
    increments=np.diff(history)
    violation=max(0.,float(-increments.min())) if len(increments) else 0.
    return dict(**theta, objective=history[-1] if history else float('nan'),
        iterations=iteration,converged=converged,residual=residual,failure=failure,
        monotonicity_violation=violation,objective_history=history,
        elapsed_seconds=time.perf_counter()-start,**diagnostics)


def align_estimate(parameters,estimate):
    cost=((parameters['mu'][:,None,:]-estimate['mu'][None,:,:])**2).sum(axis=2)
    true_index,estimated_index=linear_sum_assignment(cost)
    order=np.empty(len(true_index),dtype=int)
    order[true_index]=estimated_index
    return dict(alpha=estimate['alpha'][order],mu=estimate['mu'][order],
                sigma=estimate['sigma'][order],pi=estimate['pi'][np.ix_(order,order)]),order


def parameter_mses(parameters,estimate,align=True):
    if align:
        estimate,_=align_estimate(parameters,estimate)
    # Full K entries / full K*p / full K*p*p / full K*K, Section 4.1.
    return {out:float(np.mean((estimate[key]-parameters[key])**2))
            for out,key in [('alpha','alpha'),('mu','mu'),('Sigma','sigma'),('Pi','pi')]}

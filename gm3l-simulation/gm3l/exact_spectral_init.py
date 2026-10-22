#!/usr/bin/env python3
"""Same spectral initializer, faster computation of its top-K eigenspace.

The mathematical initializer is unchanged: the K largest algebraic eigenvectors
of B=A-diag(A), row normalization, KMeans with n_init=10 and the same seed,
class proportions/means, pooled MLE covariance, and Pi diagonal .7/offdiag .3.
Only the numerical eigensolver differs. ARPACK uses a deterministic starting
vector independent of latent labels/true parameters. No EM updates change.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
import time
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans


def spectral_decomposition(B,K,solver='eigsh',tol=1e-10):
    """Return ascending largest-K eigenpairs and audit diagnostics."""
    n=len(B)
    if not 1<=K<n-1:
        raise ValueError('Require 1 <= K < n-1')
    started=time.perf_counter()
    if solver=='eigsh':
        # Reproducible numerical starting direction, independent of all data
        # parameters and labels. It does not initialize the statistical fit.
        v0=np.random.default_rng(981723).standard_normal(n)
        v0/=np.linalg.norm(v0)
        values,vectors=eigsh(B,k=K,which='LA',tol=tol,
                             ncv=min(n,max(4*K+1,40)),maxiter=10*n,v0=v0)
        order=np.argsort(values)
        values,vectors=values[order],vectors[:,order]
    elif solver=='dense':
        values,vectors=eigh(B,subset_by_index=[n-K,n-1],check_finite=False)
    else:
        raise ValueError("solver must be 'eigsh' or 'dense'")
    eigensolver_seconds=time.perf_counter()-started
    residual=np.linalg.norm(B@vectors-vectors*values[None,:],axis=0)
    relative_residual=residual/np.maximum(1.,np.abs(values))
    orthogonality=float(np.linalg.norm(vectors.T@vectors-np.eye(K),ord=2))
    if np.max(relative_residual)>1e-8 or orthogonality>1e-8:
        raise FloatingPointError('Computed eigenspace did not meet numerical residual audit')
    diagnostics=dict(solver=solver,tolerance=tol,eigensolver_seconds=eigensolver_seconds,
                     max_eigen_residual=float(np.max(residual)),
                     max_relative_eigen_residual=float(np.max(relative_residual)),
                     orthogonality_error=orthogonality,eigenvalues=values.tolist())
    return values,vectors,diagnostics


def theta_from_labels(X,labels,K,config=None):
    n,p=X.shape
    counts=np.bincount(labels,minlength=K)
    if np.any(counts==0):
        raise FloatingPointError('Empty initializer class')
    alpha=counts/n
    mu=np.asarray([X[labels==k].mean(axis=0) for k in range(K)])
    residual=X-X.mean(axis=0)
    covariance=residual.T@residual/n
    sigma=np.repeat(covariance[None,:,:],K,axis=0)
    pi=np.full((K,K),getattr(config,'pi_init_off_diagonal',.30))
    np.fill_diagonal(pi,getattr(config,'pi_init_diagonal',.70))
    return dict(alpha=alpha,mu=mu,sigma=sigma,pi=pi)


def initialize_spectral_theta(X,A,K,seed,config=None,*,solver='eigsh',tol=1e-10,return_diagnostics=False):
    """Drop-in spectral-only initializer; normally returns a parameter dict.

    Set return_diagnostics=True to return (theta, diagnostics). No true labels,
    true means, or other oracle quantities are accepted by this interface.
    """
    started=time.perf_counter()
    X=np.asarray(X,dtype=float)
    B=np.asarray(A,dtype=float).copy()
    np.fill_diagonal(B,0.)
    _,vectors,diagnostics=spectral_decomposition(B,K,solver=solver,tol=tol)
    norm=np.linalg.norm(vectors,axis=1)
    if np.any(norm==0):
        raise FloatingPointError('Zero spectral embedding row')
    features=vectors/norm[:,None]
    labels=KMeans(n_clusters=K,n_init=getattr(config,'kmeans_n_init',10),
                   max_iter=300,random_state=int(seed%(2**31-1))).fit_predict(features)
    theta=theta_from_labels(X,labels,K,config)
    diagnostics['initializer_seconds']=time.perf_counter()-started
    return (theta,diagnostics) if return_diagnostics else theta

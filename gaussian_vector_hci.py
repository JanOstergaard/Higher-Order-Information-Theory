"""Covariance-based Gaussian-family HCI approximation for vector sources.

Inputs may be non-Gaussian. Only the empirical covariance is used; all mutual
information quantities and auxiliaries are computed under a Gaussian model.
For non-Gaussian inputs the estimate is a Gaussian surrogate, with no claim
of optimality for the true data distribution.

All information values are in bits. Requires NumPy; no IDTxl/JVM.
"""
from __future__ import annotations

from itertools import permutations
from typing import Sequence

import numpy as np


def _sqrtm_inv(a: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    e, u = np.linalg.eigh(0.5 * (a + a.T))
    if np.min(e) <= floor:
        raise ValueError(
            "Covariance is singular or insufficiently positive definite; "
            "increase sample size or ridge."
        )
    return (u * (1.0 / np.sqrt(e))) @ u.T


def _as_covariance(
    sources: Sequence[np.ndarray], ridge: float
) -> tuple[np.ndarray, list[slice]]:
    if len(sources) < 2:
        raise ValueError("At least two sources are required.")
    matrices = []
    for x in sources:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2 or x.shape[1] == 0:
            raise ValueError(
                "Each source must have shape (samples,) or (samples, dimensions)."
            )
        if not np.isfinite(x).all():
            raise ValueError("Sources must contain only finite values.")
        matrices.append(x)
    ns = {x.shape[0] for x in matrices}
    if len(ns) != 1 or next(iter(ns)) < 3:
        raise ValueError(
            "Sources must have equal sample counts of at least three."
        )
    dims = [x.shape[1] for x in matrices]
    endpoints = np.cumsum([0] + dims)
    slices = [
        slice(int(a), int(b))
        for a, b in zip(endpoints[:-1], endpoints[1:])
    ]
    x_all = np.concatenate(matrices, axis=1)
    cov = np.atleast_2d(np.cov(x_all, rowvar=False))
    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError("ridge must be a finite nonnegative number.")
    if ridge:
        cov += ridge * np.diag(np.diag(cov))
    return cov, slices


def _gaussian_mi(cov: np.ndarray, x: slice, y: slice) -> float:
    a, b = cov[x, x], cov[y, y]
    cross = _sqrtm_inv(a) @ cov[x, y] @ _sqrtm_inv(b)
    lam = np.clip(
        np.linalg.svd(cross, compute_uv=False) ** 2, 0, 1 - 1e-14
    )
    return float(-0.5 * np.log2(1 - lam).sum())


def _gaussian_ib(cov: np.ndarray, v: slice, w: slice) -> np.ndarray:
    """Adjoin the Gaussian information-bottleneck representation S of V.

    The information budget is I(S;V) = I(V;W). The representation uses
    independent Gaussian noise on canonical-correlation modes.
    """
    sv, sw = cov[v, v], cov[w, w]
    inv_v = _sqrtm_inv(sv)
    cross = inv_v @ cov[v, w] @ _sqrtm_inv(sw)
    matrix = cross @ cross.T
    vals, u = np.linalg.eigh(0.5 * (matrix + matrix.T))
    vals = np.clip(vals[::-1], 0, 1 - 1e-14)
    u = u[:, ::-1]
    capacity = _gaussian_mi(cov, v, w)
    alpha = np.ones(len(vals))

    if capacity > 0 and np.any(vals > 1e-14):
        # For active squared canonical correlations lambda > t:
        # alpha = t * (1-lambda) / (lambda * (1-t)).
        def evaluate(t):
            a = np.ones_like(vals)
            active = vals > t
            a[active] = (
                t * (1 - vals[active]) / (vals[active] * (1 - t))
            )
            rate = float(-0.5 * np.log2(a[active]).sum())
            return rate, a

        lo, hi = 0.0, float(np.max(vals))
        for _ in range(110):
            t = 0.5 * (lo + hi)
            rate, _ = evaluate(t)
            if rate > capacity:
                lo = t
            else:
                hi = t
        _, alpha = evaluate(0.5 * (lo + hi))

    transform = (
        np.sqrt(np.maximum(1 - alpha, 0))[:, None] * u.T
    ) @ inv_v
    cross_all = transform @ cov[v, :]
    variance_s = transform @ sv @ transform.T + np.diag(alpha)
    result = np.block([
        [cov, cross_all.T],
        [cross_all, variance_s],
    ])
    return 0.5 * (result + result.T)


def gaussian_vector_hci(
    sources: Sequence[np.ndarray], *, ridge: float = 1e-9
) -> tuple[float, float]:
    """Return (R2, Rn) using the empirical Gaussian covariance, in bits.

    Each source has shape (samples,) or (samples, dimensions), with possibly
    different dimensions. For n>=3, optimize Gaussian bottleneck stages over
    every reference index and ordering; at n=2 return R2 as Rn.

    The method is valid as a covariance-based Gaussian-family calculation even
    on non-Gaussian observations, but is not generally optimal for those laws.
    """
    cov, indices = _as_covariance(sources, ridge)
    n = len(indices)
    r2 = min(
        _gaussian_mi(cov, indices[i], indices[j])
        for i in range(n) for j in range(i + 1, n)
    )
    if n == 2:
        return r2, r2

    reference_scores = []
    for i in range(n):
        ordering_scores = []
        for order in permutations(j for j in range(n) if j != i):
            current_cov = cov.copy()
            current = indices[i]
            # n-2 successive bottlenecks; last source is for evaluation.
            for k in order[:-1]:
                current_cov = _gaussian_ib(
                    current_cov, current, indices[k]
                )
                d = current.stop - current.start
                current = slice(
                    current_cov.shape[0] - d, current_cov.shape[0]
                )
            ordering_scores.append(
                min(
                    _gaussian_mi(current_cov, current, indices[j])
                    for j in range(n)
                )
            )
        reference_scores.append(max(ordering_scores))
    return max(0., r2), max(0., min(r2, min(reference_scores)))


compute_gaussian_vector_hci = gaussian_vector_hci

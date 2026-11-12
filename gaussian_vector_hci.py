"""Covariance-based Gaussian-family HCI approximation for vector sources.

Inputs may be non-Gaussian. Only the empirical covariance is used; all mutual
information quantities and auxiliaries are computed under a Gaussian model.
For non-Gaussian inputs the estimate is a Gaussian surrogate, with no claim
of optimality for the true data distribution.

All information values are in bits. Requires NumPy; no IDTxl/JVM.
"""
from __future__ import annotations

from itertools import permutations
from numbers import Integral
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


def delay_embed_sources(
    sources: Sequence[np.ndarray], dimension: int, *, lag: int = 1
) -> list[np.ndarray]:
    """Build aligned delay-embedding vectors from scalar time series.

    Each output has shape (N - (dimension-1)*lag, dimension), with row t
    equal to [x[t], x[t+lag], ..., x[t+(dimension-1)*lag]].
    No zero padding or wraparound is performed. The windows overlap.
    """
    for name, value in (("dimension", dimension), ("lag", lag)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if len(sources) < 2:
        raise ValueError("At least two sources are required.")
    arrays = [np.asarray(x, dtype=float) for x in sources]
    if any(x.ndim != 1 for x in arrays):
        raise ValueError("Embedding requires one-dimensional scalar source inputs.")
    if len({len(x) for x in arrays}) != 1:
        raise ValueError("All input sources must have the same number of samples.")
    window = (dimension - 1) * lag + 1
    if len(arrays[0]) < window + 2:
        raise ValueError("At least three observations must remain after embedding.")
    if any(not np.isfinite(x).all() for x in arrays):
        raise ValueError("All input samples must be finite.")
    return [
        np.lib.stride_tricks.sliding_window_view(x, window)[:, ::lag].copy()
        for x in arrays
    ]


def gaussian_vector_hci(
    sources: Sequence[np.ndarray], *, ridge: float = 1e-9,
    embedding_dim: int | None = None, embedding_lag: int = 1,
    normalize: bool = False,
) -> tuple[float, float]:
    """Return (R2, Rn) using the empirical Gaussian covariance, in bits.

    Each source has shape (samples,) or (samples, dimensions), with possibly
    different dimensions. For n>=3, optimize Gaussian bottleneck stages over
    every reference index and ordering; at n=2 return R2 as Rn.

    embedding_dim: any positive integer d, or None for pre-formed vectors.
    embedding_lag: positive integer spacing between time samples (default 1).
    normalize: if True divide both returned quantities by common vector
        dimension d. For unequal pre-formed source dimensions, normalization
        is ambiguous and raises ValueError.

    For embedded scalar series, there are N-(d-1)*embedding_lag rows.
    Normalization is bits per vector component / block sample, not a
    guaranteed information rate for dependent (overlapping) windows.
    The Gaussian-family surrogate is not generally optimal for
    non-Gaussian distributions.
    """
    if embedding_dim is not None:
        sources = delay_embed_sources(sources, embedding_dim, lag=embedding_lag)
    elif embedding_lag != 1:
        raise ValueError("embedding_lag requires embedding_dim.")
    if normalize:
        dimensions = []
        for source in sources:
            a = np.asarray(source)
            if a.ndim not in (1, 2):
                raise ValueError("Sources must be 1D or 2D arrays.")
            dimensions.append(1 if a.ndim == 1 else a.shape[1])
        if len(set(dimensions)) != 1:
            raise ValueError("normalize=True requires all source dimensions equal.")
        divisor = dimensions[0]
    else:
        divisor = 1
    cov, indices = _as_covariance(sources, ridge)
    n = len(indices)
    r2 = min(
        _gaussian_mi(cov, indices[i], indices[j])
        for i in range(n) for j in range(i + 1, n)
    )
    if n == 2:
        return r2 / divisor, r2 / divisor

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
    return max(0., r2) / divisor, max(0., min(r2, min(reference_scores))) / divisor


compute_gaussian_vector_hci = gaussian_vector_hci

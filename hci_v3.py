import numpy as np
from joblib import Parallel, delayed
from idtxl.estimators_jidt import JidtKraskovMI

import numpy as np



def compute_R2_R3(
    Xn,
    target_idx=3,
    x1_idx=0,
    x2_idx=1,
    knn=6,
    repeats=10,
    aggregation="median",
    max_itr=10,
    threshold=0.1,
    n_jobs=1,
):
    Xn = np.asarray(Xn)

    if Xn.ndim != 2:
        raise ValueError("Xn must be 2D with shape (samples, variables).")

    mi_estimator = JidtKraskovMI({"kraskov_k": knn})

    pair_orders = [
        (target_idx, x1_idx),
        (target_idx, x2_idx),
        (x1_idx, x2_idx),
    ]

    R2_mi_s = np.asarray([
        _idtxl_mi_1d(Xn[:, a], Xn[:, b], mi_estimator)
        for a, b in pair_orders
    ], dtype=float)    

    r2 = np.nanmin(R2_mi_s, axis=0)
    r2 = np.maximum(r2, 0.0)

    r3_orders = [
        (target_idx, x1_idx, x2_idx),
        (target_idx, x2_idx, x1_idx),
        (x1_idx, x2_idx, target_idx),
        (x1_idx, target_idx, x2_idx),
        (x2_idx, x1_idx, target_idx),
        (x2_idx, target_idx, x1_idx),
    ]

    R3_mi_s = np.asarray([
        R3_approx_mi_fast(
            Xn, i, j, k,
            knn=knn,
            repeats=repeats,
            aggregation=aggregation,
            max_itr=max_itr,
            threshold=threshold,
            n_jobs=n_jobs,
        )
        for i, j, k in r3_orders
    ], dtype=float)

    r3_a = np.nanmax(R3_mi_s[[0, 1]], axis=0)
    r3_b = np.nanmax(R3_mi_s[[2, 3]], axis=0)
    r3_c = np.nanmax(R3_mi_s[[4, 5]], axis=0)

    r3 = np.minimum(np.minimum(r3_a, r3_b), r3_c)
    r3 = np.maximum(r3, 0.0)

    r3 = np.minimum(r3, r2)

    return r2, r3


def _idtxl_mi_1d(x, y, mi_estimator):
    x = np.asarray(x)
    y = np.asarray(y)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid].reshape(-1, 1)
    y = y[valid].reshape(-1, 1)

    if x.shape[0] < 10:
        return np.nan

    return mi_estimator.estimate(x, y)

def _succ_info_fast(source, partner, mi_estimator, rng, max_itr=30, threshold=0.1):
    n = source.shape[0]

    MI_target = _idtxl_mi_1d(source, partner, mi_estimator)

    if MI_target < 1e-4 or not np.isfinite(MI_target):
        return np.zeros(n), 0.0, source.copy(), np.nan, 0

    Tinit = source.copy()
    eps_noise = rng.standard_normal(n)

    s_low = 1e-8
    s_high = 10.0

    best_T = None
    best_MI = np.inf
    best_s = None

    for tt in range(max_itr):
        s = 0.5 * (s_low + s_high)

        T = Tinit + s * eps_noise

        # IMPORTANT: match I(T ; source), not I(T ; partner)
        MI_new = _idtxl_mi_1d(T, source, mi_estimator)

        if not np.isfinite(MI_new):
            s_high = s
            continue

        if abs(MI_new - MI_target) < abs(best_MI - MI_target):
            best_T = T.copy()
            best_MI = MI_new
            best_s = s

        # More noise lowers I(T ; source)
        if MI_new > MI_target:
            s_low = s
        else:
            s_high = s

        if abs(MI_new - MI_target) / max(abs(MI_target), 1e-12) < threshold:
            break

    return best_T, best_MI, Tinit, best_s, tt + 1





def _single_R3_repeat(source, partner, Yk, knn, seed, max_itr, threshold):
    rng = np.random.default_rng(seed)
    mi_estimator = JidtKraskovMI({"kraskov_k": knn})

    Tsuff, _, _, _, _ = _succ_info_fast(
        source=source,
        partner=partner,
        mi_estimator=mi_estimator,
        rng=rng,
        max_itr=max_itr,
        threshold=threshold,
    )

    return _idtxl_mi_1d(Tsuff, Yk, mi_estimator)



def R3_approx_mi_fast(
    X,
    i,
    j,
    k,
    *,
    knn=6,
    repeats=30,
    aggregation="max",
    max_itr=10,
    threshold=0.1,
    n_jobs=1,
):
    """
    Faster independent R3 approximation using plain MI.

    Parameters
    ----------
    X : ndarray, shape (samples, variables)
    i, j, k : int
        Column indices.
    knn : int
        Kraskov k.
    repeats : int
        Number of stochastic repeats.
    aggregation : {"mean", "median", "max"}
    max_itr : int
        Maximum (bisections) iterations inside sufficient-statistic search.
    threshold: real number between 0 and 1. Typically 0.1. The accuracy of the relative MI appproximation.
    n_jobs : int
        Parallel workers. Use n_jobs=-1 for all cores.

    Returns
    -------
    float
    """

    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("X must be 2D: shape (samples, variables).")

    n_vars = X.shape[1]
    for idx in (i, j, k):
        if idx < 0 or idx >= n_vars:
            raise IndexError(f"Index {idx} is out of bounds for {n_vars} variables.")


    Y1 = np.ascontiguousarray(X[:, i], dtype=float)  # source Xi
    Y2 = np.ascontiguousarray(X[:, j], dtype=float)  # partner Xj
    Yk = np.ascontiguousarray(X[:, k], dtype=float)

    # This is invariant across all repeats, so estimate it once.
    base_estimator = JidtKraskovMI({"kraskov_k": knn})
    MI_12 = _idtxl_mi_1d(Y2, Y1, base_estimator)
    
    seed_rng = np.random.default_rng(42)
    seeds = seed_rng.integers(0, np.iinfo(np.uint32).max, size=repeats)


    if n_jobs == 1:
        vals = []
        mi_estimator = JidtKraskovMI({"kraskov_k": knn})

        for seed in seeds:
            rng = np.random.default_rng(int(seed))

            Tsuff, _, _, _, _ = _succ_info_fast(
                source=Y1,
                partner=Y2,
                mi_estimator=mi_estimator,
                rng=rng,
                max_itr=max_itr,
                threshold=threshold,
            )

            vals.append(_idtxl_mi_1d(Tsuff, Yk, mi_estimator))

    else:
        vals = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_single_R3_repeat)(
                Y1, Y2, Yk, knn, int(seed), max_itr, threshold
            )
            for seed in seeds
        )

        
    vals = np.asarray(vals, dtype=float)

    if aggregation == "mean":
        return np.nanmean(vals)
    if aggregation == "median":
        return np.nanmedian(vals)
    if aggregation == "max":
        return np.nanmax(vals)

    raise ValueError("aggregation must be 'mean', 'median', or 'max'.")


def R4_approx_mi_fast(
    X,
    i,
    j,
    k,
    l,
    *,
    knn=6,
    repeats=30,
    refine_repeats=5,
    aggregation="max",
    max_itr=10,
    threshold=0.1,
    eps=1e-6,
):
    X = np.asarray(X)

    Y1 = np.ascontiguousarray(X[:, i], dtype=float)
    Y2 = np.ascontiguousarray(X[:, j], dtype=float)
    Yk = np.ascontiguousarray(X[:, k], dtype=float)
    Yl = np.ascontiguousarray(X[:, l], dtype=float)

    mi_estimator = JidtKraskovMI({"kraskov_k": knn})
    seed_rng = np.random.default_rng(42)
    seeds = seed_rng.integers(0, np.iinfo(np.uint32).max, size=repeats)

    Tsuffs = []
    R3_vals = []

    MI_12 = _idtxl_mi_1d(Y2, Y1, mi_estimator)

    # First stage: compute several Tsuff realizations
    for seed in seeds:
        rng = np.random.default_rng(int(seed))

        Tsuff, _, _, _, _ = _succ_info_fast(
            source=Y1,
            partner=Y2,
            mi_estimator=mi_estimator,
            rng=rng,
            max_itr=max_itr,
            threshold=threshold,
        )
        
        val = _idtxl_mi_1d(Tsuff, Yk, mi_estimator)

        Tsuffs.append(Tsuff)
        R3_vals.append(val)

    Tsuffs = np.column_stack(Tsuffs)
    R3_vals = np.asarray(R3_vals, dtype=float)

    valid = np.isfinite(R3_vals) #& (R3_vals > eps)

    if not np.any(valid):
        return np.nan, np.nan, MI_12 


    if aggregation == "mean":
        R3_best = np.mean(R3_vals[valid])

    if aggregation == "median":
        R3_best = np.median(R3_vals[valid])
        
    if aggregation == "max":
        R3_best = np.max(R3_vals[valid])


    # Find the best sufficient statistics
    loc = np.nanargmin(np.abs(R3_vals - R3_best))
    Tbest = Tsuffs[:, loc]

    # Second stage: refine using Tbest and X[:, k]
    R4_vals = []


    for r in range(refine_repeats):
        rng = np.random.default_rng(10_000 + r)

        Tsuff_T, _, _, _, _ = _succ_info_fast(
            source=Tbest,
            partner=Yk,
            mi_estimator=mi_estimator,
            rng=rng,
            max_itr=max_itr,
            threshold=threshold,
        )

        R4_vals.append(_idtxl_mi_1d(Tsuff_T, Yl, mi_estimator))
    
    
    R4_vals = np.asarray(R4_vals, dtype=float)
    
    if aggregation == "mean":
        return np.nanmean(R4_vals), R3_best, MI_12
    if aggregation == "median":
        return np.nanmedian(R4_vals), R3_best, MI_12
    if aggregation == "max":
        return np.nanmax(R4_vals), R3_best, MI_12

    raise ValueError("aggregation must be 'mean', 'median', or 'max'.")


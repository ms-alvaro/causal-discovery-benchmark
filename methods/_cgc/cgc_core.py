"""
Conditional Granger Causality (CGC) core.

Implementation based on:
    Geweke, J. (1982). J. Am. Stat. Assoc. 77(378):304-313.
    Barnett, L. & Seth, A.K. (2014). J. Neurosci. Methods 223:50-68.
    As described in Supplementary Section S2.1 of:
    Martínez-Sánchez, Arranz & Lozano-Durán, Nat. Commun. 15, 9296 (2024).
    https://doi.org/10.1038/s41467-024-53373-4

For each ordered pair (source j → target i):
  Restricted:   Q_i(t) = a0 + Σ a_k Q_i(t-k) + Σ c_k Q'(t-k) + ε̂
  Unrestricted: Q_i(t) = a0 + Σ a_k Q_i(t-k) + Σ b_k Q_j(t-k) + Σ c_k Q'(t-k) + ε
  where Q' = all variables except Q_i and Q_j (conditioning set).

  CGC_{j→i} = log₂[ var(ε̂) / var(ε) ]  ≥ 0
"""

import numpy as np

from _surd import it_tools as it


def _ols_resid_var(y: np.ndarray, X: np.ndarray) -> float:
    """OLS residual variance: var(y - X @ θ̂), corrected for dof."""
    theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ theta
    dof   = max(len(y) - X.shape[1], 1)
    return float(np.sum(resid**2) / dof)


def cgc_pairwise(X: np.ndarray, p: int = 1, nbins: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute CGC for every ordered pair (source j → target i).

    Parameters
    ----------
    X : np.ndarray, shape (nvars, N)
    p : int — VAR order (number of lags)

    Returns
    -------
    cgc_matrix : np.ndarray, shape (nvars, nvars)
        cgc_matrix[i, j] = CGC_{j→i}, including self-causation on the diagonal.
    mi_total : np.ndarray, shape (nvars,)
        mi_total[i] = I(Q_i⁺ ; Q), where Q contains all present-time variables.
    """
    nvars, N = X.shape
    T = N - p  # usable time steps

    # Build lag-feature matrix: shape (T, nvars * p)
    # Column v*p + (lag-1) = X[v, t - lag] for t = p..N-1
    lag_feats = np.zeros((T, nvars * p))
    for lag in range(1, p + 1):
        for v in range(nvars):
            lag_feats[:, v * p + (lag - 1)] = X[v, p - lag: N - lag]

    ones = np.ones((T, 1))
    cgc_matrix = np.zeros((nvars, nvars))
    mi_total = np.zeros(nvars)

    ndim = nvars + 1
    nbins = max(3, int((T / 10) ** (1.0 / ndim))) if nbins <= 0 else nbins
    past_present = X[:, :T]
    future = X[:, p:]

    for i in range(nvars):
        y = X[i, p:]  # target: Q_i(t) for t = p..N-1
        joint = np.vstack([future[i], past_present])
        p_hist = it.myhistogram(joint.T, nbins)
        mi_total[i] = float(it.mutual_info(p_hist, (0,), tuple(range(1, p_hist.ndim))))

        all_cols = np.arange(nvars * p)
        X_full = np.hstack([ones, lag_feats[:, all_cols]])
        var_full = _ols_resid_var(y, X_full)

        for j in range(nvars):
            src_cols  = np.arange(j * p, j * p + p)
            restricted_cols = np.array([c for c in all_cols if c // p != j])
            X_rest = np.hstack([ones, lag_feats[:, restricted_cols]]) if len(restricted_cols) else ones
            var_rest = _ols_resid_var(y, X_rest)

            if var_full <= 0 or var_rest <= 0:
                cgc_matrix[i, j] = 0.0
            else:
                cgc_matrix[i, j] = max(0.0, np.log2(var_rest / var_full))

    return cgc_matrix, mi_total

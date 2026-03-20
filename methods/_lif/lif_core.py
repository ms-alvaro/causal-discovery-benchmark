"""
Liang's Information Flow (LIF) core — bivariate (pairwise) version.

Implementation based on:
    Liang, X.S. (2014). Phys. Rev. E 90:052150.
        (original bivariate formula)
    Liang, X.S. (2021). Entropy 23(6):679.
        (normalized multivariate extension)

For each ordered pair (source j → target i):

  T_{j→i} = (C_ii · C_ij · C_{j,di} − C_ij² · C_{i,di})
             / (C_ii² · C_jj − C_ii · C_ij²)

where:
  C_ij    = sample covariance of X_i and X_j
  C_{k,di} = sample covariance of X_k and dX_i/dt
  dX_i/dt ≈ (X_i(t+ΔT) − X_i(t)) / ΔT  (Euler forward difference)

T_{j→i} > 0: information flows from j to i (j causally influences i).
T_{j→i} ≤ 0: no causal influence (or negative feedback).
Clamped at 0 for display; diagonal = 0.
"""

import numpy as np


def lif_pairwise(X: np.ndarray, nlag: int = 1) -> np.ndarray:
    """
    Compute Liang Information Flow for every ordered pair (source j → target i).

    Parameters
    ----------
    X    : np.ndarray, shape (nvars, N)
    nlag : int — time lag used for Euler forward differences

    Returns
    -------
    lif_matrix : np.ndarray, shape (nvars, nvars)
        lif_matrix[i, j] = T_{j→i} (raw; may be negative).
        Diagonal = 0.
    """
    nvars, N = X.shape
    dt = float(nlag)

    # Time series at t and forward-difference derivative dX/dt
    X_t   = X[:, :-nlag]                         # (nvars, n)
    X_dot = (X[:, nlag:] - X[:, :-nlag]) / dt   # (nvars, n)

    n = X_t.shape[1]

    # Center
    X_c    = X_t   - X_t.mean(axis=1, keepdims=True)
    Xdot_c = X_dot - X_dot.mean(axis=1, keepdims=True)

    # C[i, j] = cov(Xi, Xj) — standard sample covariance matrix
    C = (X_c @ X_c.T) / (n - 1)          # (nvars, nvars)

    # D[i, j] = cov(Xj, dXi/dt)
    # = (1/(n-1)) * sum_t (Xj_c(t) * Xdot_c_i(t))
    # = Xdot_c[i] · X_c[j] / (n-1)
    # As a matrix: D = Xdot_c @ X_c.T / (n-1)
    D = (Xdot_c @ X_c.T) / (n - 1)      # D[i, j] = cov(Xj, dXi/dt)

    lif_matrix = np.zeros((nvars, nvars))

    for i in range(nvars):
        for j in range(nvars):
            if i == j:
                continue

            Cii   = C[i, i]
            Cjj   = C[j, j]
            Cij   = C[i, j]
            Cidi  = D[i, i]   # cov(Xi, dXi/dt)
            Cjdi  = D[i, j]   # cov(Xj, dXi/dt)

            denom = Cii ** 2 * Cjj - Cii * Cij ** 2
            if abs(denom) < 1e-15:
                continue

            lif_matrix[i, j] = (Cii * Cij * Cjdi - Cij ** 2 * Cidi) / denom

    return lif_matrix

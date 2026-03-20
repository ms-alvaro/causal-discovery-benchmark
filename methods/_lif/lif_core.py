"""
Liang's Information Flow (LIF) core — multivariate version.

Implementation based on:
    Liang, X.S. (2014). Phys. Rev. E 90:052150.
        (bivariate closed-form estimator)
    Liang, X.S. (2021). Entropy 23(6):679.
        (multivariate extension via cofactor matrix)
    Climdyn/Liang_Index_climdyn (reference implementation).

For each ordered pair (source j → target i), conditioned on all other variables:

  T_{j→i} = (C_{ij} / C_{ii}) * (1/det(C)) * sum_k [ Delta_{jk} * dC_{ki} ]

where:
  C       = nvars × nvars sample covariance matrix of X
  Delta   = cofactor matrix of C: Delta = inv(C).T * det(C)
  dC_{ki} = sample covariance of X_k and dX_i/dt
  dX_i/dt ≈ (X_i(t+ΔT) − X_i(t)) / ΔT  (Euler forward difference)

For nvars=2 this reduces exactly to the bivariate Liang 2014 formula.
For nvars>2 the cofactor construction conditions on all remaining variables.

T_{j→i} > 0: information flows from j to i (j causally influences i).
T_{j→i} ≤ 0: no causal influence (or negative feedback).
Diagonal = 0.
"""

import numpy as np


def lif_pairwise(X: np.ndarray, nlag: int = 1) -> np.ndarray:
    """
    Compute Liang Information Flow for every ordered pair (source j → target i).

    Uses the multivariate formula (Liang 2021, Entropy 23:679) which conditions
    on all observed variables simultaneously via the cofactor matrix of C.

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

    # C[i, j] = cov(Xi, Xj) — full nvars × nvars covariance matrix
    C = (X_c @ X_c.T) / (n - 1)                 # (nvars, nvars)

    # D[i, j] = cov(Xj, dXi/dt)
    # D[i, j] = Xdot_c[i] · X_c[j] / (n-1)
    D = (Xdot_c @ X_c.T) / (n - 1)              # D[i, j] = cov(Xj, dXi/dt)

    # Cofactor matrix: Delta = inv(C).T * det(C)
    detC = np.linalg.det(C)
    if abs(detC) < 1e-15:
        return np.zeros((nvars, nvars))
    Delta = np.linalg.inv(C).T * detC            # (nvars, nvars)

    lif_matrix = np.zeros((nvars, nvars))

    for i in range(nvars):
        Cii = C[i, i]
        if abs(Cii) < 1e-15:
            continue
        # D[i, :] = cov(X_k, dXi/dt) for all k  →  dC[:, i] in Liang notation
        for j in range(nvars):
            if i == j:
                continue
            # T[j→i] = (C[i,j] / C[i,i]) * (1/detC) * sum_k( Delta[j,k] * D[i,k] )
            lif_matrix[i, j] = (C[i, j] / Cii) * (1.0 / detC) * np.dot(Delta[j], D[i])

    return lif_matrix

"""
Conditional Transfer Entropy (CTE) core.

Implementation based on:
    Schreiber, T. (2000). Phys. Rev. Lett. 85:461-464.  (transfer entropy)
    Barnett, L., Barrett, A.B. & Seth, A.K. (2009). Phys. Rev. Lett. 103:238701.
        (equivalence CGC = 2 * CTE for Gaussian variables)
    As described in Supplementary Section S2.4 of:
    Martínez-Sánchez, Arranz & Lozano-Durán, Nat. Commun. 15, 9296 (2024).
    https://doi.org/10.1038/s41467-024-53373-4

For each ordered pair (source j → target i), with single lag ΔT:

  CTE_{j→i} = H(Q_i⁺ | Q̄_j) − H(Q_i⁺ | Q)

where:
  Q   = [Q_1(t−ΔT), …, Q_N(t−ΔT)]  (all variables at lag ΔT)
  Q̄_j = Q with Q_j removed
  Q_i⁺ = Q_i(t)

Entropy is estimated via histogram (uniform binning).
CTE ≥ 0 by construction (clamped at 0 if negative due to estimation noise).
"""

import numpy as np


def _entropy(data: np.ndarray, nbins: int) -> float:
    """Shannon entropy H(data) in bits, estimated by uniform histogramming."""
    hist, _ = np.histogramdd(data, bins=nbins)
    p = hist.ravel()
    p = p[p > 0].astype(float)
    p /= p.sum()
    return float(-np.sum(p * np.log2(p)))


def cte_pairwise(X: np.ndarray, nbins: int = 0, nlag: int = 1) -> np.ndarray:
    """
    Compute CTE for every ordered pair (source j → target i).

    Parameters
    ----------
    X     : np.ndarray, shape (nvars, N)
    nbins : int — histogram bins per dimension
    nlag  : int — time lag

    Returns
    -------
    cte_matrix : np.ndarray, shape (nvars, nvars)
        cte_matrix[i, j] = CTE_{j→i}.  Diagonal = 0.
    """
    nvars, N = X.shape
    n  = N - nlag

    # Adaptive nbins: aim for ~10 samples per histogram cell in (nvars+1)-D space.
    # Overrides any externally passed value to prevent sparse histograms.
    ndim  = nvars + 1          # joint space dimension (target future + all past vars)
    nbins = max(3, int((n / 10) ** (1.0 / ndim)))

    past   = X[:, :n].T          # (n, nvars) — all variables at t
    future = X[:, nlag:].T       # (n, nvars) — all variables at t+nlag

    cte_matrix = np.zeros((nvars, nvars))

    for i in range(nvars):
        yi_plus = future[:, i: i + 1]              # (n, 1)  target future

        # H(Q_i⁺ | Q) = H(Q_i⁺, Q) − H(Q)
        joint_Q    = np.hstack([yi_plus, past])     # (n, nvars+1)
        H_full     = _entropy(joint_Q,  nbins) - _entropy(past, nbins)

        for j in range(nvars):
            if i == j:
                continue

            # Q̄_j: all past variables except Q_j
            other_cols = [k for k in range(nvars) if k != j]
            Q_bar_j    = past[:, other_cols]        # (n, nvars-1)

            # H(Q_i⁺ | Q̄_j) = H(Q_i⁺, Q̄_j) − H(Q̄_j)
            joint_bar  = np.hstack([yi_plus, Q_bar_j])  # (n, nvars)
            H_bar      = _entropy(joint_bar, nbins) - _entropy(Q_bar_j, nbins)

            cte_matrix[i, j] = max(0.0, H_bar - H_full)

    return cte_matrix

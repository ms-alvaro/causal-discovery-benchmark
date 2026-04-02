"""
Conditional Transfer Entropy (CTE) core.

Implementation based on:
    Schreiber, T. (2000). Phys. Rev. Lett. 85:461-464.  (transfer entropy)
    Barnett, L., Barrett, A.B. & Seth, A.K. (2009). Phys. Rev. Lett. 103:238701.
        (equivalence CGC = 2 * CTE for Gaussian variables)
    As described in Supplementary Section S2.4 of:
    Martínez-Sánchez, Arranz & Lozano-Durán, Nat. Commun. 15, 9296 (2024).
    https://doi.org/10.1038/s41467-024-53373-4

The implementation below follows the multivariate information-flux
decomposition:

  T(j) = H(Q_i⁺ | Q_not_j) - H(Q_i⁺ | Q_all)

for every source subset j, followed by inclusion-exclusion subtraction so each
subset contribution is non-overlapping. The benchmark wrapper still reports the
singleton terms T((j,)), which coincide with pairwise conditional transfer
entropy from source j to target i.
"""

from itertools import chain as ichain
from itertools import combinations as icmb

import numpy as np

from _surd import it_tools as it


def information_flux(p: np.ndarray) -> dict:
    """
    Compute the full information-flux decomposition from present-time sources
    to a future target, given a joint PDF p(target_future, sources_present...).

    Returns
    -------
    dict
        Maps tuples of source indices (1-based within p dimensions) to their
        flux into the target variable in dimension 0.
    """
    num_dims = p.ndim
    source_inds = tuple(range(1, num_dims))
    flux = {}

    for size in source_inds:
        for subset in icmb(source_inds, size):
            non_subset = tuple(idx for idx in source_inds if idx not in subset)
            h_without_subset = it.cond_entropy(p, (0,), non_subset)
            h_with_all = it.cond_entropy(p, (0,), source_inds)
            flux[subset] = h_without_subset - h_with_all

    for size in source_inds:
        for subset in icmb(source_inds, size):
            lower_orders = [list(icmb(subset, k)) for k in range(1, len(subset))]
            flux[subset] -= sum(flux[sub] for sub in ichain.from_iterable(lower_orders))

    return flux


def cte_pairwise(X: np.ndarray, nbins: int = 0, nlag: int = 1) -> tuple[np.ndarray, np.ndarray]:
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
        cte_matrix[i, j] = CTE_{j→i}, including self-flux on the diagonal.
    mi_total : np.ndarray, shape (nvars,)
        mi_total[i] = I(Q_i⁺ ; Q), where Q contains all present-time variables.
    """
    nvars, N = X.shape
    n  = N - nlag

    # Adaptive nbins: aim for ~10 samples per histogram cell in (nvars+1)-D space.
    # Overrides any externally passed value to prevent sparse histograms.
    ndim  = nvars + 1          # joint space dimension (target future + all past vars)
    nbins = max(3, int((n / 10) ** (1.0 / ndim)))

    past   = X[:, :n]
    future = X[:, nlag:]

    cte_matrix = np.zeros((nvars, nvars))
    mi_total = np.zeros(nvars)

    for i in range(nvars):
        # Build [Q_i(t+nlag), Q_1(t), ..., Q_N(t)] so dimension 0 is the target future.
        joint = np.vstack([future[i], past])
        p = it.myhistogram(joint.T, nbins)
        flux = information_flux(p)
        mi_total[i] = float(it.mutual_info(p, (0,), tuple(range(1, p.ndim))))

        for j in range(nvars):
            # Histogram dimensions are 1-based after the future target in dim 0.
            cte_matrix[i, j] = max(0.0, float(flux.get((j + 1,), 0.0)))

    return cte_matrix, mi_total

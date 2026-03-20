"""
Convergent Cross-Mapping (CCM) core.

Implementation based on:
    Sugihara, G. et al. (2012). Science 338(6106):496-500.
    Takens, F. (1981). Lecture Notes in Mathematics 898:366-381.
    As described in Supplementary Section S2.2 of:
    Martínez-Sánchez, Arranz & Lozano-Durán, Nat. Commun. 15, 9296 (2024).
    https://doi.org/10.1038/s41467-024-53373-4

For each ordered pair (source j → target i):
  - Build shadow manifold M_i using delay-coordinate embedding of Q_i:
      M_i(t) = [Q_i(t), Q_i(t − ΔT), …, Q_i(t − (E−1)ΔT)]
  - For each time point t, find the E+1 nearest neighbours in M_i.
  - Reconstruct Q_j(t) using exponentially weighted neighbours:
      Q̂_j(t)|M_i = Σ w_k Q_j(t_k)  with  w_k ∝ exp(−d_k / d_1)
  - CCM_{j→i} = corr(Q_j, Q̂_j|M_i)

Interpretation: a high correlation means M_i retains information about Q_j,
which implies Q_j causally influences Q_i (Takens' theorem).
"""

import numpy as np


def _embed(x: np.ndarray, E: int, nlag: int) -> np.ndarray:
    """
    Time-delay embedding of 1-D series x.

    Returns array of shape (T, E) where T = len(x) − (E−1)*nlag and
    row t = [x[t + (E−1)*nlag], x[t + (E−2)*nlag], …, x[t]].
    """
    N = len(x)
    T = N - (E - 1) * nlag
    M = np.zeros((T, E))
    for e in range(E):
        M[:, e] = x[(E - 1 - e) * nlag: N - e * nlag if e > 0 else N]
    return M


def ccm_pairwise(
    X: np.ndarray,
    E: int | None = None,
    nlag: int = 1,
    N_max: int = 3000,
) -> np.ndarray:
    """
    Compute CCM for every ordered pair (source j → target i).

    Parameters
    ----------
    X     : np.ndarray, shape (nvars, N)
    E     : int — embedding dimension (default: nvars, as in the paper)
    nlag  : int — embedding lag ΔT
    N_max : int — maximum library size (subsampled for speed)

    Returns
    -------
    ccm_matrix : np.ndarray, shape (nvars, nvars)
        ccm_matrix[i, j] = CCM_{j→i}.  Diagonal = 0.
    """
    nvars, N = X.shape
    if E is None:
        E = nvars

    # Build delay-embedded manifolds for all variables
    T_full = N - (E - 1) * nlag      # total usable length

    # Subsample if needed
    if T_full > N_max:
        rng  = np.random.default_rng(42)
        idx  = rng.choice(T_full, size=N_max, replace=False)
        idx.sort()
    else:
        idx = np.arange(T_full)

    # Manifolds: M[i] has shape (N_max, E)
    manifolds = {}
    Qj_vals   = {}
    for v in range(nvars):
        M_full       = _embed(X[v], E, nlag)        # (T_full, E)
        manifolds[v] = M_full[idx]                  # (N_max, E)
        # Q_j aligned with M: the "present" time for row k is t = idx[k] + (E-1)*nlag
        # so Q_j at the same time = X[v, idx[k] + (E-1)*nlag]
        Qj_vals[v]   = X[v, idx + (E - 1) * nlag]  # (N_max,)

    ccm_matrix = np.zeros((nvars, nvars))
    n          = len(idx)

    for i in range(nvars):
        Mi   = manifolds[i]   # shadow manifold of Q_i, shape (n, E)

        # Pairwise squared distances in M_i
        sq_dist = np.sum((Mi[:, None, :] - Mi[None, :, :])**2, axis=2)
        np.fill_diagonal(sq_dist, np.inf)

        # E+1 nearest neighbours for every point
        nn_idx = np.argpartition(sq_dist, E + 1, axis=1)[:, :E + 1]  # (n, E+1)
        nn_dist = np.take_along_axis(sq_dist, nn_idx, axis=1)**0.5    # (n, E+1)

        for j in range(nvars):
            if i == j:
                continue

            Qj = Qj_vals[j]     # (n,) — Q_j values at the manifold times
            reconstructed = np.zeros(n)

            for t in range(n):
                d  = nn_dist[t]           # distances to E+1 neighbours
                d1 = d.min()
                if d1 == 0.0:
                    # Degenerate: average over neighbours
                    reconstructed[t] = Qj[nn_idx[t]].mean()
                else:
                    u = np.exp(-d / d1)
                    w = u / u.sum()
                    reconstructed[t] = np.dot(w, Qj[nn_idx[t]])

            Qj_target = Qj_vals[j]
            # Pearson correlation between true Q_j and reconstructed Q̂_j|M_i
            corr = float(np.corrcoef(Qj_target, reconstructed)[0, 1])
            ccm_matrix[i, j] = max(0.0, corr)  # clamp negative values to 0

    return ccm_matrix

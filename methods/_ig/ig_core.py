"""
Imbalance Gain (IG) causality core.

Faithful Python implementation based on the code from:
    Del Tatto, Fortunato, Bueti & Laio, PNAS 121, e2317256121 (2024).
    https://doi.org/10.1073/pnas.2317256121
    Source: https://github.com/vdeltatto/imbalance-gain-causality

For each ordered pair (source j → target i):
  - Form the combined present state: Z = (alpha * X_j[t], X_i[t])
  - Compute the Information Imbalance:
        Delta((alpha*X_j, X_i)_t -> X_i_{t+tau})
      = (2/N) * mean_i [ rank of Z_i's nearest neighbour in the X_i future space ]
  - At alpha=0, only X_i(t) is used; Delta(alpha=0) is the self-prediction baseline.
  - If X_j causes X_i, increasing alpha decreases Delta.
  - Imbalance Gain: IG = (Delta(alpha=0) - min_alpha Delta(alpha)) / Delta(alpha=0)
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata


def _rank_matrix(data: np.ndarray) -> np.ndarray:
    """
    (N, N) rank matrix where entry [i, j] = rank of point j among
    all neighbours of point i (1 = closest, N-1 = farthest).
    Diagonal entries set to np.inf so self is excluded from ranking.

    Matches utilities.compute_rank_matrix from the original repository.
    """
    dist = squareform(pdist(data, metric="euclidean")).astype(float)
    np.fill_diagonal(dist, np.inf)
    ranks = rankdata(dist, method="average", axis=1).astype(float)
    np.fill_diagonal(ranks, np.inf)
    return ranks


def _knn_indices(rank_mat: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Return (N, k) array of k-nearest-neighbour indices for each point.

    Matches utilities.nns_index_array from the original repository.
    """
    N = rank_mat.shape[0]
    nns = np.zeros((N, k), dtype=int)
    for i in range(N):
        nns[i] = np.argpartition(rank_mat[i], np.arange(k))[:k]
    return nns


def _info_imbalance(
    X_present: np.ndarray,
    Y_present: np.ndarray,
    rank_mat_Yf: np.ndarray,
    alpha: float,
    k: int = 1,
) -> float:
    """
    Compute Delta( (alpha * X_present, Y_present) -> Y_future ).

    At alpha=0 the combined space collapses to Y_present alone, giving the
    baseline (how well Y predicts its own future).  As alpha increases, X's
    information is added; a decreasing Delta indicates X causes Y.

    Matches imbalance_gain.compute_info_imbalance_causality from the original
    repository, with the inner loop vectorised.
    """
    N = X_present.shape[0]
    combined = np.column_stack([alpha * X_present, Y_present])
    rank_mat_A = _rank_matrix(combined)
    nns_A = _knn_indices(rank_mat_A, k=k)  # (N, k)
    # Conditional ranks: for each point i, look up the rank of its k-NNs
    # (found in combined space) within the Y_future rank matrix.
    cond_ranks = rank_mat_Yf[np.arange(N)[:, None], nns_A].mean(axis=1)
    return float(2.0 / N * np.mean(cond_ranks))


def ig_pairwise(
    X: np.ndarray,
    nlag: int = 1,
    alphas: np.ndarray | None = None,
    k: int = 1,
    N_max: int = 3000,
) -> np.ndarray:
    """
    Compute Imbalance Gain for every ordered pair (source j → target i).

    Parameters
    ----------
    X     : np.ndarray, shape (nvars, N)
    nlag  : int   — time lag for the future state
    alphas: 1-D array — alpha values to scan; must include 0.0 as first element
    k     : int   — number of nearest neighbours
    N_max : int   — subsample size (rank-matrix computation is O(N²))

    Returns
    -------
    ig_matrix : np.ndarray, shape (nvars, nvars)
        ig_matrix[i, j] = Imbalance Gain for source j → target i.
        Diagonal entries are 0 (self excluded).
    """
    if alphas is None:
        alphas = np.concatenate([[0.0], np.logspace(-1, 1, 19)])

    nvars, N_full = X.shape

    # Subsample a contiguous window of N_max time steps to keep computation tractable.
    n_pairs = N_full - nlag
    if n_pairs > N_max:
        rng  = np.random.default_rng(42)
        idx  = rng.choice(n_pairs, size=N_max, replace=False)
        idx.sort()
    else:
        idx = np.arange(n_pairs)

    ig_matrix = np.zeros((nvars, nvars))

    for i in range(nvars):
        Y_present    = X[i, idx].reshape(-1, 1)
        Y_future     = X[i, idx + nlag].reshape(-1, 1)
        rank_mat_Yf  = _rank_matrix(Y_future)

        for j in range(nvars):
            if i == j:
                continue

            X_present = X[j, idx].reshape(-1, 1)

            # Scan alphas — matches imbalance_gain.scan_alphas (sequential version)
            imbs = [
                _info_imbalance(X_present, Y_present, rank_mat_Yf, alpha, k)
                for alpha in alphas
            ]

            # Imbalance Gain — matches imbalance_gain.compute_imbalance_gain
            delta0 = imbs[0]
            ig_matrix[i, j] = (delta0 - min(imbs)) / delta0 if delta0 > 0 else 0.0

    return ig_matrix

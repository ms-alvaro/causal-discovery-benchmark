"""
ACI core — Assimilative Causal Inference (discrete-time adaptation).

Faithful Python translation of the CGNS filter/smoother from:
    Andreou, Chen & Bollt, Nat. Commun. 17, 1854 (2026).
    https://doi.org/10.1038/s41467-026-68568-0

For each ordered pair (source j → target i):
  - Treat Q_j as the unobserved variable (latent state y).
  - Treat Q_i as the observed variable (x).
  - Fit a linear observation model: x[n+1] = C*y[n] + f_x + noise_x
  - Fit a linear state model:       y[n+1] = A*y[n] + f_y + noise_y
  - Run the Kalman filter (forward) and RTS smoother (backward).
  - Compute the ACI metric (KL divergence smoother || filter) at each step.

ACI metric at time n:
    signal     = 0.5 * (mu_s[n] - mu_f[n])^2 / P_f[n]
    dispersion = 0.5 * (-log(P_s[n]/P_f[n]) + P_s[n]/P_f[n] - 1)
    ACI[n]     = signal + dispersion
"""

import numpy as np


def _ols(y, X):
    """OLS fit: y = X @ theta. Returns theta, residual variance."""
    theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ theta
    var   = float(np.var(resid, ddof=X.shape[1]))
    return theta, max(var, 1e-10)


def _estimate_params(Q_source, Q_target):
    """
    Estimate CGNS parameters for source → target causality test.

    Observation model: Q_target[n+1] = C * Q_source[n] + f_x + noise_x
    State model:       Q_source[n+1] = A * Q_source[n] + f_y + noise_y

    Returns (A, f_y, Q_noise, C, f_x, R_noise).
    """
    N = Q_source.shape[0]
    ones = np.ones(N - 1)

    # Observation: Q_target[1:] ~ C * Q_source[:-1] + f_x
    X_obs = np.column_stack([Q_source[:-1], ones])
    (C, f_x), R_noise = _ols(Q_target[1:], X_obs)

    # State: Q_source[1:] ~ A * Q_source[:-1] + f_y
    X_st = np.column_stack([Q_source[:-1], ones])
    (A, f_y), Q_noise = _ols(Q_source[1:], X_st)

    return float(A), float(f_y), Q_noise, float(C), float(f_x), R_noise


def _kalman_filter(x_obs, A, f_y, Q_noise, C, f_x, R_noise, mu0, P0):
    """
    Forward Kalman filter for scalar state y observed through scalar x.

    Returns filter_means, filter_covs, pred_means, pred_covs (all length N).
    """
    N = len(x_obs)
    mu_f = np.zeros(N)
    P_f  = np.zeros(N)
    mu_p = np.zeros(N)  # predicted (prior)
    P_p  = np.zeros(N)

    mu = mu0
    P  = P0

    for n in range(N):
        # Predict
        mu_pred = A * mu + f_y
        P_pred  = A**2 * P + Q_noise

        mu_p[n] = mu_pred
        P_p[n]  = P_pred

        # Update (innovation form)
        innov = x_obs[n] - (C * mu_pred + f_x)
        S_inn = C**2 * P_pred + R_noise      # innovation variance
        K     = P_pred * C / S_inn           # Kalman gain
        mu    = mu_pred + K * innov
        P     = (1.0 - K * C) * P_pred
        P     = max(P, 1e-12)               # numerical floor

        mu_f[n] = mu
        P_f[n]  = P

    return mu_f, P_f, mu_p, P_p


def _rts_smoother(mu_f, P_f, mu_p, P_p, A):
    """
    Rauch–Tung–Striebel (RTS) smoother — backward pass.

    Returns smoother_means, smoother_covs (same length as filter arrays).
    Matches the backward CGNS smoother equations in the MATLAB code.
    """
    N = len(mu_f)
    mu_s = np.zeros(N)
    P_s  = np.zeros(N)

    # Terminal condition: smoother = filter at last step
    mu_s[-1] = mu_f[-1]
    P_s[-1]  = P_f[-1]

    for n in range(N - 2, -1, -1):
        G        = P_f[n] * A / P_p[n + 1]      # smoother gain
        mu_s[n]  = mu_f[n] + G * (mu_s[n + 1] - mu_p[n + 1])
        P_s[n]   = P_f[n] + G**2 * (P_s[n + 1] - P_p[n + 1])
        P_s[n]   = max(P_s[n], 1e-12)

    return mu_s, P_s


def _aci_from_distributions(mu_f, P_f, mu_s, P_s):
    """
    ACI metric = KL(smoother || filter) at each time step.
    Matches Eq. in Andreou et al. (2026):
        signal     = 0.5 * (mu_s - mu_f)^2 / P_f
        dispersion = 0.5 * (-log(P_s/P_f) + P_s/P_f - 1)
        ACI        = signal + dispersion
    """
    cov_ratio  = P_s / P_f
    signal     = 0.5 * (mu_s - mu_f)**2 / P_f
    dispersion = 0.5 * (-np.log(cov_ratio) + cov_ratio - 1.0)
    return signal + dispersion


def aci_pairwise(X: np.ndarray, nlag: int = 1) -> np.ndarray:
    """
    Compute mean ACI for every ordered pair (source j → target i).

    Parameters
    ----------
    X : np.ndarray, shape (nvars, N)
    nlag : int
        Time lag (number of steps). Currently only nlag=1 is supported.

    Returns
    -------
    aci_matrix : np.ndarray, shape (nvars, nvars)
        aci_matrix[i, j] = mean ACI for source j → target i.
        Diagonal entries are 0.
    aci_ts : dict {(i, j): np.ndarray}
        Full ACI time series for each pair.
    """
    nvars, N = X.shape
    aci_matrix = np.zeros((nvars, nvars))
    aci_ts     = {}

    for i in range(nvars):
        for j in range(nvars):
            if i == j:
                continue

            Q_source = X[j]
            Q_target = X[i]

            A, f_y, Q_noise, C, f_x, R_noise = _estimate_params(Q_source, Q_target)

            # Initial conditions: prior mean = sample mean, large variance
            mu0 = float(np.mean(Q_source))
            P0  = float(np.var(Q_source)) * 10.0

            # Observations start from index 1 (first predicted observation)
            x_obs = Q_target[1:]

            mu_f, P_f, mu_p, P_p = _kalman_filter(
                x_obs, A, f_y, Q_noise, C, f_x, R_noise, mu0, P0
            )
            mu_s, P_s = _rts_smoother(mu_f, P_f, mu_p, P_p, A)

            aci = _aci_from_distributions(mu_f, P_f, mu_s, P_s)

            aci_ts[(i, j)]     = aci
            aci_matrix[i, j]   = float(np.mean(aci))

    return aci_matrix, aci_ts

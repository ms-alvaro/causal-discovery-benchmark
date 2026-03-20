"""
PCMCI — Peter–Clark Algorithm with Momentary Conditional Independence Test

Definition of causality:
    Phase 1 (PC): prune a fully connected graph by testing conditional
    independence with increasing conditioning sets (significance αPC).
    Phase 2 (MCI): for each remaining link Q_j(t−ΔT) → Q_i(t), test
        Q_j(t−ΔT) ⊥⊥ Q_i(t) | P[Q_i(t)] setminus {Q_j}, P[Q_j(t−ΔT)]
    using conditional mutual information (k-NN estimator).
    Causal strength = CMI of remaining links; pruned links = 0.

Reference:
    Runge, J. et al., Sci. Adv. 5(11):eaau4996 (2019).
    Python package: Tigramite (https://github.com/jakobrunge/tigramite).
    As described in Supplementary S2.3 of:
    Martínez-Sánchez, Arranz & Lozano-Durán, Nat. Commun. 15, 9296 (2024).
    https://doi.org/10.1038/s41467-024-53373-4
"""
import os
import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import numpy as np
from pathlib import Path

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

# Tigramite imports — suppress verbose output
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI as _PCMCI
    from tigramite.independence_tests.cmiknn import CMIknn

NAME       = "PCMCI"
DEFINITION = (
    "Two-phase causal discovery: PC parent selection via conditional independence "
    "testing, followed by MCI test using CMI (k-NN) to remove spurious links."
)
REFERENCE  = (
    "Runge et al., Sci. Adv. 5:eaau4996 (2019); "
    "Tigramite package: https://github.com/jakobrunge/tigramite; "
    "Martínez-Sánchez, Arranz & Lozano-Durán, Nat. Commun. 15, 9296 (2024). "
    "https://doi.org/10.1038/s41467-024-53373-4"
)

_EXPECTED_CAUSAL = {
    1: {(0, 1), (1, 2)},
    2: {(0, 2), (1, 2)},
    3: {(0, 1), (0, 2)},
    4: {(0, 1), (0, 2), (1, 2), (2, 1)},
}
_DIAGRAM_PNG = {
    1: Path(__file__).parent.parent / "benchmarks" / "figures" / "mediator.png",
    2: Path(__file__).parent.parent / "benchmarks" / "figures" / "confounder.png",
    3: Path(__file__).parent.parent / "benchmarks" / "figures" / "synergistic.png",
    4: Path(__file__).parent.parent / "benchmarks" / "figures" / "redundant.png",
}
_BAR_COLOR     = "#D8D8D8"
_HATCH_PATTERN = "///"
_SPURIOUS_THR  = 0.25
_ABS_THR       = 1e-6

# PCMCI parameters (from the paper's supplement)
_ALPHA_PC  = 0.05
_ALPHA_MCI = 0.01
_TAU_MAX   = 1
_N_MAX     = 5000   # subsample for k-NN CMI (expensive for large N)


def run(X: np.ndarray, nbins: int = 50, nlag: int = 1) -> list:
    """
    Run PCMCI on multivariate time series X.

    Parameters
    ----------
    X     : np.ndarray, shape (nvars, N)
    nbins : int  — unused (kept for interface compatibility)
    nlag  : int  — maximum lag (tau_max)

    Returns
    -------
    list of dict per target variable i:
        'pcmci_row' : np.ndarray (nvars,) — MCI test statistic (0 if pruned)
        'nvars'     : int
    """
    nvars, N_full = X.shape

    # Subsample if needed (k-NN CMI is O(N log N) but expensive at N=200k)
    if N_full > _N_MAX:
        rng = np.random.default_rng(42)
        idx = rng.choice(N_full, size=_N_MAX, replace=False)
        idx.sort()
        X_use = X[:, idx]
    else:
        X_use = X

    # Tigramite expects shape (T, nvars)
    data_obj = pp.DataFrame(X_use.T)

    cmi_test = CMIknn(significance="shuffle_test",
                      knn=0.1,
                      shuffle_neighbors=5,
                      transform="ranks")

    pcmci = _PCMCI(dataframe=data_obj,
                   cond_ind_test=cmi_test,
                   verbosity=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results_pcmci = pcmci.run_pcmci(
            tau_min=1,
            tau_max=nlag,
            pc_alpha=_ALPHA_PC,
        )

    # val_matrix[i, j, tau-1] = MCI test statistic for Q_j(t-tau) → Q_i(t)
    # p_matrix[i, j, tau-1]   = p-value
    val_matrix = results_pcmci["val_matrix"]    # (nvars, nvars, tau_max)
    p_matrix   = results_pcmci["p_matrix"]

    # Build pairwise causal strength matrix at lag nlag:
    # Zero out links that are NOT significant at alpha_MCI
    causal_matrix = np.zeros((nvars, nvars))
    for i in range(nvars):
        for j in range(nvars):
            if i == j:
                continue
            val = float(val_matrix[i, j, nlag])
            p   = float(p_matrix[i, j, nlag])
            causal_matrix[i, j] = val if p < _ALPHA_MCI else 0.0

    return [{"pcmci_row": causal_matrix[i], "nvars": nvars} for i in range(nvars)]


def _rel_scores(row: np.ndarray, self_idx: int) -> np.ndarray:
    r = row.copy()
    r[self_idx] = 0.0
    total = r.sum()
    return r / total if total > 0 else r


def plot_all_cases(all_raw: dict, case_info: dict) -> plt.Figure:
    case_ids = sorted(all_raw.keys())
    ncases   = len(case_ids)
    nvars    = all_raw[case_ids[0]][0]["nvars"]

    fig   = plt.figure(figsize=(4.5 * ncases, 12.0 + 2.0 * nvars))
    top_m = 0.99; bot_m = 0.04
    split = bot_m + (top_m - bot_m) * (3.0 * nvars) / (15.0 + 3.0 * nvars)

    gs_diag = gridspec.GridSpec(1, ncases, figure=fig,
                                top=top_m, bottom=split - 0.35,
                                wspace=0.45, hspace=0)
    gs_bars = gridspec.GridSpec(nvars, ncases, figure=fig,
                                top=split, bottom=bot_m,
                                hspace=0.6, wspace=0.45)

    for c_idx, case_id in enumerate(case_ids):
        results      = all_raw[case_id]
        expected_set = _EXPECTED_CAUSAL.get(case_id, set())

        ax_diag = fig.add_subplot(gs_diag[0, c_idx])
        png_path = _DIAGRAM_PNG.get(case_id)
        if png_path and png_path.exists():
            ax_diag.imshow(mpimg.imread(str(png_path)))
        ax_diag.axis("off")

        for v_idx, res in enumerate(results):
            ax  = fig.add_subplot(gs_bars[v_idx, c_idx])
            row = res["pcmci_row"]
            rel = _rel_scores(row, v_idx)

            for j in range(nvars):
                if j == v_idx:
                    pass
                else:
                    hatch = _HATCH_PATTERN if (v_idx, j) in expected_set else ""
                    ax.bar(j, rel[j], color=_BAR_COLOR, edgecolor="black",
                           linewidth=1.5, hatch=hatch, zorder=2, width=0.8)

            ax.set_xlim(-0.5, nvars - 0.5)
            ax.set_ylim([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticks(range(nvars))
            ax.set_xticklabels([f"$Q_{j+1}$" for j in range(nvars)], fontsize=18)
            ax.tick_params(axis="y", labelsize=20)
            ax.set_ylabel("")
            if c_idx > 0:
                ax.set_yticklabels([])
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
            ax.tick_params(width=1.5)
            ax.set_title(
                f"$\\mathrm{{PCMCI}}_{{(\\cdot)\\rightarrow Q_{v_idx+1}^+}}$",
                fontsize=20, pad=16)

    return fig


def evaluate(results: list, case: int) -> dict:
    """
    Pass criteria (Q1 primary, spurious check all targets):
      Case 1 — Mediator:    Q2 dominates, Q3 pruned
      Case 2 — Confounder:  Q3 dominates, Q2 pruned
      Case 3 — Synergistic: Q2 and Q3 detected (CMI handles nonlinear interactions)
      Case 4 — Redundant:   Q2 and Q3 detected
    """
    _CRITERIA = {
        1: (1, "Q2 dominates PCMCI→Q1 (MCI prunes indirect Q3 path)"),
        2: (2, "Q3 dominates PCMCI→Q1 (MCI prunes spurious Q2 link)"),
        3: (None, "Both Q2 and Q3 detected via CMI-based PCMCI"),
        4: (None, "Both Q2 and Q3 detected (redundant information)"),
    }

    res   = results[0]
    row   = res["pcmci_row"]
    nvars = res["nvars"]

    sources    = [j for j in range(nvars) if j != 0]
    abs_scores = {j: float(row[j]) for j in sources}
    dominant_j = int(max(sources, key=lambda j: abs_scores[j]))
    dom_lbl    = f"Q{dominant_j + 1}"
    dom_abs    = abs_scores[dominant_j]

    rel       = _rel_scores(row, 0)
    dom_score = float(rel[dominant_j])

    expected_j, note = _CRITERIA.get(case, (None, "unknown"))

    if case == 3:
        passed = bool(abs_scores[1] > _ABS_THR and abs_scores[2] > _ABS_THR)
    elif case == 4:
        both_nz  = abs_scores[1] > _ABS_THR and abs_scores[2] > _ABS_THR
        max_val  = max(abs_scores[1], abs_scores[2])
        comparable = max_val > 0 and min(abs_scores[1], abs_scores[2]) / max_val > 0.5
        passed = bool(both_nz and comparable)
    elif expected_j is not None:
        passed = bool(dominant_j == expected_j and dom_score > 0.4)
    else:
        passed = None

    # Spurious check — all targets
    all_spurious = []
    for v_idx, res_v in enumerate(results):
        row_v = res_v["pcmci_row"]
        rel_v = _rel_scores(row_v, v_idx)
        expected_v = {j for (i, j) in _EXPECTED_CAUSAL.get(case, set()) if i == v_idx}
        for j in range(nvars):
            if j == v_idx:
                continue
            if (j not in expected_v
                    and rel_v[j] > _SPURIOUS_THR
                    and float(row_v[j]) > _ABS_THR):
                all_spurious.append(f"Q{j+1}→Q{v_idx+1}⁺")
    if all_spurious and passed is not None:
        passed = False

    return {
        "pass":      passed,
        "dominant":  dom_lbl,
        "score":     dom_score,
        "expected":  f"Q{expected_j + 1}" if expected_j is not None else "none",
        "note":      note,
        "spurious":  all_spurious,
        "all_scores": {f"Q{j+1}": float(rel[j]) for j in sources},
    }

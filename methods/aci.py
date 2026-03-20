"""
ACI — Assimilative Causal Inference

Definition of causality:
    Measures the causal influence of an unobserved variable (source) on an
    observed variable (target) as the KL divergence between the posterior
    distributions of the Bayesian filter p(y_t | x_{0:t}) and the Bayesian
    smoother p(y_t | x_{0:T}). High ACI indicates that future observations
    of the target carry information about the past state of the source.

Reference:
    Andreou, Chen & Bollt, Nat. Commun. 17, 1854 (2026).
    https://doi.org/10.1038/s41467-026-68568-0
"""
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _aci.aci_core import aci_pairwise  # noqa: E402

NAME       = "ACI"
DEFINITION = (
    "Measures causal influence via KL divergence between the Bayesian filter "
    "p(y_t|x_{0:t}) and smoother p(y_t|x_{0:T}) in a linearised CGNS framework."
)
REFERENCE  = (
    "Andreou, Chen & Bollt, Nat. Commun. 17, 1854 (2026). "
    "https://doi.org/10.1038/s41467-026-68568-0"
)

# Color per target variable (rows of ACI matrix)
_TARGET_COLORS = ["#4477AA", "#EE6677", "#228833"]


def run(X: np.ndarray, nbins: int = 50, nlag: int = 1) -> list:
    """
    Run ACI on multivariate time series X.

    Parameters
    ----------
    X : np.ndarray, shape (nvars, N)
    nbins : int  — unused (kept for interface compatibility)
    nlag  : int  — time lag

    Returns
    -------
    list of dict, one entry per target variable i:
        'aci_row'  : np.ndarray, shape (nvars,)  — mean ACI from each source j
        'aci_ts'   : dict {j: np.ndarray}        — full ACI time series
        'nvars'    : int
    """
    aci_matrix, aci_ts = aci_pairwise(X, nlag=nlag)
    nvars = X.shape[0]
    results = []
    for i in range(nvars):
        results.append({
            "aci_row": aci_matrix[i],   # shape (nvars,) — absolute mean ACI
            "aci_ts":  {j: aci_ts[(i, j)]
                        for j in range(nvars) if j != i},
            "nvars":   nvars,
        })
    return results


def _rel_scores(aci_row: np.ndarray, self_idx: int) -> np.ndarray:
    """Normalise ACI values (excluding self) to [0, 1] for plotting."""
    row = aci_row.copy()
    row[self_idx] = 0.0
    total = row.sum()
    if total <= 0:
        return row
    return row / total


def plot_all_cases(all_raw: dict, case_info: dict) -> plt.Figure:
    """
    Single-page figure with ACI results for all benchmark cases.

    Layout: nvars rows × ncases columns.
    Each cell: bar chart of normalised mean ACI from each source.
    """
    case_ids = sorted(all_raw.keys())
    ncases   = len(case_ids)
    nvars    = all_raw[case_ids[0]][0]["nvars"]
    var_names = [f"Q{i+1}" for i in range(nvars)]

    fig, axs = plt.subplots(
        nvars, ncases,
        figsize=(4.5 * ncases, 3.0 * nvars),
        gridspec_kw={"wspace": 0.45, "hspace": 0.75},
    )
    if ncases == 1:
        axs = axs[:, np.newaxis]

    for c_idx, case_id in enumerate(case_ids):
        results   = all_raw[case_id]
        case_name = case_info[case_id]["name"]

        for v_idx, res in enumerate(results):
            ax  = axs[v_idx, c_idx]
            row = res["aci_row"]   # shape (nvars,)
            rel = _rel_scores(row, v_idx)

            sources = [j for j in range(nvars) if j != v_idx]
            vals    = [rel[j] for j in sources]
            labels  = [f"$Q_{j+1}$" for j in sources]
            colors  = [_TARGET_COLORS[j % len(_TARGET_COLORS)] for j in sources]

            ax.bar(range(len(sources)), vals,
                   color=colors, edgecolor="black", linewidth=1.5)
            ax.set_ylim([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticks(range(len(sources)))
            ax.set_xticklabels(labels, fontsize=16)
            ax.tick_params(axis="y", labelsize=16)

            if c_idx == 0:
                ax.set_ylabel(
                    f"$\\mathrm{{ACI}}_{{(\\cdot)\\rightarrow Q_{v_idx+1}^+}}$",
                    fontsize=18,
                )
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])

            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
            ax.tick_params(width=1.5)

            # Titles
            if v_idx == 0:
                ax.set_title(
                    f"\\textbf{{Case {case_id}: {case_name}}}\n"
                    f"$\\mathrm{{ACI}}_{{(\\cdot)\\rightarrow Q_{v_idx+1}^+}}$",
                    fontsize=14, pad=8,
                )
            else:
                ax.set_title(
                    f"$\\mathrm{{ACI}}_{{(\\cdot)\\rightarrow Q_{v_idx+1}^+}}$",
                    fontsize=14, pad=8,
                )

    return fig


# ------------------------------------------------------------------
# Pass / fail criteria (confirmed after visual inspection)
# ------------------------------------------------------------------
_CRITERIA = {
    # (expected_dominant_source_idx_base0, threshold, note)
    # For Q1 (target 0): which source j should dominate?
    1: (1, 0.4, "Q2 dominates ACI→Q1 (direct driver in mediator chain)"),
    2: (2, 0.4, "Q3 dominates ACI→Q1 (common cause; Q2 link is spurious)"),
    3: (None, None, "No single source dominates (synergistic case)"),
    4: (1, 0.3, "Q2 (and Q3=Q2) both show high ACI→Q1 (redundant)"),
}


def evaluate(results: list, case: int) -> dict:
    """
    Evaluate ACI results for the Q1 target (results[0]).

    Pass criteria:
      Case 1 — Mediator:    Q2 absolute ACI >> Q3 (Q2 is direct driver)
      Case 2 — Confounder:  Q3 absolute ACI >> Q2 (Q3 is common cause)
      Case 3 — Synergistic: all absolute ACI < 1e-5 (no individual linear driver)
      Case 4 — Redundant:   both Q2 and Q3 have comparable nonzero ACI
    All confirmed visually.
    """
    res   = results[0]  # Q1 target
    row   = res["aci_row"]   # absolute mean ACI per source
    nvars = res["nvars"]

    sources      = [j for j in range(nvars) if j != 0]
    abs_scores   = {j: float(row[j]) for j in sources}
    dominant_j   = int(max(sources, key=lambda j: abs_scores[j]))
    dominant_lbl = f"Q{dominant_j + 1}"
    dom_abs      = abs_scores[dominant_j]

    # Relative scores for the score field (normalised for display)
    rel = _rel_scores(row, 0)
    dom_score = float(rel[dominant_j])

    expected_j, _, note = _CRITERIA.get(case, (None, None, "unknown"))

    if case == 3:
        # Synergistic: pass if all absolute ACI values are negligible
        passed = bool(dom_abs < 1e-5)
    elif case == 4:
        # Redundant: pass if both Q2 and Q3 show comparable nonzero ACI
        passed = bool(
            abs_scores[1] > 1e-7 and abs_scores[2] > 1e-7
            and min(abs_scores[1], abs_scores[2]) / max(abs_scores[1], abs_scores[2]) > 0.5
        )
    elif expected_j is not None:
        # Dominant source must match expected and relative score > 40%
        passed = bool(dominant_j == expected_j and dom_score > 0.4)
    else:
        passed = None

    all_scores = {f"Q{j+1}": float(rel[j]) for j in sources}

    return {
        "pass":      passed,
        "dominant":  dominant_lbl,
        "score":     dom_score,
        "expected":  f"Q{expected_j + 1}" if expected_j is not None else "none",
        "note":      note,
        "all_scores": all_scores,
    }

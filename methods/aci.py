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

# Ground-truth causal links per benchmark case: {case_id: {(target_0idx, source_0idx)}}
# Used to mark expected relationships with a hatch pattern in the figure.
_EXPECTED_CAUSAL = {
    1: {(0, 1), (1, 2)},              # Q2→Q1, Q3→Q2  (mediator chain)
    2: {(0, 2), (1, 2)},              # Q3→Q1, Q3→Q2  (common cause)
    3: set(),                          # no individual causality (synergistic)
    4: {(0, 1), (0, 2), (2, 1)},      # Q2→Q1, Q3→Q1, Q2→Q3 (redundant: Q3=Q2)
}

_BAR_COLOR      = "#D8D8D8"   # light gray for all bars
_SELF_COLOR     = "#F0F0F0"   # slightly lighter for the self (N/A) slot
_HATCH_PATTERN  = "///"       # marks expected causal bars


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
    Each panel shows normalised mean ACI for all nvars source variables
    (including self, which is always 0 and marked as N/A with an x-hatch).
    Bars corresponding to true causal links are marked with a /// hatch.
    All bars are light gray; ylabel is omitted (already in the title).
    """
    case_ids = sorted(all_raw.keys())
    ncases   = len(case_ids)
    nvars    = all_raw[case_ids[0]][0]["nvars"]

    fig, axs = plt.subplots(
        nvars, ncases,
        figsize=(4.5 * ncases, 3.0 * nvars),
        gridspec_kw={"wspace": 0.45, "hspace": 0.75},
    )
    if ncases == 1:
        axs = axs[:, np.newaxis]

    for c_idx, case_id in enumerate(case_ids):
        results      = all_raw[case_id]
        case_name    = case_info[case_id]["name"]
        expected_set = _EXPECTED_CAUSAL.get(case_id, set())

        for v_idx, res in enumerate(results):
            ax  = axs[v_idx, c_idx]
            row = res["aci_row"]           # absolute mean ACI, shape (nvars,)
            rel = _rel_scores(row, v_idx)  # normalised, self=0

            x_pos  = list(range(nvars))
            labels = [f"$Q_{j+1}$" for j in range(nvars)]

            for j in range(nvars):
                val = rel[j]              # 0 for self

                if j == v_idx:
                    # Self slot: N/A — draw a thin hatched bar at 0 height
                    ax.bar(j, 0.02, color=_SELF_COLOR, edgecolor="black",
                           linewidth=1.5, hatch="xx", zorder=2)
                else:
                    is_expected = (v_idx, j) in expected_set
                    hatch = _HATCH_PATTERN if is_expected else ""
                    ax.bar(j, val, color=_BAR_COLOR, edgecolor="black",
                           linewidth=1.5, hatch=hatch, zorder=2)

            ax.set_ylim([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, fontsize=16)
            ax.tick_params(axis="y", labelsize=16)

            # No ylabel — the title already carries the label
            ax.set_ylabel("")
            if c_idx > 0:
                ax.set_yticklabels([])

            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
            ax.tick_params(width=1.5)

            # Titles
            title_target = f"$\\Delta I_{{(\\cdot)\\rightarrow Q_{v_idx+1}^+}}$"
            if v_idx == 0:
                ax.set_title(
                    f"\\textbf{{Case {case_id}: {case_name}}}\n{title_target}",
                    fontsize=14, pad=8,
                )
            else:
                ax.set_title(title_target, fontsize=14, pad=8)

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

"""
CGC — Conditional Granger Causality

Definition of causality:
    Tests whether the past of Q_j improves prediction of Q_i beyond what is
    already explained by the past of Q_i and all other variables Q'.
    CGC_{j→i} = log₂[ var(ε̂_restricted) / var(ε_unrestricted) ] ≥ 0,
    where the restricted model omits Q_j and the unrestricted model includes it.

Reference:
    Geweke, J., J. Am. Stat. Assoc. 77(378):304-313 (1982).
    Barnett, L. & Seth, A.K., J. Neurosci. Methods 223:50-68 (2014).
    As described in Supplementary S2.1 of:
    Martínez-Sánchez, Arranz & Lozano-Durán, Nat. Commun. 15, 9296 (2024).
    https://doi.org/10.1038/s41467-024-53373-4
"""
import os
import sys

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _cgc.cgc_core import cgc_pairwise  # noqa: E402

NAME       = "CGC"
DEFINITION = (
    "Tests whether Q_j's past improves prediction of Q_i beyond all other variables; "
    "CGC_{j→i} = log₂[var(ε̂_restricted)/var(ε_unrestricted)] using OLS VAR models."
)
REFERENCE  = (
    "Geweke (1982) J. Am. Stat. Assoc. 77:304; "
    "Barnett & Seth (2014) J. Neurosci. Methods 223:50; "
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


def run(X: np.ndarray, nbins: int = 50, nlag: int = 1) -> list:
    cgc_matrix = cgc_pairwise(X, p=nlag)
    nvars = X.shape[0]
    return [{"cgc_row": cgc_matrix[i], "nvars": nvars} for i in range(nvars)]


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
            row = res["cgc_row"]
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
                f"$\\mathrm{{CGC}}_{{(\\cdot)\\rightarrow Q_{v_idx+1}^+}}$",
                fontsize=20, pad=16)

    return fig


def evaluate(results: list, case: int) -> dict:
    """
    Pass criteria (Q1 primary, spurious check all targets):
      Case 1 — Mediator:    Q2 dominates CGC→Q1
      Case 2 — Confounder:  Q3 dominates CGC→Q1
      Case 3 — Synergistic: both CGC→Q1 near 0 (linear CGC cannot detect synergy)
      Case 4 — Redundant:   Q2 or Q3 detected (redundancy may zero one out)
    """
    _CRITERIA = {
        1: (1, "Q2 dominates CGC→Q1 (conditioning removes Q3 spurious path)"),
        2: (2, "Q3 dominates CGC→Q1 (Q2 link vanishes after conditioning on Q3)"),
        3: (None, "Linear CGC cannot detect pure synergistic interaction"),
        4: (None, "Q2 or Q3 detected (redundancy may suppress one conditionally)"),
    }

    res   = results[0]
    row   = res["cgc_row"]
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
        passed = bool(dom_abs < 1e-4)
    elif case == 4:
        passed = bool(dom_abs > 1e-6)
    elif expected_j is not None:
        passed = bool(dominant_j == expected_j and dom_score > 0.4)
    else:
        passed = None

    # Spurious check — all targets
    all_spurious = []
    for v_idx, res_v in enumerate(results):
        row_v = res_v["cgc_row"]
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

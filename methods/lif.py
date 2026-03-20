"""
LIF — Liang's Information Flow (multivariate)

Definition of causality:
    T_{j→i} measures the rate of Shannon entropy transfer from X_j to X_i,
    conditioned on all other observed variables.
    Positive T_{j→i} means X_j causally influences X_i.
    Multivariate formula (Liang 2021):
      T_{j→i} = (C_{ij}/C_{ii}) * (1/det(C)) * sum_k [ Delta_{jk} * cov(X_k, dX_i/dt) ]
    where Delta is the cofactor matrix of the full covariance matrix C.
    Reduces exactly to the bivariate formula (Liang 2014) when nvars=2.

Reference:
    Liang, X.S., Phys. Rev. E 90:052150 (2014).
    Liang, X.S., Entropy 23(6):679 (2021).
    Liang, X.S. & Kleeman, R., Phys. Rev. Lett. 95:244101 (2005).
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
from _lif.lif_core import lif_pairwise  # noqa: E402

NAME       = "LIF"
DEFINITION = (
    "T_{j→i} = rate of Shannon entropy transfer from X_j to X_i, conditioned on all "
    "other variables via the cofactor matrix of C (multivariate Liang 2021 formula)."
)
REFERENCE  = (
    "Liang (2014) Phys. Rev. E 90:052150; "
    "Liang (2021) Entropy 23:679; "
    "Liang & Kleeman (2005) Phys. Rev. Lett. 95:244101."
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
_ABS_THR       = 1e-4   # LIF noise floor; values below this are estimation noise


def run(X: np.ndarray, nbins: int = 50, nlag: int = 1) -> list:
    """
    Run LIF on multivariate time series X.

    Parameters
    ----------
    X     : np.ndarray, shape (nvars, N)
    nbins : int  — unused (kept for interface compatibility)
    nlag  : int  — time lag for forward differences

    Returns
    -------
    list of dict per target variable i:
        'lif_row' : np.ndarray (nvars,) — raw T_{j→i} values (may be negative)
        'nvars'   : int
    """
    lif_matrix = lif_pairwise(X, nlag=nlag)
    nvars = X.shape[0]
    return [{"lif_row": lif_matrix[i], "nvars": nvars} for i in range(nvars)]


def _rel_scores(row: np.ndarray, self_idx: int) -> np.ndarray:
    """Normalise clamped-at-zero row to relative scores summing to 1."""
    r = np.maximum(row, 0.0)
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
            row = res["lif_row"]
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
                f"$\\mathrm{{LIF}}_{{(\\cdot)\\rightarrow Q_{v_idx+1}^+}}$",
                fontsize=20, pad=16)

    return fig


def evaluate(results: list, case: int) -> dict:
    """
    Pass criteria (Q1 primary, spurious check all targets):
      Case 1 — Mediator:    Q2 dominates LIF→Q1 (direct driver)
      Case 2 — Confounder:  Q3 dominates LIF→Q1 (pairwise LIF may detect spurious Q2)
      Case 3 — Synergistic: both Q2 and Q3 detected (pairwise may partially detect)
      Case 4 — Redundant:   both Q2 and Q3 detected
    """
    _CRITERIA = {
        1: (1, "Q2 dominates LIF→Q1 (direct driver in mediator chain)"),
        2: (2, "Q3 dominates LIF→Q1 (common cause; pairwise LIF may flag spurious Q2)"),
        3: (None, "Both Q2 and Q3 show positive LIF→Q1 (synergistic case)"),
        4: (None, "Both Q2 and Q3 show comparable positive LIF→Q1 (redundant)"),
    }

    res   = results[0]
    row   = res["lif_row"]
    nvars = res["nvars"]

    sources    = [j for j in range(nvars) if j != 0]
    abs_scores = {j: float(max(row[j], 0.0)) for j in sources}
    dominant_j = int(max(sources, key=lambda j: abs_scores[j]))
    dom_lbl    = f"Q{dominant_j + 1}"

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
        row_v = res_v["lif_row"]
        rel_v = _rel_scores(row_v, v_idx)
        expected_v = {j for (i, j) in _EXPECTED_CAUSAL.get(case, set()) if i == v_idx}
        for j in range(nvars):
            if j == v_idx:
                continue
            if (j not in expected_v
                    and rel_v[j] > _SPURIOUS_THR
                    and float(max(row_v[j], 0.0)) > _ABS_THR):
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

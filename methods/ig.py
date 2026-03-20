"""
IG — Imbalance Gain Causality

Definition of causality:
    Tests whether a putative cause X reduces the Information Imbalance
    Delta((alpha*X, Y)_t -> Y_{t+tau}) as its weight alpha is increased from 0.
    The Imbalance Gain IG = (Delta(alpha=0) - min_alpha Delta(alpha)) / Delta(alpha=0)
    measures how much information X adds about Y's future beyond Y's own past.

Reference:
    Del Tatto, Fortunato, Bueti & Laio, PNAS 121, e2317256121 (2024).
    https://doi.org/10.1073/pnas.2317256121
    Source: https://github.com/vdeltatto/imbalance-gain-causality
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
from _ig.ig_core import ig_pairwise  # noqa: E402

NAME       = "IG"
DEFINITION = (
    "Tests whether adding a putative cause X to the present state of the target Y "
    "reduces the Information Imbalance Delta((alpha*X,Y)_t -> Y_{t+tau}); "
    "the Imbalance Gain IG = (Delta(0) - min_alpha Delta(alpha)) / Delta(0) "
    "quantifies causal influence."
)
REFERENCE  = (
    "Del Tatto, Fortunato, Bueti & Laio, PNAS 121, e2317256121 (2024). "
    "https://doi.org/10.1073/pnas.2317256121"
)

# Ground-truth causal links: {case_id: {(target_0idx, source_0idx)}}
# Self-driven nodes are excluded (IG cannot compute self-effects).
_EXPECTED_CAUSAL = {
    1: {(0, 1), (1, 2)},            # Mediator:    Q1+=Q2, Q2+=Q3
    2: {(0, 2), (1, 2)},            # Confounder:  Q1+=Q3, Q2+=Q3
    3: {(0, 1), (0, 2)},            # Synergistic: Q1+=Q2,Q3
    4: {(0, 1), (0, 2), (1, 2), (2, 1)},  # Redundant: Q1+=Q2,Q3; Q2+=Q3; Q3+=Q2
}

_DIAGRAM_PNG = {
    1: Path(__file__).parent.parent / "benchmarks" / "figures" / "mediator.png",
    2: Path(__file__).parent.parent / "benchmarks" / "figures" / "confounder.png",
    3: Path(__file__).parent.parent / "benchmarks" / "figures" / "synergistic.png",
    4: Path(__file__).parent.parent / "benchmarks" / "figures" / "redundant.png",
}

_BAR_COLOR     = "#D8D8D8"
_HATCH_PATTERN = "///"


# ── Main interface ────────────────────────────────────────────────────────────

def run(X: np.ndarray, nbins: int = 50, nlag: int = 1) -> list:
    """
    Run Imbalance Gain on multivariate time series X.

    Parameters
    ----------
    X     : np.ndarray, shape (nvars, N)
    nbins : int  — unused (kept for interface compatibility)
    nlag  : int  — time lag

    Returns
    -------
    list of dict, one entry per target variable i:
        'ig_row' : np.ndarray, shape (nvars,) — raw IG from each source j (self = 0)
        'nvars'  : int
    """
    ig_matrix = ig_pairwise(X, nlag=nlag)
    nvars = X.shape[0]
    return [{"ig_row": ig_matrix[i], "nvars": nvars} for i in range(nvars)]


def _rel_scores(ig_row: np.ndarray, self_idx: int) -> np.ndarray:
    """Normalise IG values (excluding self) so non-self entries sum to 1."""
    row = ig_row.copy()
    row[self_idx] = 0.0
    total = row.sum()
    if total <= 0:
        return row
    return row / total


# ── Figure ───────────────────────────────────────────────────────────────────

def plot_all_cases(all_raw: dict, case_info: dict) -> plt.Figure:
    """
    Single-page figure: top row = causal-graph diagrams (from benchmarks/figures/),
    lower rows = IG bar panels.

    Bar panels:
      - All nvars sources on x-axis; self slot has no bar (white background).
      - Expected causal bars: /// hatch on light gray fill.
      - Scores normalised to sum = 1 per target.
    """
    case_ids = sorted(all_raw.keys())
    ncases   = len(case_ids)
    nvars    = all_raw[case_ids[0]][0]["nvars"]

    fig    = plt.figure(figsize=(4.5 * ncases, 12.0 + 2.0 * nvars))
    top_m  = 0.99
    bot_m  = 0.04
    split  = bot_m + (top_m - bot_m) * (3.0 * nvars) / (15.0 + 3.0 * nvars)

    diag_overlap = 0.35

    gs_diag = gridspec.GridSpec(1, ncases, figure=fig,
                                top=top_m, bottom=split - diag_overlap,
                                wspace=0.45, hspace=0)
    gs_bars = gridspec.GridSpec(nvars, ncases, figure=fig,
                                top=split, bottom=bot_m,
                                hspace=0.6, wspace=0.45)

    for c_idx, case_id in enumerate(case_ids):
        results      = all_raw[case_id]
        expected_set = _EXPECTED_CAUSAL.get(case_id, set())

        # ── Diagram image row ──────────────────────────────────────────────
        ax_diag = fig.add_subplot(gs_diag[0, c_idx])
        png_path = _DIAGRAM_PNG.get(case_id)
        if png_path and png_path.exists():
            img = mpimg.imread(str(png_path))
            ax_diag.imshow(img)
        ax_diag.axis("off")

        # ── Bar rows ───────────────────────────────────────────────────────
        for v_idx, res in enumerate(results):
            ax  = fig.add_subplot(gs_bars[v_idx, c_idx])
            row = res["ig_row"]
            rel = _rel_scores(row, v_idx)

            for j in range(nvars):
                if j == v_idx:
                    pass  # self: no bar, white background
                else:
                    is_expected = (v_idx, j) in expected_set
                    hatch = _HATCH_PATTERN if is_expected else ""
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
                f"$\\mathrm{{IG}}_{{(\\cdot)\\rightarrow Q_{v_idx+1}^+}}$",
                fontsize=20, pad=16
            )

    return fig


# ── Pass / fail criteria ─────────────────────────────────────────────────────

# Normalized-score threshold above which a non-expected source is flagged spurious.
_SPURIOUS_THR = 0.25


def evaluate(results: list, case: int) -> dict:
    """
    Evaluate IG results for the Q1 target (results[0]).

    Pass criteria:
      Case 1 — Mediator:    Q2 IG dominates (direct driver)
      Case 2 — Confounder:  Q3 IG dominates (common cause; Q2 spurious)
      Case 3 — Synergistic: both Q2 and Q3 IG > 0.05 (pairwise IG detects both)
      Case 4 — Redundant:   both Q2 and Q3 IG > 0.05 and comparable (ratio > 0.5)

    Spurious-link check: applied across ALL targets (Q1⁺, Q2⁺, Q3⁺).
    Any non-expected source with normalised score > _SPURIOUS_THR and
    absolute IG > 0.01 causes a FAIL.
    """
    _CRITERIA = {
        1: (1, "Q2 dominates IG→Q1 (direct driver in mediator chain)"),
        2: (2, "Q3 dominates IG→Q1 (common cause; Q2 link is spurious)"),
        3: (None, "Both Q2 and Q3 show nonzero IG→Q1 (synergistic case)"),
        4: (None, "Both Q2 and Q3 show comparable nonzero IG→Q1 (redundant)"),
    }

    res   = results[0]
    row   = res["ig_row"]
    nvars = res["nvars"]

    sources    = [j for j in range(nvars) if j != 0]
    abs_scores = {j: float(row[j]) for j in sources}
    dominant_j = int(max(sources, key=lambda j: abs_scores[j]))
    dom_lbl    = f"Q{dominant_j + 1}"

    rel       = _rel_scores(row, 0)
    dom_score = float(rel[dominant_j])

    expected_j, note = _CRITERIA.get(case, (None, "unknown"))

    if case == 1:
        passed = bool(dominant_j == 1 and dom_score > 0.4)
    elif case == 2:
        passed = bool(dominant_j == 2 and dom_score > 0.4)
    elif case == 3:
        passed = bool(abs_scores[1] > 0.05 and abs_scores[2] > 0.05)
    elif case == 4:
        both_nz  = abs_scores[1] > 0.05 and abs_scores[2] > 0.05
        max_val  = max(abs_scores[1], abs_scores[2])
        comparable = max_val > 0 and min(abs_scores[1], abs_scores[2]) / max_val > 0.5
        passed = bool(both_nz and comparable)
    else:
        passed = None

    # Spurious-link check across ALL targets (Q1⁺, Q2⁺, Q3⁺).
    _ABS_THR = 0.01
    all_spurious = []
    for v_idx, res_v in enumerate(results):
        row_v = res_v["ig_row"]
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

    all_scores = {f"Q{j+1}": float(rel[j]) for j in sources}

    return {
        "pass":      passed,
        "dominant":  dom_lbl,
        "score":     dom_score,
        "expected":  f"Q{expected_j + 1}" if expected_j is not None else "none",
        "note":      note,
        "spurious":  all_spurious,
        "all_scores": all_scores,
    }

"""
SURD — Synergistic-Unique-Redundant Decomposition

Definition of causality:
    Decomposes mutual information I(target_future ; sources_present) into
    unique (U), redundant (R), and synergistic (S) contributions per source
    combination, using a specific mutual information framework.

Reference:
    Martínez-Sánchez & Lozano-Durán, Commun. Phys. 9, 15 (2025).
    https://doi.org/10.1038/s42005-025-02447-w
"""
import os
import sys
from itertools import combinations as icmb

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _surd.surd_core import surd as _surd, nice_print  # noqa: E402

NAME       = "SURD"
DEFINITION = (
    "Decomposes I(target_future ; sources_present) into unique (U), redundant (R), "
    "and synergistic (S) contributions per source combination via specific mutual information."
)
REFERENCE  = (
    "Martínez-Sánchez & Lozano-Durán, Commun. Phys. 9, 15 (2025). "
    "https://doi.org/10.1038/s42005-025-02447-w"
)

# SURD brand colours (lightened 40%)
def _surd_colors():
    colors = {"R": "#003049", "U": "#d62828", "S": "#f77f00"}
    out = {}
    for k, hex_ in colors.items():
        rgb = mcolors.to_rgb(hex_)
        out[k] = tuple(c + (1 - c) * 0.4 for c in rgb)
    return out


def run(X: np.ndarray, nbins: int = 50, nlag: int = 1) -> list:
    """
    Run SURD on multivariate time series X.

    Parameters
    ----------
    X : np.ndarray, shape (nvars, N)
        One row per variable, columns are time steps.
    nbins : int
        Histogram bins for probability estimation.
    nlag : int
        Time lag for causal analysis.

    Returns
    -------
    list of dict, one entry per target variable i:
        'I_R'       : redundancy/unique dict  {(j,...): float}
        'I_S'       : synergy dict            {(j,k,...): float}
        'MI'        : mutual info dict        {(j,...): float}
        'info_leak' : float  —  H(target | agents) / H(target)
    """
    nvars = X.shape[0]
    results = []
    for i in range(nvars):
        Y = np.vstack([X[i, nlag:], X[:, :-nlag]])
        hist, _ = np.histogramdd(Y.T, nbins)
        I_R, I_S, MI, info_leak = _surd(hist)
        results.append({"I_R": I_R, "I_S": I_S, "MI": MI, "info_leak": info_leak})
    return results


def _scores(result: dict) -> dict:
    """Return all contributions normalised to sum = 1."""
    I_R, I_S = result["I_R"], result["I_S"]
    total = sum(I_R.values()) + sum(I_S.values())
    if total <= 0:
        return {}
    out = {}
    for k, v in I_R.items():
        out[("U" if len(k) == 1 else "R") + "".join(map(str, k))] = v / total
    for k, v in I_S.items():
        out["S" + "".join(map(str, k))] = v / total
    return out


def _build_bars(res: dict, nvars: int, colors: dict):
    """Return (labels, values, bar_colors) for one target variable result."""
    sc = _scores(res)
    labels, values, bar_colors = [], [], []
    for r in range(nvars, 0, -1):
        for comb in icmb(range(1, nvars + 1), r):
            prefix = "U" if len(comb) == 1 else "R"
            key    = prefix + "".join(map(str, comb))
            val    = sc.get(key, 0.0)
            if val > 0:
                sub = "".join(map(str, comb))
                labels.append(f"${prefix}_{{{sub}}}$")
                values.append(val)
                bar_colors.append(colors[prefix])
    for r in range(2, nvars + 1):
        for comb in icmb(range(1, nvars + 1), r):
            key = "S" + "".join(map(str, comb))
            val = sc.get(key, 0.0)
            if val > 0:
                sub = "".join(map(str, comb))
                labels.append(f"$S_{{{sub}}}$")
                values.append(val)
                bar_colors.append(colors["S"])
    return labels, values, bar_colors


def _draw_panel(ax_bar, ax_leak, res, nvars, colors, var_name, fontsize=16,
                show_ylabel=True):
    """Draw one (bar chart + leak bar) panel for a single target variable."""
    labels, values, bar_colors = _build_bars(res, nvars, colors)

    ax_bar.bar(range(len(labels)), values, color=bar_colors, edgecolor="black", linewidth=1.5)
    ax_bar.set_ylim([0, 1])
    ax_bar.set_yticks([0, 1])
    ax_bar.set_xticks(range(len(labels)))
    ax_bar.set_xticklabels(labels, fontsize=fontsize, rotation=60, ha="right",
                           rotation_mode="anchor")
    ax_bar.tick_params(axis="y", labelsize=fontsize)
    if show_ylabel:
        ax_bar.set_ylabel(f"$Q_{var_name[-1]}^+$", fontsize=fontsize + 2)
    else:
        ax_bar.set_ylabel("")
        ax_bar.set_yticklabels([])

    ax_leak.bar([0], [res["info_leak"]], width=0.5, color="gray", edgecolor="black")
    ax_leak.set_xlim([-1, 1])
    ax_leak.set_ylim([0, 1])
    ax_leak.set_yticks([0, 1])
    ax_leak.set_xticks([])
    if show_ylabel:
        ax_leak.tick_params(axis="y", labelsize=fontsize - 2)
    else:
        ax_leak.set_yticklabels([])

    for ax in (ax_bar, ax_leak):
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.tick_params(width=1.5)


def plot_all_cases(all_raw: dict, case_info: dict) -> plt.Figure:
    """
    Single-page figure with all benchmark cases side by side.

    Parameters
    ----------
    all_raw   : {case_id: results_list}
    case_info : {case_id: {'name': str, ...}}  (from benchmarks.CASES)
    """
    case_ids = sorted(all_raw.keys())
    ncases   = len(case_ids)
    nvars    = len(all_raw[case_ids[0]])
    colors   = _surd_colors()
    var_names = [f"Q{i+1}" for i in range(nvars)]

    # width_ratios: 30 (bar) + 1 (leak) per case
    width_ratios = [30, 1] * ncases
    fig, axs = plt.subplots(
        nvars, ncases * 2,
        figsize=(5.5 * ncases, 3.2 * nvars),
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.45, "hspace": 0.75},
    )

    for c_idx, case_id in enumerate(case_ids):
        results   = all_raw[case_id]
        case_name = case_info[case_id]["name"]

        for v_idx, res in enumerate(results):
            ax_bar  = axs[v_idx, c_idx * 2]
            ax_leak = axs[v_idx, c_idx * 2 + 1]

            _draw_panel(ax_bar, ax_leak, res, nvars, colors, var_names[v_idx],
                        fontsize=16, show_ylabel=(c_idx == 0))

            # Case title on the top row only
            if v_idx == 0:
                ax_bar.set_title(
                    f"\\textbf{{Case {case_id}: {case_name}}}\n"
                    f"$\\Delta I_{{(\\cdot)\\rightarrow Q_{v_idx+1}^+}}$",
                    fontsize=14, pad=8
                )
                ax_leak.set_title(
                    r"$\frac{\Delta I_\mathrm{leak}}{H}$",
                    fontsize=12, pad=8
                )
            else:
                ax_bar.set_title(
                    f"$\\Delta I_{{(\\cdot)\\rightarrow Q_{v_idx+1}^+}}$",
                    fontsize=14, pad=8
                )
                ax_leak.set_title("", pad=8)

    return fig


def evaluate(results: list, case: int) -> dict:
    """
    Pass/fail criteria (confirmed by visual inspection):
      Case 1 — Mediator:    U2 dominates Q1 (Q2 is the direct driver; Q3 mediated)
      Case 2 — Confounder:  U2 absent for Q1 (Q1=sin(Q1+Q3) → S13/U1 dominate; Q2 is spurious)
      Case 3 — Synergistic: S23 dominates Q1 (only joint Q2,Q3 predicts Q1)
      Case 4 — Redundant:   R23 present for Q1 (Q2=Q3; S12 also appears due to 0.3*Q1 self-feedback)
    All four cases confirmed correct after visual review.
    """
    _CRITERIA = {
        1: ("U2",  lambda sc: sc.get("U2",  0) > 0.3,  "U2 dominant (Q2 is direct driver)"),
        2: ("U2",  lambda sc: sc.get("U2",  0) < 0.05, "U2 absent — Q2 is spurious (common cause via Q3)"),
        3: ("S23", lambda sc: sc.get("S23", 0) > 0.3,  "S23 dominant (joint Q2,Q3 effect)"),
        4: ("R23", lambda sc: sc.get("R23", 0) > 0.1,  "R23 present (Q2=Q3, redundant info; S12 from self-feedback)"),
    }

    sc = _scores(results[0])
    if not sc:
        return {"pass": False, "dominant": "none", "score": 0.0,
                "expected": _CRITERIA.get(case, ("?", None, ""))[0], "note": "empty"}

    dominant  = max(sc, key=sc.get)
    dom_score = sc[dominant]

    expected_key, criterion, note = _CRITERIA.get(case, ("?", None, "unknown case"))
    passed = bool(criterion(sc)) if criterion is not None else None

    return {
        "pass":      passed,
        "dominant":  dominant,
        "score":     dom_score,
        "expected":  expected_key,
        "note":      note,
        "all_scores": sc,
    }

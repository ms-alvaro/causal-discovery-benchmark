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
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Circle, Arc
import matplotlib.image as mpimg
import numpy as np
from pathlib import Path

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

# ── Causal-graph metadata ───────────────────────────────────────────────────

# Ground-truth causal links: {case_id: {(target_0idx, source_0idx)}}
# Used both for hatch marking on bars and for drawing diagram arrows.
_EXPECTED_CAUSAL = {
    # Non-self expected causal sources per (target, source) — 0-indexed.
    # Self-driven nodes (AR/self-feedback) are omitted since ACI cannot compute self-effects.
    # Mediator  Q1+=Q2, Q2+=Q3, Q3+=Q3(self only→no ext.)
    1: {(0, 1), (1, 2)},
    # Confounder  Q1+=Q1(self)+Q3, Q2+=Q2(self)+Q3, Q3+=Q3(self only)
    2: {(0, 2), (1, 2)},
    # Synergistic  Q1+=Q2+Q3, Q2+=Q2(self only), Q3+=Q3(self only)
    3: {(0, 1), (0, 2)},
    # Redundant  Q1+=Q1(self)+Q2+Q3, Q2+=Q2(self)+Q3, Q3+=Q2+Q3(self)
    4: {(0, 1), (0, 2), (1, 2), (2, 1)},
}

# Edges shown in the diagram (Case 3 shows arrows even though they're synergistic)
_DIAGRAM_EDGES = {
    1: [(0, 1), (1, 2)],
    2: [(0, 2), (1, 2)],
    3: [(0, 1), (0, 2)],              # drawn in red to indicate synergy
    4: [(0, 1), (0, 2), (2, 1)],
}

# Which (tgt, src) edges are synergistic (drawn red) or equality (dashed)
_SYN_EDGES = {3: {(0, 1), (0, 2)}}
_EQ_EDGES  = {4: {(2, 1)}}           # Q3 = Q2

# Which nodes have a self-loop drawn (variables with self-feedback)
_SELF_LOOPS = {
    1: {2},          # Q3 is AR(1)
    2: {0, 1, 2},    # all three have self-feedback
    3: {1, 2},       # Q2 and Q3 are AR(1)
    4: {0, 1},       # Q1 (0.3·Q1 term) and Q2 (AR(1))
}

# ── Diagram geometry ────────────────────────────────────────────────────────
# Node positions in the [0,1]² diagram canvas (same layout for all cases)
_POS = {
    0: np.array([0.79, 0.52]),   # Q1  right
    1: np.array([0.34, 0.32]),   # Q2  lower-left
    2: np.array([0.34, 0.73]),   # Q3  upper-left
}
_R   = 0.12      # node radius in data units (axes xlim/ylim = [0,1])
_NODE_FC = "#C8DDB8"   # light sage-green fill (matches reference image)
_NODE_EC = "#3D5A3E"   # dark green edge

# ── Bar chart style ─────────────────────────────────────────────────────────
_BAR_COLOR     = "#D8D8D8"   # light gray for all bars
_HATCH_PATTERN = "///"        # marks expected causal bars


# ── Diagram drawing helpers ─────────────────────────────────────────────────

def _wavy_arrow(ax, src, dst, n_waves=3, amp=0.032, color="black", lw=1.8):
    """Draw a wiggly (sine-wave) arrow from src to dst."""
    x0, y0 = src
    x1, y1 = dst
    dx, dy  = x1 - x0, y1 - y0
    L       = np.hypot(dx, dy)
    ux, uy  = dx / L, dy / L   # unit tangent
    nx, ny  = -uy, ux           # unit normal

    t_body  = np.linspace(0, 0.78, 200)
    wave    = amp * np.sin(n_waves * 2 * np.pi * t_body)
    xs = x0 + t_body * dx + wave * nx
    ys = y0 + t_body * dy + wave * ny
    ax.plot(xs, ys, color=color, lw=lw, zorder=3, solid_capstyle="round")
    ax.annotate(
        "", xy=(x1, y1), xytext=(xs[-1], ys[-1]),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=14),
        annotation_clip=False,
    )


def _straight_arrow(ax, p0, p1, color="black", lw=2.0, rad=0.0):
    """Straight (or gently curved) arrow between two node-boundary points."""
    patch = FancyArrowPatch(
        p0, p1,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        color=color, lw=lw,
        mutation_scale=18,
        zorder=3,
    )
    ax.add_patch(patch)


def _self_loop(ax, center, direction="top", color="black", lw=2.0):
    """Draw a self-loop arc above (or beside) a node."""
    cx, cy = center
    r  = _R
    lr = r * 0.72       # loop circle radius
    # Loop centre above/beside the node
    if direction == "top":
        lc = np.array([cx, cy + r + lr * 0.65])
    else:
        lc = np.array([cx - r - lr * 0.65, cy])

    # Draw arc (nearly full circle, gap at the side facing the node)
    theta = np.linspace(np.radians(30), np.radians(330), 300)
    if direction == "top":
        theta = np.linspace(np.radians(30), np.radians(330), 300)
    else:
        theta = np.linspace(np.radians(120), np.radians(420), 300)

    xs = lc[0] + lr * np.cos(theta)
    ys = lc[1] + lr * np.sin(theta)
    ax.plot(xs, ys, color=color, lw=lw, zorder=3, solid_capstyle="round")

    # Arrowhead at the end of the arc
    ax.annotate(
        "", xy=(xs[-1], ys[-1]), xytext=(xs[-3], ys[-3]),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=14),
        annotation_clip=False,
    )


def _boundary_point(center, toward, r=_R):
    """Return the point on a node's circular boundary facing 'toward'."""
    d = np.array(toward) - np.array(center)
    d = d / np.linalg.norm(d)
    return np.array(center) + r * d


def _draw_causal_graph(ax, case_id: int, case_name: str, fontsize: int = 13):
    """
    Draw the causal-graph diagram for one benchmark case.

    Style mirrors the reference image: light-green circular nodes, wiggly
    noise-input arrows (W1/W2/W3), self-loops for AR(1) variables, and
    straight/curved arrows for causal links.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    pos       = _POS
    r         = _R
    syn_edges = _SYN_EDGES.get(case_id, set())
    eq_edges  = _EQ_EDGES.get(case_id, set())
    self_loop_nodes = _SELF_LOOPS.get(case_id, set())

    # ── Noise input sources (outside the canvas, approx.) ──────────────────
    # W3 enters Q3 from the left; W2 enters Q2 from left; W1 enters Q1 from below
    noise_srcs = {
        2: (np.array([0.00, pos[2][1]]), "$W_3$", "left"),   # Q3
        1: (np.array([0.00, pos[1][1]]), "$W_2$", "left"),   # Q2
        0: (np.array([pos[0][0], 0.00]), "$W_1$", "below"),  # Q1
    }

    for node_idx, (nsrc, wlabel, side) in noise_srcs.items():
        ndst = _boundary_point(pos[node_idx], nsrc)   # boundary facing noise
        # Wavy arrow toward the node from noise source
        _wavy_arrow(ax, nsrc, ndst, n_waves=3, amp=0.028, color="black", lw=1.6)
        # Label
        if side == "left":
            ax.text(nsrc[0] - 0.03, nsrc[1], wlabel,
                    ha="right", va="center", fontsize=fontsize)
        else:
            ax.text(nsrc[0], nsrc[1] - 0.05, wlabel,
                    ha="center", va="top", fontsize=fontsize)

    # ── Self-loops ──────────────────────────────────────────────────────────
    for ni in self_loop_nodes:
        # Choose loop direction based on available space
        direction = "top" if ni in (1, 2) else "top"
        _self_loop(ax, pos[ni], direction=direction, color="black", lw=1.8)

    # ── Causal arrows ───────────────────────────────────────────────────────
    for (tgt, src) in _DIAGRAM_EDGES.get(case_id, []):
        is_syn = (tgt, src) in syn_edges
        is_eq  = (tgt, src) in eq_edges
        color  = "#C0392B" if is_syn else "black"
        rad    = 0.15 if abs(tgt - src) == 1 else 0.0   # slight curve between adjacent nodes

        p0 = _boundary_point(pos[src], pos[tgt])
        p1 = _boundary_point(pos[tgt], pos[src])

        if is_eq:
            # Dashed arrow for Q2→Q3 (equality)
            patch = FancyArrowPatch(
                p0, p1,
                connectionstyle="arc3,rad=0.35",
                arrowstyle="-|>",
                color="black", lw=1.8,
                mutation_scale=16,
                linestyle="dashed",
                zorder=3,
            )
            ax.add_patch(patch)
            # "=" annotation
            mid = (np.array(p0) + np.array(p1)) / 2
            ax.text(mid[0] - 0.10, mid[1], "$=$",
                    ha="center", va="center", fontsize=fontsize,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"),
                    zorder=5)
        else:
            _straight_arrow(ax, p0, p1, color=color, lw=1.8, rad=rad)

    # ── Nodes (drawn last so they cover arrow tails) ─────────────────────────
    for idx, p in pos.items():
        circle = Circle(p, r, facecolor=_NODE_FC, edgecolor=_NODE_EC, lw=2.2, zorder=4)
        ax.add_patch(circle)
        ax.text(p[0], p[1], f"$Q_{idx+1}$",
                ha="center", va="center", fontsize=fontsize + 1, zorder=5)

    # ── Case label ──────────────────────────────────────────────────────────
    ax.text(0.5, -0.04, case_name.lower(),
            ha="center", va="top", fontsize=fontsize,
            transform=ax.transAxes)


# ── Main interface ──────────────────────────────────────────────────────────

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


_DIAGRAM_PNG = {
    1: Path(__file__).parent.parent / "benchmarks" / "figures" / "mediator.png",
    2: Path(__file__).parent.parent / "benchmarks" / "figures" / "confounder.png",
    3: Path(__file__).parent.parent / "benchmarks" / "figures" / "synergistic.png",
    4: Path(__file__).parent.parent / "benchmarks" / "figures" / "redundant.png",
}


def plot_all_cases(all_raw: dict, case_info: dict) -> plt.Figure:
    """
    Single-page figure: top row = causal-graph diagrams (from benchmarks/figures/),
    lower rows = ACI bar panels.

    Bar panels:
      - All nvars sources on x-axis; self slot = exactly 0 (gray background, no bar).
      - Expected causal bars: /// hatch on light gray fill.
      - No ylabel (already in the title).
    """
    case_ids  = sorted(all_raw.keys())
    ncases    = len(case_ids)
    nvars     = all_raw[case_ids[0]][0]["nvars"]

    # Two separate GridSpecs so diagram row and bar rows can have independent spacing.
    # The diagram bottom == bar top → zero gap between them; bars keep their own hspace.
    fig    = plt.figure(figsize=(4.5 * ncases, 10.0 + 2.0 * nvars))
    top_m  = 0.99
    bot_m  = 0.04
    # Split point: fraction of figure height where diagrams end / bars begin
    # Diagram units = 11.0, bar units = 3.0 * nvars → diagrams get more height
    split  = bot_m + (top_m - bot_m) * (3.0 * nvars) / (11.0 + 3.0 * nvars)

    gs_diag = gridspec.GridSpec(1, ncases, figure=fig,
                                top=top_m, bottom=split,
                                wspace=0.45, hspace=0)
    gs_bars = gridspec.GridSpec(nvars, ncases, figure=fig,
                                top=split, bottom=bot_m,
                                hspace=0.4, wspace=0.45)

    for c_idx, case_id in enumerate(case_ids):
        results      = all_raw[case_id]
        case_name    = case_info[case_id]["name"]
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
            row = res["aci_row"]           # absolute mean ACI, shape (nvars,)
            rel = _rel_scores(row, v_idx)  # normalised, self = 0

            for j in range(nvars):
                if j == v_idx:
                    pass  # self slot: no bar, white background (height exactly 0)
                else:
                    is_expected = (v_idx, j) in expected_set
                    hatch = _HATCH_PATTERN if is_expected else ""
                    ax.bar(j, rel[j], color=_BAR_COLOR, edgecolor="black",
                           linewidth=1.5, hatch=hatch, zorder=2, width=0.8)

            ax.set_xlim(-0.5, nvars - 0.5)
            ax.set_ylim([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticks(range(nvars))
            ax.set_xticklabels([f"$Q_{j+1}$" for j in range(nvars)], fontsize=20)
            ax.tick_params(axis="y", labelsize=20)

            ax.set_ylabel("")
            if c_idx > 0:
                ax.set_yticklabels([])

            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
            ax.tick_params(width=1.5)

            title_lbl = f"$\\Delta I_{{(\\cdot)\\rightarrow Q_{v_idx+1}^+}}$"
            ax.set_title(title_lbl, fontsize=18, pad=4)

    return fig


# ── Pass / fail criteria ────────────────────────────────────────────────────
_CRITERIA = {
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
    res   = results[0]
    row   = res["aci_row"]
    nvars = res["nvars"]

    sources      = [j for j in range(nvars) if j != 0]
    abs_scores   = {j: float(row[j]) for j in sources}
    dominant_j   = int(max(sources, key=lambda j: abs_scores[j]))
    dominant_lbl = f"Q{dominant_j + 1}"
    dom_abs      = abs_scores[dominant_j]

    rel       = _rel_scores(row, 0)
    dom_score = float(rel[dominant_j])

    expected_j, _, note = _CRITERIA.get(case, (None, None, "unknown"))

    if case == 3:
        passed = bool(dom_abs < 1e-5)
    elif case == 4:
        passed = bool(
            abs_scores[1] > 1e-7 and abs_scores[2] > 1e-7
            and min(abs_scores[1], abs_scores[2]) / max(abs_scores[1], abs_scores[2]) > 0.5
        )
    elif expected_j is not None:
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

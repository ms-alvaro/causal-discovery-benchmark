#!/usr/bin/env python3
"""
Run all methods in methods/ on the four building-block benchmark cases,
save figures to results/figures/, and update LOG.md.

Usage:
    python run_benchmarks.py           # default N=200_000
    python run_benchmarks.py --N 5000000
"""
import argparse
import importlib.util
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from benchmarks.building_blocks import CASES
from generate_data import load as load_data, generate_and_save, data_path

METHODS_DIR  = Path("methods")
FIGURES_DIR  = Path("results/figures")
RESULTS_DIR  = Path("results")
LOG_FILE     = Path("LOG.md")

_RESULTS_START  = "<!-- RESULTS:START -->"
_RESULTS_END    = "<!-- RESULTS:END -->"
_METHODS_START  = "<!-- METHODS:START -->"
_METHODS_END    = "<!-- METHODS:END -->"


def load_methods() -> dict:
    methods = {}
    for f in sorted(METHODS_DIR.glob("*.py")):
        if f.name.startswith("_") or f.name == "__init__.py":
            continue
        spec = importlib.util.spec_from_file_location(f.stem, f)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        methods[f.stem] = mod
    return methods


def run_all(N: int, nbins: int, nlag: int, seed: int) -> tuple:
    np.random.seed(seed)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    methods = load_methods()
    all_results = {}   # all_results[method_key][case_id] = eval dict
    all_raw     = {}   # all_raw[method_key][case_id] = raw results list

    for case_id, case_info in CASES.items():
        print(f"\n{'='*60}")
        print(f"Case {case_id}: {case_info['name']}  —  {case_info['description']}")
        print(f"{'='*60}")

        # Load pre-generated data if available, otherwise generate on the fly
        try:
            X = load_data(case_id, N)
            print(f"  (loaded from cache)")
        except FileNotFoundError:
            X = case_info["fn"](N + 5_000)
            X = X[:, 5_000:]

        for key, method in methods.items():
            print(f"  [{method.NAME}] running...", end=" ", flush=True)
            t0      = time.perf_counter()
            results = method.run(X, nbins=nbins, nlag=nlag)
            elapsed = time.perf_counter() - t0
            ev      = method.evaluate(results, case_id)

            all_results.setdefault(key, {})[case_id] = ev
            all_raw.setdefault(key, {})[case_id]     = results

            status = (
                "PASS ✓" if ev["pass"] is True else
                "FAIL ✗" if ev["pass"] is False else
                "?"
            )
            print(f"{status}  dominant={ev['dominant']} ({ev['score']:.2f})  [{elapsed:.1f}s]")

        # collect raw results for combined figure
        pass

    # ── save single-page PDF per method with all cases ───────────────────────
    import matplotlib.pyplot as plt

    for key, method in methods.items():
        if not hasattr(method, "plot_all_cases"):
            continue
        fig      = method.plot_all_cases(all_raw[key], CASES)
        pdf_path = FIGURES_DIR / f"{key}_all_cases.pdf"
        fig.savefig(pdf_path, dpi=150, bbox_inches="tight", format="pdf")
        plt.close(fig)
        print(f"\n  [{key}] single-page PDF saved → {pdf_path}")

    return methods, all_results


def _replace_section(text, start_marker, end_marker, new_content):
    if start_marker in text and end_marker in text:
        before = text[: text.index(start_marker) + len(start_marker)]
        after  = text[text.index(end_marker):]
        return before + "\n\n" + new_content + "\n\n" + after
    return text + f"\n\n{start_marker}\n\n{new_content}\n\n{end_marker}\n"


def build_results_block(methods, all_results, N):
    case_ids = sorted(CASES)
    header   = "| Method | " + " | ".join(f"Case {i}: {CASES[i]['name']}" for i in case_ids) + " |"
    sep      = "| --- | " + " | ".join(["---"] * len(case_ids)) + " |"
    rows = []
    for key, method in sorted(methods.items()):
        cells = []
        for i in case_ids:
            ev   = all_results[key][i]
            mark = "✓" if ev["pass"] is True else ("✗" if ev["pass"] is False else "?")
            cells.append(f"{mark} `{ev['dominant']}` ({ev['score']:.2f})")
        rows.append(f"| {method.NAME} | " + " | ".join(cells) + " |")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"_Last run: {ts} — N={N:,}_\n\n" + header + "\n" + sep + "\n" + "\n".join(rows)


def build_methods_block(methods):
    lines = []
    for key, method in sorted(methods.items()):
        lines += [f"### {method.NAME}", f"**Definition:** {method.DEFINITION}",
                  f"**Reference:** {method.REFERENCE}", ""]
    return "\n".join(lines)


def save_results_log(methods, all_results, N):
    """Save a detailed per-method results file to results/<method>_results.txt."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for key, method in sorted(methods.items()):
        lines = [
            f"{'='*60}",
            f"Method : {method.NAME}",
            f"Run    : {ts}",
            f"N      : {N:,}",
            f"{'='*60}",
            "",
        ]
        for case_id, case_info in CASES.items():
            ev = all_results[key][case_id]
            status = "PASS" if ev["pass"] is True else ("FAIL" if ev["pass"] is False else "PENDING")
            lines += [
                f"Case {case_id}: {case_info['name']}  ({case_info['description']})",
                f"  Status   : {status}",
                f"  Dominant : {ev['dominant']}  (score={ev['score']:.4f})",
                f"  Expected : {ev['expected']}",
                f"  Note     : {ev['note']}",
            ]
            if "all_scores" in ev:
                lines.append("  All scores (normalised):")
                for label, val in sorted(ev["all_scores"].items(), key=lambda x: -x[1]):
                    if val > 0.001:
                        lines.append(f"    {label:8s}: {val:.4f}")
            lines.append("")
        out_path = RESULTS_DIR / f"{key}_results.txt"
        out_path.write_text("\n".join(lines))
        print(f"  Results log saved → {out_path}")


def update_log(methods, all_results, N):
    text = LOG_FILE.read_text() if LOG_FILE.exists() else _default_log()
    text = _replace_section(text, _RESULTS_START, _RESULTS_END,
                            build_results_block(methods, all_results, N))
    text = _replace_section(text, _METHODS_START, _METHODS_END,
                            build_methods_block(methods))
    LOG_FILE.write_text(text)
    print(f"\nLOG.md updated.")


def _default_log():
    return """\
# Causal Inference Benchmark — LOG

Benchmark suite for comparing causal inference methods on four canonical
building-block cases. Each new method is added as a script in `methods/`;
running `python run_benchmarks.py` updates this file automatically.

---

## Benchmark Cases

| # | Name | Description | Pass criterion for Q1 |
|---|------|-------------|----------------------|
| 1 | Mediator    | Q3→Q2→Q1 (no direct Q3→Q1)                | `U2` dominates (Q2 is the direct driver)           |
| 2 | Confounder  | Q3→Q1 and Q3→Q2 (common cause)            | `U2` must be absent (Q2→Q1 would be spurious)      |
| 3 | Synergistic | Q2×Q3→Q1 (interaction required)           | `S23` dominates (only joint Q2,Q3 predicts Q1)     |
| 4 | Redundant   | Q2=Q3→Q1 (identical information)          | `R23` dominates (Q2 and Q3 carry the same info)    |

---

## Results

<!-- RESULTS:START -->
_Run `python run_benchmarks.py` to populate this table._
<!-- RESULTS:END -->

---

## Method Descriptions

<!-- METHODS:START -->
_Run `python run_benchmarks.py` to populate this section._
<!-- METHODS:END -->
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N",     type=int, default=200_000)
    parser.add_argument("--nbins", type=int, default=50)
    parser.add_argument("--nlag",  type=int, default=1)
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    methods, all_results = run_all(args.N, args.nbins, args.nlag, args.seed)
    update_log(methods, all_results, args.N)
    save_results_log(methods, all_results, args.N)

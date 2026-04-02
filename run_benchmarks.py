#!/usr/bin/env python3
"""
Run all methods in methods/ on the four building-block benchmark cases,
save figures to results/figures/, and update LOG.md.

Usage:
    python run_benchmarks.py           # default N=200_000
    python run_benchmarks.py --method all --case all
    python run_benchmarks.py --method cte --case all
    python run_benchmarks.py --method all --case mediator
    python run_benchmarks.py --N 5000000
    python run_benchmarks.py --method cte
    python run_benchmarks.py --method cte --case mediator --case synergistic --N 5000000
"""
import argparse
import importlib.util
import re
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
CASE_NAME_TO_ID = {
    info["name"].lower(): case_id for case_id, info in CASES.items()
}
ALL_CASE_IDS = sorted(CASES)
README_METHOD_ORDER = ["cgc", "cte", "ccm", "lif", "pcmci", "ig", "surd", "aci"]


def load_methods(selected=None) -> dict:
    methods = {}
    for f in sorted(METHODS_DIR.glob("*.py")):
        if f.name.startswith("_") or f.name == "__init__.py":
            continue
        if selected is not None and f.stem not in selected:
            continue
        spec = importlib.util.spec_from_file_location(f.stem, f)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        methods[f.stem] = mod
    return methods


def _normalise_method_selection(method_names):
    if not method_names:
        return None
    selected = {name.strip().lower() for name in method_names if name.strip()}
    if "all" in selected:
        return None
    return sorted(selected)


def _validate_method_selection(selected):
    if selected is None:
        return
    available = sorted(
        f.stem for f in METHODS_DIR.glob("*.py")
        if not f.name.startswith("_") and f.name != "__init__.py"
    )
    missing = [name for name in selected if name not in available]
    if missing:
        raise SystemExit(
            "Unknown method(s): "
            + ", ".join(missing)
            + "\nAvailable methods: "
            + ", ".join(available)
        )


def _normalise_case_selection(case_names):
    if not case_names:
        return sorted(CASES)

    normalised = {case_name.strip().lower() for case_name in case_names if case_name.strip()}
    if "all" in normalised:
        return sorted(CASES)

    selected_case_ids = []
    unknown = []
    for case_name in case_names:
        key = case_name.strip().lower()
        case_id = CASE_NAME_TO_ID.get(key)
        if case_id is None:
            unknown.append(case_name)
        else:
            selected_case_ids.append(case_id)

    if unknown:
        raise SystemExit(
            "Unknown case name(s): "
            + ", ".join(unknown)
            + "\nAvailable cases: "
            + ", ".join(sorted(CASE_NAME_TO_ID))
        )

    return sorted(set(selected_case_ids))


def _validate_case_selection(case_ids):
    missing = [case_id for case_id in case_ids if case_id not in CASES]
    if missing:
        raise SystemExit(
            "Unknown case id(s): "
            + ", ".join(map(str, missing))
            + "\nAvailable cases: "
            + ", ".join(map(str, sorted(CASES)))
        )


def _all_cases_selected(case_ids) -> bool:
    return sorted(case_ids) == ALL_CASE_IDS


def _ordered_method_items(methods: dict):
    ordered = []
    seen = set()

    for key in README_METHOD_ORDER:
        if key in methods:
            ordered.append((key, methods[key]))
            seen.add(key)

    for key in sorted(methods):
        if key not in seen:
            ordered.append((key, methods[key]))

    return ordered


def run_all(N: int, nbins: int, nlag: int, seed: int,
            method_names=None, case_names=None) -> tuple:
    np.random.seed(seed)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    selected_methods = _normalise_method_selection(method_names)
    _validate_method_selection(selected_methods)
    methods = load_methods(selected_methods)
    if not methods:
        raise SystemExit("No methods selected.")

    selected_cases = _normalise_case_selection(case_names)
    _validate_case_selection(selected_cases)
    generated_cache = False

    all_results = {}   # all_results[method_key][case_id] = eval dict
    all_raw     = {}   # all_raw[method_key][case_id] = raw results list

    for case_id in selected_cases:
        case_info = CASES[case_id]
        print(f"\n{'='*60}")
        print(f"Case {case_id}: {case_info['name']}  —  {case_info['description']}")
        print(f"{'='*60}")

        # Load pre-generated data if available, otherwise generate on the fly
        try:
            X = load_data(case_id, N)
            print(f"  (loaded from cache)")
        except FileNotFoundError:
            if not generated_cache:
                print(f"  (cache miss for N={N:,}; generating and saving datasets)")
                generate_and_save(N, seed)
                generated_cache = True
            X = load_data(case_id, N)
            print(f"  (generated and loaded from cache)")

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
            spur = ev.get("spurious", [])
            spur_str = f"  spurious={spur}" if spur else ""
            print(f"{status}  dominant={ev['dominant']} ({ev['score']:.2f})  [{elapsed:.1f}s]{spur_str}")

        # collect raw results for combined figure
        pass

    if _all_cases_selected(selected_cases):
        # ── save single-page PDF per method with all cases ───────────────────
        import matplotlib.pyplot as plt

        for key, method in methods.items():
            if not hasattr(method, "plot_all_cases"):
                continue
            case_info = {case_id: CASES[case_id] for case_id in selected_cases}
            fig      = method.plot_all_cases(all_raw[key], case_info)
            pdf_path = FIGURES_DIR / f"{key}_all_cases.pdf"
            fig.savefig(pdf_path, dpi=150, bbox_inches="tight", format="pdf")
            plt.close(fig)
            print(f"\n  [{key}] single-page PDF saved → {pdf_path}")
    else:
        print("\nFigures not saved because not all benchmark cases were run.")

    return methods, all_results, selected_cases


def _replace_section(text, start_marker, end_marker, new_content):
    if start_marker in text and end_marker in text:
        before = text[: text.index(start_marker) + len(start_marker)]
        after  = text[text.index(end_marker):]
        return before + "\n\n" + new_content + "\n\n" + after
    return text + f"\n\n{start_marker}\n\n{new_content}\n\n{end_marker}\n"


def _format_result_cell(ev: dict) -> str:
    return "✓" if ev["pass"] is True else ("✗" if ev["pass"] is False else "?")


def _parse_existing_results_table(text: str) -> dict:
    if _RESULTS_START not in text or _RESULTS_END not in text:
        return {}

    start = text.index(_RESULTS_START) + len(_RESULTS_START)
    end = text.index(_RESULTS_END)
    block = text[start:end]

    rows = {}
    for line in block.splitlines():
        striped = line.strip()
        if not striped.startswith("|"):
            continue
        parts = [part.strip() for part in striped.strip("|").split("|")]
        if len(parts) < len(ALL_CASE_IDS) + 1:
            continue
        if parts[0] in {"Method", "---"} or re.fullmatch(r"-+", parts[0]):
            continue
        rows[parts[0]] = parts[1:1 + len(ALL_CASE_IDS)]
    return rows


def build_results_block(methods, all_results, N, case_ids, previous_rows=None):
    header   = "| Method | " + " | ".join(f"Case {i}: {CASES[i]['name']}" for i in ALL_CASE_IDS) + " |"
    sep      = "| --- | " + " | ".join(["---"] * len(ALL_CASE_IDS)) + " |"
    previous_rows = previous_rows or {}
    rows = []

    for key, method in _ordered_method_items(methods):
        cells = list(previous_rows.get(method.NAME, ["—"] * len(ALL_CASE_IDS)))
        if len(cells) < len(ALL_CASE_IDS):
            cells.extend(["—"] * (len(ALL_CASE_IDS) - len(cells)))
        for case_id in case_ids:
            if key in all_results and case_id in all_results[key]:
                cells[ALL_CASE_IDS.index(case_id)] = _format_result_cell(all_results[key][case_id])
        rows.append(f"| {method.NAME} | " + " | ".join(cells[:len(ALL_CASE_IDS)]) + " |")

    scope_methods = ", ".join(method.NAME for _, method in _ordered_method_items(methods))
    scope_cases = ", ".join(CASES[case_id]["name"] for case_id in case_ids)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        f"_Last run: {ts} — N={N:,} — updated: {scope_methods} / {scope_cases}_\n\n"
        + header + "\n" + sep + "\n" + "\n".join(rows)
    )


def build_methods_block(methods):
    lines = []
    for key, method in _ordered_method_items(methods):
        lines += [f"### {method.NAME}", f"**Definition:** {method.DEFINITION}",
                  f"**Reference:** {method.REFERENCE}", ""]
    return "\n".join(lines)


def save_results_log(methods, all_results, N, case_ids):
    """Save a detailed per-method results file to results/<method>_results.txt."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for key, method in _ordered_method_items(methods):
        lines = [
            f"{'='*60}",
            f"Method : {method.NAME}",
            f"Run    : {ts}",
            f"N      : {N:,}",
            f"{'='*60}",
            "",
        ]
        for case_id in case_ids:
            case_info = CASES[case_id]
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
            if ev.get("spurious"):
                lines.append(f"  Spurious   : {', '.join(ev['spurious'])}")
            lines.append("")
        out_path = RESULTS_DIR / f"{key}_results.txt"
        out_path.write_text("\n".join(lines))
        print(f"  Results log saved → {out_path}")


def update_log(methods, all_results, N, case_ids):
    text = LOG_FILE.read_text() if LOG_FILE.exists() else _default_log()
    all_methods = load_methods()
    previous_rows = _parse_existing_results_table(text)
    text = _replace_section(text, _RESULTS_START, _RESULTS_END,
                            build_results_block(all_methods, all_results, N, case_ids, previous_rows))
    text = _replace_section(text, _METHODS_START, _METHODS_END,
                            build_methods_block(all_methods))
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
    parser.add_argument("--method", action="append",
                        help="Method key in methods/ to run, or 'all'. Repeat to run a subset, e.g. --method cte --method surd.")
    parser.add_argument("--case", action="append",
                        help="Benchmark case name to run, or 'all'. Repeat to run a subset, e.g. --case mediator --case redundant.")
    args = parser.parse_args()

    methods, all_results, case_ids = run_all(
        args.N,
        args.nbins,
        args.nlag,
        args.seed,
        method_names=args.method,
        case_names=args.case,
    )
    save_results_log(methods, all_results, args.N, case_ids)
    update_log(methods, all_results, args.N, case_ids)

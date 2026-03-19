#!/usr/bin/env python3
"""
Pre-generate benchmark time series and save to results/data/.

Each dataset is saved as a .npz file:
    results/data/case<N>_<name>_N<samples>.npz

Usage:
    python generate_data.py                        # default N values
    python generate_data.py --N 200000 5000000
    python generate_data.py --N 200000 --seed 42
"""
import argparse
from pathlib import Path

import numpy as np

from benchmarks.building_blocks import CASES

DATA_DIR = Path("results/data")
TRANSIENT = 5_000   # samples discarded as transient


def data_path(case_id: int, N: int) -> Path:
    name = CASES[case_id]["name"].lower()
    return DATA_DIR / f"case{case_id}_{name}_N{N}.npz"


def generate_and_save(N: int, seed: int):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    for case_id, case_info in CASES.items():
        path = data_path(case_id, N)
        if path.exists():
            print(f"  [skip] {path.name} already exists")
            continue
        print(f"  Generating Case {case_id}: {case_info['name']}  N={N:,}...", end=" ", flush=True)
        X = case_info["fn"](N + TRANSIENT)
        X = X[:, TRANSIENT:]   # discard transient
        np.savez_compressed(path, X=X, N=N, case_id=case_id,
                            name=case_info["name"], seed=seed)
        print(f"saved → {path.name}")


def load(case_id: int, N: int) -> np.ndarray:
    """Load pre-generated data for a given case and N. Raises if not found."""
    path = data_path(case_id, N)
    if not path.exists():
        raise FileNotFoundError(
            f"Data not found: {path}\n"
            f"Run `python generate_data.py --N {N}` first."
        )
    return np.load(path)["X"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N",    type=int, nargs="+",
                        default=[200_000, 1_000_000, 5_000_000])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for N in args.N:
        print(f"\nN = {N:,}")
        generate_and_save(N, args.seed)

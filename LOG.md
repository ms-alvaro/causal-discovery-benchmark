# Causal Inference Benchmark — LOG

Benchmark suite for comparing causal inference methods on four canonical
building-block cases. Each new method is added as a script in `methods/`;
running `python run_benchmarks.py` updates this file automatically.

---

## Benchmark Cases

| # | Name | Description | Pass criterion for Q1 |
|---|------|-------------|----------------------|
| 1 | Mediator    | Q3→Q2→Q1 (no direct Q3→Q1)                | `U2` dominates   |
| 2 | Confounder  | Q3→Q1 and Q3→Q2 (common cause)            | `U2` must be absent (spurious) |
| 3 | Synergistic | Q2×Q3→Q1 (interaction required)           | `S23` dominates  |
| 4 | Redundant   | Q2=Q3→Q1 (identical information)          | `R23` dominates  |

---

## Results

<!-- RESULTS:START -->

_Last run: 2026-03-19 16:48 — N=200,000_

| Method | Case 1: Mediator | Case 2: Confounder | Case 3: Synergistic | Case 4: Redundant |
| --- | --- | --- | --- | --- |
| SURD | ? `U2` (0.93) | ? `S13` (0.49) | ? `S23` (0.77) | ? `S12` (0.43) |

<!-- RESULTS:END -->

---

## Method Descriptions

<!-- METHODS:START -->

### SURD
**Definition:** Decomposes I(target_future ; sources_present) into unique (U), redundant (R), and synergistic (S) contributions per source combination via specific mutual information.
**Reference:** Martínez-Sánchez & Lozano-Durán, Commun. Phys. 9, 15 (2025). https://doi.org/10.1038/s42005-025-02447-w


<!-- METHODS:END -->

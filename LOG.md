# Causal Inference Benchmark

Benchmark suite for comparing causal inference methods on four canonical
building-block cases. Each new method is added as a script in `methods/`;
running `python run_benchmarks.py` updates this file automatically.

---

## Benchmark Cases

| # | Name | Description | Q1⁺ sources | Q2⁺ sources | Q3⁺ sources |
|---|------|-------------|----------------------|----------------------|----------------------|
| 1 | Mediator    | Q3→Q2→Q1       | Q2                | Q3                | Q3         |
| 2 | Confounder  | Q3→Q1 and Q3→Q2   | Q1, Q3     | Q2, Q3     | Q3 (self)         |
| 3 | Synergistic | Q2×Q3→Q1 | Q2, Q3            | Q2         | Q3 (self)         |
| 4 | Redundant   | Q2=Q3→Q1 | Q1, Q2, Q3 | Q2, Q3     | Q2, Q3 (self)     |

---

## Results

<!-- RESULTS:START -->

_Last run: 2026-03-19 23:37 — N=200,000_

| Method | Case 1: Mediator | Case 2: Confounder | Case 3: Synergistic | Case 4: Redundant |
| --- | --- | --- | --- | --- |
| ACI | ✓ `Q2` (1.00) | ✓ `Q3` (0.95) | ✓ `Q2` (0.57) | ✓ `Q2` (0.50) |
| IG | ✓ `Q2` (0.97) | ✓ `Q3` (0.93) | ✓ `Q2` (0.57) | ✓ `Q2` (0.50) |
| SURD | ✓ `U2` (0.93) | ✓ `S13` (0.49) | ✓ `S23` (0.77) | ✓ `S12` (0.43) |

<!-- RESULTS:END -->

---

## Method Descriptions

<!-- METHODS:START -->

### ACI
**Definition:** Measures causal influence via KL divergence between the Bayesian filter p(y_t|x_{0:t}) and smoother p(y_t|x_{0:T}) in a linearised CGNS framework.
**Reference:** Andreou, Chen & Bollt, Nat. Commun. 17, 1854 (2026). https://doi.org/10.1038/s41467-026-68568-0

### IG
**Definition:** Tests whether adding a putative cause X to the present state of the target Y reduces the Information Imbalance Delta((alpha*X,Y)_t -> Y_{t+tau}); the Imbalance Gain IG = (Delta(0) - min_alpha Delta(alpha)) / Delta(0) quantifies causal influence.
**Reference:** Del Tatto, Fortunato, Bueti & Laio, PNAS 121, e2317256121 (2024). https://doi.org/10.1073/pnas.2317256121

### SURD
**Definition:** Decomposes I(target_future ; sources_present) into unique (U), redundant (R), and synergistic (S) contributions per source combination via specific mutual information.
**Reference:** Martínez-Sánchez, Arranz & Lozano-Durán, Nat. Commun. 15, 9296 (2024). https://doi.org/10.1038/s41467-024-53373-4


<!-- METHODS:END -->

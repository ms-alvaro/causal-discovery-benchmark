# Causal Inference Benchmark

Benchmark suite for comparing causal inference methods on four canonical
building-block cases. Each new method is added as a script in `methods/`;
running `python run_benchmarks.py` updates this file automatically.

---

## Benchmark Cases

| # | Name | Description | Q1⁺ sources | Q2⁺ sources | Q3⁺ sources |
|---|------|-------------|----------------------|----------------------|----------------------|
| 1 | Mediator    | Q3→Q2→Q1       | Q2                | Q3                | Q3         |
| 2 | Confounder  | Q3→Q1 and Q3→Q2   | Q1, Q3     | Q2, Q3     | Q3         |
| 3 | Synergistic | Q2×Q3→Q1 | Q2, Q3            | Q2         | Q3         |
| 4 | Redundant   | Q2=Q3→Q1 | Q1, Q2, Q3 | Q2, Q3     | Q2, Q3     |

---

## Results

<!-- RESULTS:START -->

_Last run: 2026-03-20 12:30 — N=200,000_

| Method | Case 1: Mediator | Case 2: Confounder | Case 3: Synergistic | Case 4: Redundant |
| --- | --- | --- | --- | --- |
| ACI | ✓ `Q2` (1.00) | ✗ `Q3` (0.95) ⚠`Q1→Q2⁺`,`Q1→Q3⁺` | ✓ `Q2` (0.57) | ✓ `Q2` (0.50) |
| CCM | ✗ `Q3` (0.51) ⚠`Q1→Q2⁺`,`Q1→Q3⁺`,`Q2→Q3⁺` | ✗ `Q2` (0.72) ⚠`Q2→Q1⁺`,`Q1→Q2⁺`,`Q1→Q3⁺`,`Q2→Q3⁺` | ✗ `Q2` (1.00) | ✗ `Q2` (0.00) ⚠`Q1→Q2⁺`,`Q1→Q3⁺` |
| CGC | ✗ `Q2` (1.00) ⚠`Q1→Q2⁺` | ✓ `Q3` (1.00) | ✗ `Q2` (0.63) ⚠`Q1→Q2⁺` | ✗ `Q2` (0.00) |
| CTE | ✗ `Q2` (1.00) ⚠`Q1→Q3⁺`,`Q2→Q3⁺` | ✗ `Q3` (0.99) ⚠`Q1→Q3⁺`,`Q2→Q3⁺` | ✗ `Q3` (0.51) ⚠`Q1→Q2⁺`,`Q3→Q2⁺`,`Q1→Q3⁺`,`Q2→Q3⁺` | ✗ `Q2` (0.00) ⚠`Q1→Q2⁺`,`Q1→Q3⁺` |
| IG | ✗ `Q2` (0.97) ⚠`Q1→Q3⁺` | ✗ `Q3` (0.93) ⚠`Q1→Q3⁺`,`Q2→Q3⁺` | ✗ `Q2` (0.57) ⚠`Q1→Q2⁺`,`Q3→Q2⁺`,`Q1→Q3⁺`,`Q2→Q3⁺` | ✓ `Q2` (0.50) |
| LIF | ✗ `Q2` (1.00) ⚠`Q1→Q2⁺` | ✗ `Q3` (1.00) ⚠`Q2→Q3⁺` | ✗ `Q2` (0.00) | ✗ `Q2` (0.00) |
| PCMCI | ✗ `Q2` (0.00) | ✗ `Q2` (0.00) ⚠`Q1→Q3⁺` | ✗ `Q2` (0.00) | ✗ `Q2` (0.00) |
| SURD | ✓ `U2` (0.93) | ✓ `S13` (0.49) | ✓ `S23` (0.77) | ✓ `S12` (0.43) |

<!-- RESULTS:END -->

---

## Method Descriptions

<!-- METHODS:START -->

### ACI
**Definition:** Measures causal influence via KL divergence between the Bayesian filter p(y_t|x_{0:t}) and smoother p(y_t|x_{0:T}) in a linearised CGNS framework.
**Reference:** Andreou, Chen & Bollt, Nat. Commun. 17, 1854 (2026). https://doi.org/10.1038/s41467-026-68568-0

### CCM
**Definition:** Cross-maps Q_j from shadow manifold M_i (delay embedding of Q_i); CCM_{j→i} = corr(Q_j, Q̂_j|M_i) via E+1 nearest-neighbour reconstruction.
**Reference:** Sugihara et al., Science 338:496 (2012); Takens, Lecture Notes Math. 898:366 (1981); Martínez-Sánchez, Arranz & Lozano-Durán, Nat. Commun. 15, 9296 (2024). https://doi.org/10.1038/s41467-024-53373-4

### CGC
**Definition:** Tests whether Q_j's past improves prediction of Q_i beyond all other variables; CGC_{j→i} = log₂[var(ε̂_restricted)/var(ε_unrestricted)] using OLS VAR models.
**Reference:** Geweke (1982) J. Am. Stat. Assoc. 77:304; Barnett & Seth (2014) J. Neurosci. Methods 223:50; Martínez-Sánchez, Arranz & Lozano-Durán, Nat. Commun. 15, 9296 (2024). https://doi.org/10.1038/s41467-024-53373-4

### CTE
**Definition:** CTE_{j→i} = H(Q_i⁺|Q̄_j) − H(Q_i⁺|Q): unique information Q_j provides about Q_i's future beyond all other variables, estimated via histograms.
**Reference:** Schreiber (2000) Phys. Rev. Lett. 85:461; Barnett, Barrett & Seth (2009) Phys. Rev. Lett. 103:238701; Martínez-Sánchez, Arranz & Lozano-Durán, Nat. Commun. 15, 9296 (2024). https://doi.org/10.1038/s41467-024-53373-4

### IG
**Definition:** Tests whether adding a putative cause X to the present state of the target Y reduces the Information Imbalance Delta((alpha*X,Y)_t -> Y_{t+tau}); the Imbalance Gain IG = (Delta(0) - min_alpha Delta(alpha)) / Delta(0) quantifies causal influence.
**Reference:** Del Tatto, Fortunato, Bueti & Laio, PNAS 121, e2317256121 (2024). https://doi.org/10.1073/pnas.2317256121

### LIF
**Definition:** T_{j→i} = rate of Shannon entropy transfer from X_j to X_i, conditioned on all other variables via the cofactor matrix of C (multivariate Liang 2021 formula).
**Reference:** Liang (2014) Phys. Rev. E 90:052150; Liang (2021) Entropy 23:679; Liang & Kleeman (2005) Phys. Rev. Lett. 95:244101.

### PCMCI
**Definition:** Two-phase causal discovery: PC parent selection via conditional independence testing, followed by MCI test using CMI (k-NN) to remove spurious links.
**Reference:** Runge et al., Sci. Adv. 5:eaau4996 (2019); Tigramite package: https://github.com/jakobrunge/tigramite; Martínez-Sánchez, Arranz & Lozano-Durán, Nat. Commun. 15, 9296 (2024). https://doi.org/10.1038/s41467-024-53373-4

### SURD
**Definition:** Decomposes I(target_future ; sources_present) into unique (U), redundant (R), and synergistic (S) contributions per source combination via specific mutual information.
**Reference:** Martínez-Sánchez, Arranz & Lozano-Durán, Nat. Commun. 15, 9296 (2024). https://doi.org/10.1038/s41467-024-53373-4


<!-- METHODS:END -->

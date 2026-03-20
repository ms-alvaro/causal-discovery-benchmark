# Causal Inference Benchmark

A growing suite for comparing causal inference methods on four canonical
building-block cases. Every time a new method is found in the literature,
it is implemented as a script in `methods/`, run on all cases, and the
results table in `LOG.md` is updated automatically.

---

## Benchmark cases

The four cases are the *building blocks* of causal interaction from:

> Martínez-Sánchez, Arranz & Lozano-Durán, *Nature Communications* 15, 9296 (2024). 
> <https://doi.org/10.1038/s41467-024-53373-4>

| # | Name | Governing equations (Q1, Q2, Q3) | Expected structure |
|---|------|----------------------------------|--------------------|
| 1 | **Mediator**    | Q1 = sin(Q2) + ε, Q2 = cos(Q3) + ε, Q3 = AR(1)       | Q3→Q2→Q1     |
| 2 | **Confounder**  | Q1 = sin(Q1+Q3) + ε, Q2 = cos(Q2−Q3) + ε, Q3 = AR(1) | Q3→Q1 and Q3→Q2 |
| 3 | **Synergistic** | Q1 = sin(Q2·Q3) + ε, Q2 = AR(1), Q3 = AR(1)           | Q2×Q3→Q1   |
| 4 | **Redundant**   | Q1 = 0.3Q1 + sin(Q2·Q3) + ε, Q2 = AR(1), Q3 = Q2     | Q2=Q3→Q1|

---

## Methods

| Method | Multivariate | Nonlinear | Stochastic | Contemporaneous | Leak | Time-delay | Self-causation |
|--------|:-----------:|:---------:|:----------:|:---------------:|:----:|:----------:|:--------------:|
| ACI    | ✗ᵃ | ✗  | ✓  | ✗  | ✓  | ✓  | ✗  |
| CCM    | ✗  | ✓  | ✗ᵇ | ✓  | ✗  | ✗ᶜ | ✗  |
| CGC    | ✓  | ✗  | ✓  | ✗  | ✗  | ✓  | ✓  |
| CTE    | ✓  | ✓  | ✗  | ✗  | ✓  | ✓  | ✓  |
| IG     | ✗ᵃ | ✓  | ✓  | ✗  | ✗  | ✓  | ✗  |
| LIF    | ✓  | ✗  | ✓  | ✓  | ✗  | ✓  | ✓  |
| PCMCI  | ✓  | ✓  | ✗ᵈ | ✗  | ✓  | ✓  | ✓  |
| SURD   | ✓  | ✓  | ✓  | ✓  | ✓  | ✓  | ✓  |

**Column definitions:** *Multivariate* — conditions on all observed variables simultaneously; *Nonlinear* — detects nonlinear dependencies; *Stochastic* — designed for nondeterministic processes; *Contemporaneous* — detects instantaneous (lag-0) links; *Leak* — estimates information from unobserved variables; *Time-delay* — detects time-lagged causal links; *Self-causation* — detects auto-causal (self-lagged) effects.

ᵃ Inherently pairwise (one source, one target at a time); multivariate inference requires all pairwise comparisons.
ᵇ CCM aims to reconstruct the attractor manifold, making it potentially effective for stochastic systems; however, increased dynamical noise complicates manifold reconstruction.
ᶜ Extended CCM (eCCM) introduces time-delayed causal interactions.
ᵈ PCMCI+ accounts for contemporaneous links.

For CGC, CTE, CCM, PCMCI, and SURD, values follow Table 2 of Martínez-Sánchez, Arranz & Lozano-Durán, *Nat. Commun.* 15, 9296 (2024).

---

## Results

See [LOG.md](LOG.md) for the full results table and per-method discussion.

---

## Repository structure

```
causal-inference-benchmark/
├── benchmarks/
│   └── building_blocks.py   # data generators for the 4 cases
├── methods/
│   ├── surd.py              # SURD (first method)
│   └── _surd/               # SURD core algorithm (from ALD-Lab/SURD)
├── results/
│   ├── figures/             # output PDFs (tracked) and PNGs (gitignored)
│   ├── data/                # pre-generated benchmark time series (by N)
│   └── <method>_results.txt # detailed per-run results log
├── run_benchmarks.py        # runs all methods and updates LOG.md
├── LOG.md                   # results table + method descriptions
└── requirements.txt
```

---

## Adding a new method

1. Create `methods/<method_name>.py` with the following interface:

```python
NAME       = "MyMethod"
DEFINITION = "One-line description of what causality means in this method."
REFERENCE  = "Author et al., Journal (Year). https://doi.org/..."

def run(X: np.ndarray, nbins: int = 50, nlag: int = 1) -> list:
    """Run the method. Returns one dict per target variable."""
    ...

def evaluate(results: list, case: int) -> dict:
    """Return {'pass': bool, 'dominant': str, 'score': float, 'expected': str, 'note': str}."""
    ...
```

2. Run `python run_benchmarks.py` — the results table in `LOG.md` updates automatically.

---

## Installation

```bash
# pip
pip install -r requirements.txt

# uv
uv pip install -r requirements.txt
```

## Running the benchmarks

```bash
python run_benchmarks.py              # N=200,000 (fast, ~3 s)
python run_benchmarks.py --N 5000000  # N=5,000,000 (converged, as in original paper)
```

> **Note:** The original paper uses N=5×10⁷ for fully converged results.
> N=200,000 is sufficient to identify the dominant contribution in most cases
> but results may shift slightly for cases with weaker causal signals.

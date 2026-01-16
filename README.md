# Occam's Hedge Under Relative Entropy Uncertainty

**Research-grade Python implementation** for studying regime-dependent semantic inversion in hedging policies under information constraints.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

---

## Overview

Hedging policies that exploit market microstructure signals face a hidden fragility: when intermediation constraints bind, the relationship between activity proxies and execution costs can **invert sign**, causing identical observations to imply opposite optimal responses across regimes. We formalize this phenomenon as **semantic inversion** and demonstrate that safely exploiting sign-unstable signals requires encoding regime-distinguishing information—creating a fundamental trade-off between baseline efficiency and stress robustness.

This repository implements **Occam's Hedge**, which prices representational bandwidth through a variational information bottleneck, enforcing structural priors that distinguish payoff-anchored exposures (Greeks) from equilibrium-derived proxies (microstructure). We show that as information constraints tighten, policies rationally shed dependence on semantically unstable signals, reducing wrong-way trading exposure at the cost of higher baseline risk.

**Key finding:** In a controlled identification environment where regime information is unavailable from single snapshots by design, an oracle test providing regime labels eliminates stress degradation entirely (Deg = 0.97), confirming the failure is purely informational rather than architectural.

---

## Research Contributions

### Conceptual Framework

- **Semantic inversion formalization**: We define semantic inversion as a representation-level failure mode where the *direction* of the optimal trading response to an observable flips across regimes, distinct from mere magnitude miscalibration. The *Conflict Set* identifies observations where regime-oblivious policies face irreducible ambiguity.

- **Latent fragility definition**: We introduce an operational definition of latent fragility—microstructure features that achieve comparable in-regime performance but degrade materially under stress due to semantic inversion, manifesting as positive wrong-way trading scores.

### Information-Theoretic Mechanism

- **Regime information requirement**: We prove that avoiding wrong-way trading under semantic inversion requires encoding Ω(1) bits of regime information, which is explicitly priced by the VIB KL-to-prior penalty (Proposition 3.5, Corollary 3.7).

- **Hierarchical information regularization**: We implement a structural prior assigning different information budgets to payoff-anchored channels (low β) versus equilibrium-derived channels (high β), providing a principled basis for why information constraints suppress brittle regime-contingent logic.

### Empirical Validation

In a controlled identification environment (variance-matched simulator):

1. **Universal baseline degradation**: All representations exhibit near-identical stress degradation (Deg ≈ 1.16) when regime information is unavailable, confirming microstructure provides no exploitable advantage without regime labels.

2. **Information-robustness frontier**: Tightening β reduces wrong-way exposure from +0.11 to ≈0 and degradation from 1.16 to 1.0, at the cost of 60% higher baseline risk.

3. **Oracle disambiguation test**: Providing regime labels eliminates degradation entirely (Deg = 0.97), confirming the failure is informational and the model class has sufficient capacity.

---

## Key Results

### Oracle "Smoking Gun" Test

When provided with the true regime label, stress degradation **vanishes**:

| Model | R₀ | R₁ | Degradation |
|-------|-----|-----|-------------|
| Combined (no oracle) | 0.194 | 0.225 | **1.16** |
| Combined (oracle) | 0.252 | 0.243 | **0.97** |

**Interpretation**: The failure is **informational**, not architectural. The model has sufficient capacity; it lacks regime-distinguishing information.

### Matched-R₀ Comparison (Table 8)

At matched baseline risk, VIB achieves lower stress degradation than L₂ regularization:

| Method | R₀ | R₁ | Deg | ProbeAUC |
|--------|-----|-----|-----|----------|
| L₂ (λ*) | ≈0.19 | ≈0.23 | ≈1.19 | ≈0.52 |
| VIB (β=0.5) | 0.279 | 0.294 | **1.05** | **0.47** |

**Interpretation**: VIB's representation-level constraint selectively suppresses regime-contingent logic, unlike L₂'s uniform parameter shrinkage.

---

## Paper & Citation

**Manuscript**: [`main_paper.tex`](main_paper.tex)  
**Author**: Ray Wang ([raywang886@gmail.com](mailto:raywang886@gmail.com))

If you use this code in your research, please cite:

```bibtex
@article{occam_hedge_2026,
  title={Occam's Hedge Under Relative Entropy Uncertainty},
  author={Wang, Ray},
  journal={[Under Review]},
  year={2026}
}
```

See [`CITATION.cff`](CITATION.cff) for structured citation metadata.

---

## Installation

### Requirements

- Python 3.10+
- Core dependencies: `numpy`, `torch`, `pandas`, `scikit-learn`, `matplotlib`

### Setup

```bash
# Clone repository
git clone https://github.com/raywang886/occam-hedge.git
cd occam-hedge

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Reproducibility

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for detailed reproduction instructions.

### Quick Validation (3 minutes)

Verify the oracle disambiguation mechanism:

```bash
python scripts/validation_suite.py --mode oracle
```

**Expected output**: Degradation ratio drops from ~1.16 (no oracle) to ~0.97 (with oracle).

### Main Paper Results (4-5 hours)

Reproduce VIB sweep and mechanism closure analysis:

```bash
python scripts/run_paper.py --config configs/paper_run.json
```

**Outputs**:
- `results/sweep_results.csv` – VIB β sweep results
- `results/regime_probe_vs_beta.json` – Regime information vs. β
- `figures/mechanism_closure.png` – Diagnostic plot

### Matched-R₀ Comparison for Table 8 (4-5 hours)

Compare VIB against L₂ regularization at matched baseline risk:

```bash
python scripts/run_regularization_control.py
```

**Alternative**: Interactive notebook [`TABLE8_EXECUTION.ipynb`](TABLE8_EXECUTION.ipynb)

---

## Repository Structure

```
occam-hedge/
├── src/                          # Core library
│   ├── experiment_occam.py       # Training & evaluation loops (VIB + hierarchical penalties)
│   ├── features.py               # Feature engineering & representation construction
│   ├── policies.py               # VIB policy architectures
│   ├── world.py                  # Heston simulator with leaky volume process
│   └── risk.py                   # Risk measures (ES, KL divergence)
├── scripts/                      # Executable workflows
│   ├── run_paper.py              # Main VIB β sweep (Figure 3, Tables 2-3)
│   ├── run_regularization_control.py  # Table 8 (L₂ vs VIB at matched R₀)
│   ├── validation_suite.py       # Oracle disambiguation test (Table 4)
│   ├── plot_mechanism_closure.py # Mechanism diagnostics (β vs Deg, ProbeAUC)
│   ├── regime_probe_z.py         # Latent regime information analysis
│   └── audit_identifiability.py  # Verifies snapshot non-identifiability
├── configs/                      # Experiment configurations
│   ├── paper_run.json            # Canonical paper configuration
│   ├── hierarchical_run.json     # Hierarchical β sweep config
│   └── paper_leaky.json          # Leaky simulator config
├── tests/                        # Unit tests (pytest)
├── results/                      # Experimental outputs (gitignored, except final results)
├── figures/                      # Generated figures (gitignored, except paper figures)
├── main_paper.tex                # LaTeX manuscript
├── TABLE8_EXECUTION.ipynb        # Interactive notebook for Table 8
├── REPRODUCIBILITY.md            # Detailed reproduction guide
└── README.md                     # This file
```

---

## Core Mechanism

The paper's mechanism requires three components:

1. **Temporal leak**: AR(1) volume process with regime-dependent autocorrelation (`leak_phi_r1=0.6`), enabling temporal regime inference
2. **Lagged micro features**: 4-lag history (`micro_lags=4`) for temporal integration
3. **Mixed-regime training**: 50/50 balanced regime sampling, forcing policies to handle regime ambiguity

**Semantic inversion**: Volume-impact elasticity *flips sign* across regimes:
- **Regime 0**: λ₀(Vol) ∝ 1/Vol (high volume ⇒ low impact, liquid regime)
- **Regime 1**: λ₁(Vol) ∝ Vol (high volume ⇒ high impact, constrained regime)

**VIB mechanism**: As β increases, the KL-to-prior penalty makes regime-contingent logic costly to encode. Since avoiding wrong-way trading requires regime information (Proposition 3.5), policies rationally default to payoff-anchored features when regime disambiguation is too expensive.

---

## Diagnostics & Analysis

### Mechanism Diagnostics

**Identifiability check** (can regime be inferred from snapshots?):
```bash
python scripts/audit_identifiability.py
```

**Latent regime probe** (does VIB suppress regime information in Z?):
```bash
python scripts/regime_probe_z.py
```

**Mechanism closure plot** (β vs degradation, wrong-way exposure):
```bash
python scripts/plot_mechanism_closure.py
```

### Simulator Configurations

| Mode | Description | Config Flags |
|------|-------------|--------------|
| **Variance-matched** | No leak (φ=0), baseline | `--leak_phi_r0 0.0 --leak_phi_r1 0.0` |
| **Leaky** | AR(1) temporal signal | `--leak_phi_r0 0.0 --leak_phi_r1 0.6 --micro_lags 4` |
| **Oracle** | Regime label provided | `--representation oracle` |

---

## Testing

Run unit tests:
```bash
pytest tests/
```

**Test coverage:**
- Mechanism integrity (lagged features, leaky simulator dynamics)
- Loss function sign consistency
- Hierarchical β penalties
- Oracle test functionality
- Feature engineering correctness

---

## Additional Documentation

- **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)**: Detailed step-by-step reproduction guide
- **[WORKFLOW.md](WORKFLOW.md)**: Development workflow and configuration management
- **[TABLE8_EXECUTION.ipynb](TABLE8_EXECUTION.ipynb)**: Interactive notebook for matched-R₀ comparison
- **[main_paper.tex](main_paper.tex)**: Full paper with theoretical framework and proofs

---

## License

MIT License – see [LICENSE](LICENSE) file for details.

---

## Contact

**Ray Wang**  
Email: [raywang886@gmail.com](mailto:raywang886@gmail.com)  
GitHub: [raywang886/occam-hedge](https://github.com/raywang886/occam-hedge)

---

## Keywords

Deep hedging • Execution costs • Regime shifts • Information bottleneck • Market microstructure • Semantic inversion • Variational inference • Model risk • Rational inattention

**JEL Codes**: G13, C61, G11

# Canonical Workflow: Occam's Hedge

This document describes the end-to-end workflow for reproducing the paper's findings.

---

## Prerequisites

1. **Environment Setup**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Verify Installation**
   ```bash
   python -c "from src.experiment_occam import *; print('✅ Imports OK')"
   ```

---

## Workflow 1: Oracle Disambiguation Test (3 minutes)

**Purpose**: Verify that stress fragility is informational (not architectural).

**Command**:
```bash
python scripts/validation_suite.py --mode oracle
```

**Expected Output**:
```
Oracle Results:
  R0 (Regime 0): 0.2517
  R1 (Regime 1): 0.2433
  Degradation:   0.97
Result: ✅ SUCCESS
```

**Interpretation**: Degradation ≈ 1.0 confirms the failure is informational.

**Output Files**:
- `results/oracle_test_results.json`

---

## Workflow 2: Main VIB Sweep (4-5 hours)

**Purpose**: Generate the information--robustness frontier (Figure X).

**Command**:
```bash
python scripts/run_paper.py --config configs/paper_run.json
```

**Key Parameters** (from `paper_run.json`):
- `n_steps`: 100
- `T`: 1.0 (1 year)
- `gamma`: 0.95 (ES₉₅)
- `micro_lags`: 4 (temporal features)
- `leak_phi_r1`: 0.6 (regime-dependent autocorrelation)
- `seeds`: 20 (reproducibility)

**Expected Runtime**: ~4-5 hours on CPU, ~1 hour on GPU

**Output Files**:
- `results/paper_sweep_aggregated.csv`
- `figures/fig_frontier_beta_sweep.png`
- `figures/fig_mechanism_closure.png`

**Success Criteria**:
- ✅ Degradation at β=0 ≈ 1.16
- ✅ Degradation at β=0.5 ≈ 1.05
- ✅ Information cost increases monotonically with β

---

## Workflow 3: Matched-R₀ Comparison (4-5 hours)

**Purpose**: Demonstrate VIB's advantage over generic L₂ regularization (Table 8).

**Command**:
```bash
python scripts/run_regularization_control.py
```

**Or use interactive notebook**:
```bash
jupyter notebook TABLE8_EXECUTION.ipynb
```

**What It Does**:
1. Trains VIB baseline (β=0)
2. Trains L₂ models with λ ∈ {0, 0.001, 0.01, 0.1, 1.0}
3. Finds best L₂ match where R₀(L₂) ≈ R₀(VIB)
4. Compares R₁, Deg, and ProbeAUC at matched R₀

**Expected Output** (abbreviated):
```
Best L2 Match:
  l2_lambda: 0.01
  Matched R0: 0.192
  R1: 0.230
  Deg: 1.20
  ProbeAUC: 0.52

VIB Baseline:
  R0: 0.194
  R1: 0.225
  Deg: 1.16
  ProbeAUC: 0.49
```

**Output Files**:
- `results/regularization_control_summary.csv`
- `results/regularization_control_summary.json`

**Interpretation**:
At matched R₀, VIB achieves:
- **Lower Deg** (1.16 < 1.20) — Better stress robustness
- **Lower ProbeAUC** (0.49 < 0.52) — Suppresses regime info

---

## Workflow 4: Diagnostic Checks

### 4.1 Identifiability Audit

**Purpose**: Verify that regime cannot be inferred from single snapshots.

```bash
python scripts/audit_identifiability.py
```

**Expected**: AUC ≈ 0.5 (chance level)

### 4.2 Regime Probe from Latent Z

**Purpose**: Confirm VIB reduces regime predictability from latent representation.

```bash
python scripts/regime_probe_z.py
```

**Expected**: AUC decreases with β (from ~0.50 to ~0.47)

### 4.3 Mechanism Closure Plot

**Purpose**: Visualize β vs Deg, probe AUC, and wrong-way metrics.

```bash
python scripts/plot_mechanism_closure.py
```

**Output**: `figures/fig_mechanism_closure.png`

---

## Workflow 5: LaTeX Compilation

**Compile manuscript**:
```bash
pdflatex main_paper.tex
bibtex main_paper
pdflatex main_paper.tex
pdflatex main_paper.tex
```

**Or use provided script**:
```bash
python check_latex.py
```

---

## Full Reproduction Pipeline

To reproduce all paper results from scratch:

```bash
# 1. Oracle test (3 min)
python scripts/validation_suite.py --mode oracle

# 2. Main sweep (4-5 hrs)
python scripts/run_paper.py --config configs/paper_run.json

# 3. Table 8 (4-5 hrs)
python scripts/run_regularization_control.py

# 4. Diagnostics (~30 min total)
python scripts/audit_identifiability.py
python scripts/regime_probe_z.py
python scripts/plot_mechanism_closure.py

# 5. Compile paper
pdflatex main_paper.tex
```

**Total Time**: ~9-11 hours on CPU, ~3-4 hours on GPU

---

## Troubleshooting

**Issue**: Import errors  
**Solution**: Ensure you're running from project root and `src/` is in PYTHONPATH

**Issue**: Out of memory  
**Solution**: Reduce `n_paths_train` and `n_paths_eval` in config

**Issue**: Results don't match paper  
**Solution**: Verify you're using `configs/paper_run.json` (not older configs)

---

## Directory After Workflow Completion

```
occam-hedge/
├── results/
│   ├── oracle_test_results.json
│   ├── paper_sweep_aggregated.csv
│   ├── regularization_control_summary.csv
│   └── ...
├── figures/
│   ├── fig_frontier_beta_sweep.png
│   ├── fig_mechanism_closure.png
│   └── ...
└── main_paper.pdf  (after LaTeX compilation)
```

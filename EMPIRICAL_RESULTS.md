# Empirical Results for Paper

This directory contains the essential research outputs for the paper's empirical section.

## 🚀 Reproducing Results

To generate all paper artifacts (tables, figures, JSONs) deterministically using the research-grade pipeline:

```bash
# Full reproduction (5 seeds, ~1-2 hours)
python scripts/run_paper.py --config configs/paper_run.json --seeds 5

# Fast test run (2 seeds, ~5 mins)
python scripts/run_paper.py --config configs/paper_test.json --seeds 2 --output_dir runs/paper_example
```

This will create a directory in `runs/paper_<timestamp>_<hash>/` containing:
- **aggregated CSV/JSON**: `paper_frontier.csv`, `paper_robust_curves.json`
- **final figures**: `fig_*.png`
- **metadata**: `config_resolved.json`, `metadata.json`

---

## 📊 Paper Figures

### 1. Robustness-Information Frontier (`fig_frontier_beta_sweep.png`)
- **X-axis**: Baseline Risk ($R_0$) - Standard Expected Shortfall (ES) at $\eta=0$
- **Y-axis**: Stressed Risk ($R_\eta$) - Robust ES at stress level $\eta=0.1$
- **Color**: Information Cost (KL divergence from prior)
- **Insight**: Lower information cost (darker) generally leads to lower stressed risk for the same baseline performance.
- **Band Version**: `fig_frontier_band.png` shows error bars (std dev across seeds).

### 2. Robust Risk Curves (`fig_robust_risk_vs_eta.png`)
- **X-axis**: Stress Level ($\eta$)
- **Y-axis**: Robust Risk ($R_\eta$)
- **Insight**: Microstructure-only representations degrade faster (steeper slope) than Combined or Greeks.
- **Band Version**: `fig_robust_risk_vs_eta_band.png` shows shaded error bands (std dev).

### 3. Regime Comparison (`fig_robust_compare_regime0.png`)
- Bar chart comparing Baseline vs Stressed performance for unpenalized policies ($\beta=0$).
- Shows the "fragility gap" for each representation.

### 4. Semantic Flip (`fig_semantic_flip_correlations.png`)
- Histograms of Volume vs Impact correlation in Regime 0 vs Regime 1.
- **Insight**: Correlation flips sign (+ vs -), proving semantic instability.

---

## 📈 Aggregated Data

- **`paper_frontier.csv`**:
  - `beta`: Information penalty coefficient
  - `R0`: Baseline Expected Shortfall (mean across seeds)
  - `R_stress_eta0p1`: Stressed Robust ES (mean across seeds)
  - `info_cost`: Information cost (mean)
  - `turnover`: Trading turnover
  - `*_std`: Standard deviation columns for error bars

- **`paper_robust_curves.json`**:
  - Detailed $R_\eta$ values for plotting full curves
  - Contains means and std devs for each $(\beta, \eta)$ point.

---

## 🧪 Metrics Explained

- **Baseline Risk ($R_0$)**: Standard CVaR/Expected Shortfall on the training distribution (Regime 0).
- **Stressed Risk ($R_\eta$)**: Worst-case Expected Shortfall within a KL-divergence ball of radius $\eta$ around the training distribution. This calculates risk under optimal adversarial shifts.
- **Information Cost**: The KL divergence between the policy's action distribution and a prior (Greeks-only baseline). Measures "how much" the model relies on extra features.

## 🛡️ Guardrails

The pipeline enforces:
1. **Artifact Consistency**: Figures are generated *only* from the saved aggregated data.
2. **$\beta$ Efficacy**: Runtime checks ensure that increasing $\beta$ significantly reduces information cost and changes model weights.
3. **Reproducibility**: `metadata.json` records git commit, platform info, and exact config.

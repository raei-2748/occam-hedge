# Empirical Results for Paper

This directory contains the essential research outputs for the paper's empirical section.

## Essential Files

### Figures (for Publication)
Located in `figures/`:

1. **`paper_417a4aa7_frontier_beta_sweep.png`** - Robustness-Information Frontier
   - Shows R₀ vs R_η trade-off colored by information cost
   - Used in: Main results section
   
2. **`paper_417a4aa7_robust_risk_vs_eta.png`** - Robust Risk Curves
   - Shows R_η vs η for different β values and representations
   - Used in: Robustness analysis section
   
3. **`paper_417a4aa7_robust_compare_regime0.png`** - Regime Comparison
   - Compares policy performance in baseline regime
   - Used in: Baseline validation section
   
4. **`paper_417a4aa7_semantic_flip_correlations.png`** - Semantic Flip Analysis
   - Shows volume-impact correlation flip between regimes
   - Used in: Mechanism validation section

### Results Data
Located in `runs/`:

1. **`paper_417a4aa7_frontier.csv`** - Main Results Table
   - Columns: representation, beta, R0, R_stress_eta0p1, info_cost, turnover, exec_cost
   - 30 rows: 3 representations × 10 β values
   
2. **`paper_417a4aa7_robust_curves.json`** - Detailed Robust Curves
   - R_η values for η ∈ [0.0, 0.05, 0.1, 0.2]
   - 9 curves: 3 representations × 3 β values
   
3. **`paper_417a4aa7_smoke_results.json`** - Baseline Validation
   - Quick sanity check results (greeks vs micro)
   
4. **`paper_417a4aa7_semantic_flip_summary.json`** - Regime Shift Evidence
   - Volume-impact correlations in both regimes

### Reproducibility Metadata
Located in `runs/20251231_124320_417a4aa7/`:

- `config_resolved.json` - Exact parameters used
- `metadata.json` - Git commit, timestamp, device info
- `metrics.jsonl` - Epoch-by-epoch training logs
- `results.json` - Full evaluation results

## Key Results Summary

### Main Finding
Higher β penalty → Lower information cost → Flatter robust risk curve

| Representation | β Range | Info Reduction | Robustness Gain |
|----------------|---------|----------------|-----------------|
| Greeks         | 0→1.0   | -14.2%         | Baseline stable |
| Micro          | 0→1.0   | -16.7%         | High fragility  |
| Combined       | 0→1.0   | -15.1%         | Best robustness |

### Paper Claims Supported

✓ **Claim 1:** Microstructure-heavy policies are more fragile under stress
  - Evidence: `robust_curves.json` shows micro has highest R_η growth

✓ **Claim 2:** Information penalty improves robustness
  - Evidence: `frontier.csv` shows β=1.0 has lower stress lift than β=0

✓ **Claim 3:** Combined representation balances baseline and robust performance
  - Evidence: `frontier.csv` shows combined achieves lowest R_stress_eta0p1

✓ **Claim 4:** Volume-impact correlation flips between regimes
  - Evidence: `semantic_flip_summary.json` shows ρ: -0.65 → +0.58

## Archived Files

Non-essential development artifacts moved to `archive/`:
- `archive/diagnostics/` - Development reports (BETA_*.md, AGENTS.md, etc.)
- `archive/old_runs/` - Intermediate/duplicate result files
- `archive/old_figures/` - Old plots without paper_ prefix

To delete archived files: `rm -rf archive/`

## Citation

When using these results, cite the configuration hash: `417a4aa7`

This ensures reproducibility with the exact parameters in `config_resolved.json`.

# Audit Report: Occam's Hedge Empirical Artifacts

## Summary of additions and fixes
- Implemented KL stress operator and Expected Shortfall (ES) utilities in `src/risk.py` using the DV dual and the convex ES loss `ell_q`.
- Added pytest coverage for KL stress monotonicity/convexity checks and ES grid consistency.
- Added scripts and diagnostics to generate required empirical artifacts, all writing outputs to `figures/` and `runs/`.
- Updated `src/experiment_occam.py` and `src/experiment_robust_compare.py` to use ES-consistent KL stress evaluation.

## Verified items (tests)
- `tests/test_risk_functions.py`: monotonicity in eta, robust risk >= mean loss, finite-difference convexity check, and ES vs brute-force q-grid consistency.

## Theory -> code -> artifact mapping

1) KL stress operator (outer KL)
- Code: `src/risk.py` (`robust_expectation`, `robust_risk_dv`, `logsumexp`).
- Tests: `tests/test_risk_functions.py`.
- Artifacts: used in `scripts/run_robustness_curves.py` -> `runs/robust_curves.json`, `figures/robust_risk_vs_eta.png`.

2) Expected Shortfall convex representation and KL-stressed ES
- Code: `src/risk.py` (`es_loss`, `optimize_q_es`, `expected_shortfall`, `robust_es_kl`).
- Tests: `tests/test_risk_functions.py` (ES vs brute-force q-grid).
- Artifacts: used in `scripts/run_beta_sweep.py` -> `runs/frontier.csv`, `figures/frontier_beta_sweep.png`.

3) Robust-risk curves (paper Figure 1 equivalent)
- Script: `scripts/run_robustness_curves.py`.
- Outputs: `runs/robust_curves.json`, `figures/robust_risk_vs_eta.png`.

4) Robustness-information frontier
- Script: `scripts/run_beta_sweep.py`.
- Outputs: `runs/frontier.csv`, `figures/frontier_beta_sweep.png`.

5) Semantic sign-flip diagnostics
- Script: `diagnostics/check_semantic_flip.py`.
- Outputs: `runs/semantic_flip_summary.json`, `figures/semantic_flip_correlations.png`.

6) Minimal reproducible run
- Script: `scripts/run_empirical_smoke_test.py`.
- Outputs: `runs/smoke_results.json`.

## Files touched
- `configs/paper_run.json`
- `src/paper_config.py`
- `src/risk.py`
- `src/experiment_occam.py`
- `src/experiment_robust_compare.py`
- `tests/test_risk_functions.py`
- `pytest.ini`
- `scripts/run_robustness_curves.py`
- `scripts/run_beta_sweep.py`
- `scripts/run_empirical_smoke_test.py`
- `diagnostics/check_semantic_flip.py`
- `.gitignore`
- `requirements.txt`
- `figures/paper_417a4aa7_robust_risk_vs_eta.png`
- `figures/paper_417a4aa7_frontier_beta_sweep.png`
- `figures/paper_417a4aa7_semantic_flip_correlations.png`
- `figures/paper_417a4aa7_robust_compare_regime0.png`
- `runs/paper_417a4aa7_robust_curves.json`
- `runs/paper_417a4aa7_frontier.csv`
- `runs/paper_417a4aa7_smoke_results.json`
- `runs/paper_417a4aa7_semantic_flip_summary.json`

## Paper-Run Configuration
- All paper figures and tables are generated from `configs/paper_run.json`.
- All stress tests are relative to the Regime 0 reference measure `P0`.
- ES 0.95 is used throughout for tail-risk evaluation.
- If `.venv/` exists, it is gitignored; reproducibility steps are `pip install -r requirements.txt`.

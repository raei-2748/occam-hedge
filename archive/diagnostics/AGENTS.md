# AGENTS.md

Purpose: This repo is a compact research prototype for Occam's Hedge under relative-entropy uncertainty. It demonstrates a regime-dependent microstructure sign flip and evaluates hedging policies under KL stress anchored on Regime 0.

Key scripts and outputs:
- `src/experiment_occam.py` -> robust curves, normalized curves, frontier plots, and `runs/occam_frontier.csv`
- `src/experiment_robust_compare.py` -> baseline KL-robust comparison plot
- Figures saved to `figures/`

Style rules:
- Keep code minimal and deterministic.
- Prefer plain functions, no heavy abstractions.
- Write outputs only to `figures/` and `runs/`.

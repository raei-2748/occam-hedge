# Occam's Hedge: Project Laws

## 1. The Inversion Law
- Every simulation step in `world.py` or `simulator.py` MUST compute and log a boolean `is_conflict_set`.
- A state is a conflict state if the optimal adjustment direction (aggressiveness) flips sign between Regime 0 and Regime 1.

## 2. The Identification Law
- The **Variance-Matched Control** is the primary identification pillar. 
- All Regime 1 evaluations must be explicitly labeled as either:
  1. `Variance-Matched`: Volume dispersion $\sigma_{v,1} = \sigma_{v,0}$.
  2. `OCST`: Overlap-Conditioned Stress Test.
- Agents must NOT use distributional moments (like volume variance) to detect regimes.

## 3. The Information Law
- No model architecture changes are permitted without defining a specific `latent_dim` (default: 8) and a `beta` penalty.
- Every policy must report its **Realized Information Cost** ($\mathcal{C}(\phi)$) in nats.

## 4. Evaluation Standards
- The primary risk metric is **Expected Shortfall** ($ES_{0.95}$).
- All results must include the **Wrong-Way Trading Score** ($W_r$) to verify directional instability.
- Multi-seed reporting (default: 5 seeds) is mandatory for statistical significance.
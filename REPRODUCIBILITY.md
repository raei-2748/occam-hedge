# Reproducibility Guide

## Quick Start

Reproduce the main mechanism closure figure with one command:

```bash
python scripts/run_paper.py \
  --config configs/paper_leaky.json \
  --seeds 3 \
  --micro_lags 4 \
  --leak_phi_r0 0.0 --leak_phi_r1 0.6 \
  --regime_mix_p 0.8
```

Reproduce the "Fortress Appendix" robustness results (Adversarial Simulator + Adaptive Beta):
```bash
python scripts/run_paper.py \
  --seeds 3 \
  --simulation_mode adversarial \
  --jump_intensity 0.1 \
  --adaptive_beta_gamma 0.5
```

Then generate the mechanism closure plot:
```bash
python scripts/plot_mechanism_closure.py
```

## Minimal Dependencies

```bash
pip install numpy pandas torch scikit-learn matplotlib
```

Or use the full requirements:
```bash
pip install -r requirements.txt
```

## Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--micro_lags` | 4 | Temporal history for regime inference |
| `--leak_phi_r1` | 0.6 | AR(1) autocorrelation in regime 1 |
| `--regime_mix_p` | 0.8 | Training dominated by regime 0 (80%) |

## Expected Results

With the recommended settings:
- **β=0**: AUC(Z) > 0.55 (regime info in latent space)
- **β→high**: AUC(Z) → 0.5 (regime info priced out)
- **Robustness gap** (matched vs broken) decreases with β

## Archived Artifacts

Old runs and logs are preserved in:
- `archive/runs/` — Historical experiment runs
- `archive/logs/` — Old log files
- `archive/old_results/` — Previous result CSVs

## File Structure

```
occam-hedge/
├── configs/paper_leaky.json    # Canonical reproduction config
├── scripts/run_paper.py        # Main entrypoint
├── scripts/plot_mechanism_closure.py
├── results/
│   ├── sweep_results.csv       # Latest sweep data
│   └── regime_probe_vs_beta.json
├── figures/
│   └── mechanism_closure.png   # Main output figure
└── src/                        # Core implementation
```

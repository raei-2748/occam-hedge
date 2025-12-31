# occams-hedge

Minimal research prototype for Occam's Hedge under relative-entropy uncertainty. The simulator exposes a regime-dependent sign flip in microstructure signals and evaluates hedging policies under KL stress anchored on Regime 0.

## 🚀 Quick Start

### Reproduce Paper Results
Generate all figures and tables with a single command:
```bash
python scripts/run_paper.py --config configs/paper_run.json --seeds 5
```
Results will be saved to `runs/paper_<timestamp>_<hash>/`.

### Run Tests
```bash
python -m pytest tests/
```

### Installation

## Run

Activate the local venv, then run from repo root:

```
source venv/bin/activate
PYTHONPATH=. python src/experiment_occam.py
PYTHONPATH=. python src/experiment_robust_compare.py
```

## Outputs

Figures are written to `figures/` and the frontier table to `runs/`:
- `figures/robust_curves_occam.png`
- `figures/robust_curves_occam_normalized.png`
- `figures/robust_frontier_occam_eta0p1.png`
- `figures/robust_frontier_occam_eta0p2.png`
- `figures/occam_diagnostic_turnover.png`
- `runs/occam_frontier.csv`

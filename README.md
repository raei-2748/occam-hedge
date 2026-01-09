# occam-hedge

Minimal research prototype for Occam's Hedge under relative-entropy uncertainty. The simulator exposes a regime-dependent sign flip in microstructure signals and evaluates hedging policies under KL stress anchored on Regime 0.

## 🚀 Quick Start

### Reproduce Paper Results
Generate all figures and tables with a single command:
```bash
python scripts/run_paper.py --config configs/paper_run.json
```

### Run with Mechanism Enabled (Competence Trap)
To demonstrate the VIB mechanism with temporal regime inference:
```bash
python scripts/run_paper.py \
    --micro_lags 4 \
    --leak_phi_r1 0.6 \
    --regime_mix_p 0.7 \
    --seeds 3
```

### Hierarchical Information Regularization
Execute a hierarchical beta sweep to isolate information channels:
```bash
python scripts/run_beta_sweep.py --config configs/hierarchical_run.json
```
This workflow anchors the price channel information penalty ($\beta_{\text{price}}$) while varying the microstructure penalty ($\beta_{\text{micro}}$).

### Run Validation Suite
Ensure core mechanisms (Oracle test, Competence Trap) are functioning:
```bash
python scripts/validation_suite.py --mode all
```

### Run Tests
```bash
python -m pytest tests/
```

## 🌍 World Configurations

| Mode | Description | Command Flags |
|------|-------------|---------------|
| No-leak | Original variance-matched (φ=0) | `--leak_phi_r0 0.0 --leak_phi_r1 0.0` |
| Leaky | AR(1) temporal signal for regime inference | `--leak_phi_r0 0.0 --leak_phi_r1 0.6 --micro_lags 4` |
| Oracle | Regime label given to policy | `--representation oracle` |

## 🏗 Repository Structure

- `src/`: Core implementation.
    - `features.py`: Centralized differentiable feature extraction with lagged micro support.
    - `policies.py`: Stochastic VIB policy architectures.
    - `experiment_occam.py`: Main training and hedging loops.
- `scripts/`: Research workflows and validation.
    - `validation_suite.py`: Oracle and smoke tests.
    - `run_paper.py`: Standard experiment runner with mechanism support.
    - `audit_identifiability.py`: Probe regime identifiability from observations.
    - `regime_probe_z.py`: Probe regime predictability from latent Z.
- `configs/`: Standardized JSON configurations for reproducibility.

## 📦 Installation

```bash
pip install -r requirements.txt
```


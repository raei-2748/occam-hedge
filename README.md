# occam-hedge

Minimal research prototype for Occam's Hedge under relative-entropy uncertainty. The simulator exposes a regime-dependent sign flip in microstructure signals and evaluates hedging policies under KL stress anchored on Regime 0.

## 🚀 Quick Start

### Reproduce Paper Results
Generate all figures and tables with a single command:
```bash
python scripts/run_paper.py --config configs/paper_run.json
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

## 🏗 Repository Structure

- `src/`: Core implementation.
    - `features.py`: [Refactored] Centralized differentiable feature extraction.
    - `policies.py`: Stochastic VIB policy architectures.
    - `experiment_occam.py`: Main training and hedging loops.
- `scripts/`: Research workflows and validation.
    - `validation_suite.py`: [Consolidated] Oracle and smoke tests.
    - `run_paper.py`: Standard experiment runner.
- `configs/`: Standardized JSON configurations for reproducibility.

## 📦 Installation

```bash
pip install -r requirements.txt
```

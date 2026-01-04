# Information Autopsy - Quick Start Guide

## What is the Information Autopsy?

The Information Autopsy is a diagnostic tool that tracks **which features** the VIB model discards as the information penalty (β) increases. This allows us to identify the "Occam's Hedge" sweet spot where the model learns to ignore confusing signals while retaining essential hedging information.

## Running the Analysis

### Step 1: Run Beta Sweep with Combined Representation

```bash
# Make sure you're in the project root
cd /Users/ray/Downloads/Documents/occam-hedge

# Activate virtual environment
source .venv/bin/activate  # or: .venv/bin/python

# Run beta sweep (this will save training histories)
.venv/bin/python scripts/run_beta_sweep.py --config configs/paper_run.json
```

**Important**: Make sure your `configs/paper_run.json` includes:
- `"representations": ["combined"]` (or at least includes "combined")
- `"beta_grid": [0.0, 0.01, 0.1, 1.0, 10.0]` (or similar range)

### Step 2: Visualize the Results

```bash
# Generate information autopsy diagnostic plot
.venv/bin/python scripts/plot_info_autopsy.py runs/
```

This will create `diagnostics/info_autopsy_diagnostic.png` with two plots:
1. **Left**: Individual feature costs (Delta vs Micro) across β
2. **Right**: Micro/Delta ratio showing feature selection

### Step 3: Interpret the Results

Look for the **three regimes**:

#### Low β (β ≈ 0):
```
info_dim_0 (Delta): HIGH
info_dim_1 (Micro): HIGH
```
- Model uses both features
- No information compression
- May overfit to regime-specific patterns

#### **Critical β (Sweet Spot)** (β ≈ 0.1-1.0):
```
info_dim_0 (Delta): HIGH
info_dim_1 (Micro): NEAR ZERO ← KEY DIAGNOSTIC!
```
- Model retains Delta (essential for hedging)
- Model discards Micro (regime-specific noise)
- **This is Occam's Hedge in action!**

#### High β (β >> 1):
```
info_dim_0 (Delta): NEAR ZERO
info_dim_1 (Micro): NEAR ZERO
```
- Model collapses to constant policy
- No hedging activity

## What the Features Mean

For `representation="combined"`:
- **info_dim_0 (Delta)**: Black-Scholes delta signal
  - Essential for hedging
  - Should remain high in the sweet spot
  
- **info_dim_1 (Micro)**: Microstructure/volume signal
  - Regime-specific (confusing across regimes)
  - Should drop to zero in the sweet spot

## Expected Output

### Training History Format
Each beta run saves a JSON file like:
```json
[
  {
    "epoch": 0,
    "loss_obj": 0.5843,
    "q": -0.1234,
    "info_total": 0.2008,
    "mode": "warmup",
    "info_dim_0": 0.1325,
    "info_dim_1": 0.0683
  },
  ...
]
```

### Diagnostic Plot
The plot will show:
- **Sweet spot annotation**: Vertical line at optimal β
- **Color-coded lines**: Delta (blue), Micro (purple), Total (orange)
- **Log-log scale**: To visualize wide β range
- **Ratio plot**: Shows when Micro becomes negligible relative to Delta

## Troubleshooting

### No training_history.json files found
- Make sure you ran the beta sweep with the updated script
- Check that `runs/` directory exists and contains subdirectories

### Plot shows unexpected patterns
- Verify `representation="combined"` was used
- Check that beta grid spans sufficient range (e.g., 0.01 to 10)
- Ensure training converged (check loss_obj in history)

### All info costs are zero
- Beta might be too high
- Check model initialization and training config

## Next Steps After Diagnosis

Once you identify the sweet spot β:

1. **Validate Robustness**: Check that this β gives best ES_0.95 on stress test
2. **Compare to Baselines**: Verify it outperforms fixed Delta and pure Deep Hedging
3. **Write Up Results**: Document the feature selection behavior in paper
4. **Create Visualizations**: Use the diagnostic plots in your paper/presentation

## Files Modified

- `src/experiment_occam.py`: Added per-channel KL tracking
- `scripts/run_beta_sweep.py`: Now saves training histories
- `scripts/plot_info_autopsy.py`: Visualization tool (new)
- `tests/test_info_autopsy.py`: Unit tests (new)

## Citation

If you use this diagnostic in your research:

```
The Information Autopsy reveals that at the optimal β, the VIB model 
selectively discards the microstructure signal (info_dim_1 → 0) while 
retaining the Delta signal (info_dim_0 > 0), demonstrating learned 
feature selection that aligns with the Occam's Hedge principle.
```

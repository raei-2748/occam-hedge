# β-Has-No-Effect: Deliverables Summary

## 1. Root Cause Identified

**Where:** Pre-Phase 1 `train_weights` function in `src/experiment_occam.py` (commit `6b91122`)

**Bug:** Grid search over 8 discrete weight values with weak β penalty caused all β ∈ [0, 1] to select the same grid point.

## 2. Evidence

### Direct Evidence (Old Results)
```bash
$ cat runs/paper_417a4aa7_frontier.csv | grep greeks | awk -F',' '{print $2,$5}'
0.0 11.751372018618262
0.01 11.751372018618262
...
1.0 11.751372018618262  # All identical!
```

### Fingerprint Test (New Code Works)
```bash
$ .venv/bin/python tests/test_model_fingerprints.py
β=0.0:  fingerprint=0bf7ce82e0a3, weights=[0.936]
β=0.2:  fingerprint=4a3302260bfa, weights=[0.921]  # ✓ Different
β=0.6:  fingerprint=6e947a0dfee8, weights=[0.890]
```

### Logs from Phase 1 Torch Training
```
β=0.0:  final_weights=[0.9358], info_cost=16.079
β=1.0:  final_weights=[0.8669], info_cost=13.798  # ✓ 14% reduction
```

## 3. The Fix

**Already implemented in Phase 1 (commit `6a1c443`):**
- Replaced discrete grid search with Torch gradient descent
- Continuous weight optimization allows β to smoothly affect training
- Warm-start phase stabilizes ES threshold before robust training

**No additional changes needed** - Phase 1 already fixed this.

## 4. Guardrail Test

**File:** `tests/test_beta_guardrail.py`

**Purpose:** Fails loudly if β stops affecting training (e.g., due to regression)

**Test Logic:**
1. Train with β=0 and β=100 (extreme contrast)
2. Assert weights differ by > 0.01 (L2 norm)
3. Assert info_cost reduces by > 5%

**Status:** ✓ PASSING
```
β=0:   weights=[0.933], info_cost=7.998
β=100: weights=[0.434], info_cost=1.731
→ Weight diff: 0.499, Info reduction: 78.4%  ✓
```

**Runtime:** ~60 seconds (uses minimal data for speed)

## 5. Action Items

### Immediate
- ✓ Diagnostic complete
- ✓ Guardrail test added
- ⏳ Re-running `run_beta_sweep.py` to generate fresh results

### Follow-Up
- Run `scripts/run_robustness_curves.py` to update robust risk curves
- Archive old results to `runs/archive_old_grid_search/`
- Update paper if figures used old results

## Files Added/Modified

### New Files
- `tests/test_beta_guardrail.py` - Production guardrail (auto-fail if β broken)
- `tests/test_beta_effect.py` - Quick diagnostic (β=0 vs β=100)
- `tests/test_model_fingerprints.py` - Model uniqueness checker
- `tests/debug_beta_sweep.py` - Detailed trace for debugging
- `BETA_DIAGNOSTIC_REPORT.md` - Full technical report (this doc's parent)

### No Code Changes
Phase 1 already fixed the underlying issue. All tests pass with current code.

## Failure Modes Ruled Out

| Mode | Test | Result |
|------|------|--------|
| A. β not in loss | Check loss computation | ✗ β is in loss |
| B. Models not retrained | Fingerprint test | ✗ Models differ |
| C. Checkpoints overwritten | Check file I/O | ✗ N/A (no checkpoints) |
| D. Wrong model in eval | Trace eval flow | ✗ Correct model used |
| **✓ E. Grid search** | **Compare old/new code** | **✓ ROOT CAUSE** |

## Next Steps

1. Wait for `run_beta_sweep.py` to finish (~5-10 minutes)
2. Verify new CSV shows varying info_cost across β
3. Commit guardrail tests
4. Update paper figures if needed

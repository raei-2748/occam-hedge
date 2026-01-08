#!/bin/bash
# Helper script to run post-analysis diagnostics for Occam's Hedge Production Run

RUN_DIR=$1
if [ -z "$RUN_DIR" ]; then
    echo "Usage: $0 <run_directory>"
    echo "Example: $0 runs/paper_20260107_..."
    exit 1
fi

echo "========================================"
echo "Post-Analysis for: $RUN_DIR"
echo "========================================"

# 1. Aggregate Results Table
echo "[1] Aggregating Results Table..."
venv/bin/python aggregate_results.py --results_dir "$RUN_DIR"

# 2. Information Autopsy (Diagnostic Inversion)
# Note: diagnose_inversion.py runs a standalone simulation.
echo "[2] Running Information Autopsy (Semantic Inversion Check)..."
venv/bin/python scripts/diagnose_inversion.py

# 3. Policy Surface Visualization
echo "[3] Generating Policy Surface Visualization..."
# Visualize Seed 0, Low Beta (0.01) vs High Beta (10.0) or (0.0 vs 10.0)
# We select 0.01 (Info Constraint onset) vs 10.0 (Collapse) or 0.0 (Unconstrained).
# Let's use 0.0100 and 10.0000 as typical "Low vs High" comparison points.
LOW_BETA_DIR="$RUN_DIR/checkpoints/combined_beta_0.0100_seed_0"
HIGH_BETA_DIR="$RUN_DIR/checkpoints/combined_beta_10.0000_seed_0"

if [ ! -d "$LOW_BETA_DIR" ]; then
    echo "Warning: $LOW_BETA_DIR not found. Trying 0.0000..."
    LOW_BETA_DIR="$RUN_DIR/checkpoints/combined_beta_0.0000_seed_0"
fi

if [ -d "$LOW_BETA_DIR" ] && [ -d "$HIGH_BETA_DIR" ]; then
    venv/bin/python scripts/visualize_policy_surface.py \
        --low_beta_dir "$LOW_BETA_DIR" \
        --high_beta_dir "$HIGH_BETA_DIR" \
        --representation combined \
        --output "$RUN_DIR/post_analysis/policy_surfaces.png"
else
    echo "Error: Checkpoint directories not found. Cannot generate policy surfaces."
    echo "Checked: $LOW_BETA_DIR and $HIGH_BETA_DIR"
fi

echo "========================================"
echo "Analysis Complete."

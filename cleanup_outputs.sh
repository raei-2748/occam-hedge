#!/bin/bash
# Clean up research outputs, keeping only paper-essential files

set -e

echo "Cleaning research outputs for paper submission..."
echo "Keeping only essential empirical section files"
echo ""

# Navigate to project root
cd "$(dirname "$0")"

# Create archive directory for non-essential files
mkdir -p archive/old_runs
mkdir -p archive/old_figures
mkdir -p archive/diagnostics

echo "=== ARCHIVING NON-ESSENTIAL FILES ==="

# Archive development diagnostic reports
echo "→ Archiving diagnostic reports..."
mv BETA_COMPLETION_REPORT.md archive/diagnostics/ 2>/dev/null || true
mv BETA_DIAGNOSTIC_REPORT.md archive/diagnostics/ 2>/dev/null || true
mv BETA_FIX_SUMMARY.md archive/diagnostics/ 2>/dev/null || true
mv BETA_VERIFICATION.md archive/diagnostics/ 2>/dev/null || true
mv AGENTS.md archive/diagnostics/ 2>/dev/null || true
mv AUDIT_REPORT.md archive/diagnostics/ 2>/dev/null || true

# Archive old/duplicate result files in runs/
echo "→ Archiving old run files..."
cd runs/
mv frontier.csv ../archive/old_runs/ 2>/dev/null || true
mv occam_frontier.csv ../archive/old_runs/ 2>/dev/null || true
mv robust_curves.json ../archive/old_runs/ 2>/dev/null || true
mv semantic_flip_summary.json ../archive/old_runs/ 2>/dev/null || true
mv smoke_results.json ../archive/old_runs/ 2>/dev/null || true
cd ..

# Archive old/duplicate figures
echo "→ Archiving old/duplicate figures..."
cd figures/
mv frontier_beta_sweep.png ../archive/old_figures/ 2>/dev/null || true
mv occam_diagnostic_turnover.png ../archive/old_figures/ 2>/dev/null || true
mv robust_compare_regime0.png ../archive/old_figures/ 2>/dev/null || true
mv robust_compare_regime1.png ../archive/old_figures/ 2>/dev/null || true
mv robust_curves_occam.png ../archive/old_figures/ 2>/dev/null || true
mv robust_curves_occam_normalized.png ../archive/old_figures/ 2>/dev/null || true
mv robust_frontier_occam.png ../archive/old_figures/ 2>/dev/null || true
mv robust_frontier_occam_eta0p1.png ../archive/old_figures/ 2>/dev/null || true
mv robust_frontier_occam_eta0p2.png ../archive/old_figures/ 2>/dev/null || true
mv robust_risk_curve.png ../archive/old_figures/ 2>/dev/null || true
mv robust_risk_vs_eta.png ../archive/old_figures/ 2>/dev/null || true
mv semantic_flip_correlations.png ../archive/old_figures/ 2>/dev/null || true
mv signflip_heston_regime0.png ../archive/old_figures/ 2>/dev/null || true
mv signflip_heston_regime1.png ../archive/old_figures/ 2>/dev/null || true
mv signflip_regime0.png ../archive/old_figures/ 2>/dev/null || true
mv signflip_regime1.png ../archive/old_figures/ 2>/dev/null || true
cd ..

echo ""
echo "=== ESSENTIAL FILES RETAINED ==="
echo ""
echo "Figures (for paper):"
ls -1 figures/
echo ""
echo "Results (for paper):"
ls -1 runs/*.{csv,json} 2>/dev/null || echo "(none)"
echo ""
echo "Run metadata (for reproducibility):"
ls -1d runs/2025* 2>/dev/null || echo "(none)"
echo ""
echo "=== CLEANUP COMPLETE ==="
echo ""
echo "Essential files: runs/paper_417a4aa7_*, figures/paper_417a4aa7_*"
echo "Archived files: archive/"
echo ""
echo "To delete archived files permanently: rm -rf archive/"

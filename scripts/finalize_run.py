
import json
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from plotting import (
    plot_frontier_beta_sweep,
    plot_robust_risk_vs_eta,
    plot_semantic_flip_correlations,
    plot_robust_compare_regime0,
    plot_turnover_concentration
)
import shutil

def finalize(run_dir_path):
    run_dir = Path(run_dir_path)
    print(f"Finalizing run: {run_dir}")
    
    # 1. Load Data
    raw_csv = run_dir / "raw_results.csv"
    if not raw_csv.exists():
        print(f"Error: {raw_csv} not found")
        return
    
    df = pd.read_csv(raw_csv)
    
    with open(run_dir / "config_resolved.json", "r") as f:
        cfg = json.load(f)
    
    with open(run_dir / "paper_semantic_flip_summary.json", "r") as f:
        semantic_flips = json.load(f)

    # Re-run aggregation if needed (to be safe)
    beta_grid_raw = cfg["beta_grid"]
    frontier_df = df[df["beta"].isin(beta_grid_raw)].copy()
    agg_cols = ["R0", "R_stress_eta0p1", "info_cost", "turnover", "exec_cost", "wrong_way_0", "wrong_way_1"]
    frontier_agg = frontier_df.groupby(["representation", "beta"])[agg_cols].agg(["mean", "std"]).reset_index()
    frontier_agg.columns = ['_'.join(col).strip() if col[1] else col[0] for col in frontier_agg.columns.values]
    rename_map = {f"{c}_mean": c for c in agg_cols}
    rename_map.update({f"{c}_std": f"{c}_std" for c in agg_cols})
    frontier_agg.rename(columns=rename_map, inplace=True)
    frontier_agg.to_csv(run_dir / "paper_frontier.csv", index=False)

    # Robust Curves Data
    beta_grid_curves_raw = cfg.get("beta_grid_curves", [])
    curves_df = df[df["beta"].isin(beta_grid_curves_raw)].copy()
    curves_export = []
    if not curves_df.empty:
        # Note: We need the etas from the config
        eta_grid_curves = cfg["eta_grid"]
        for (rep, beta), group in curves_df.groupby(["representation", "beta"]):
            import numpy as np
            # R_eta_curve is stored as a stringified list in CSV
            import ast
            stack = np.stack([ast.literal_eval(v) for v in group["R_eta_curve"].values])
            means = np.mean(stack, axis=0)
            stds = np.std(stack, axis=0)
            curves_export.append({
                "representation": rep,
                "beta": float(beta),
                "etas": eta_grid_curves,
                "R_eta_mean": means.tolist(),
                "R_eta_std": stds.tolist()
            })
    
    with open(run_dir / "paper_robust_curves.json", "w") as f:
        json.dump({"results": curves_export}, f, indent=2)

    # Diagnostic results for turnover concentration (simulated from raw results isn't easy, 
    # but run_paper saved seed checkpoints... wait, run_paper saved diag_results in memory.
    # If we want Fig 5, we might need to re-run diagnostics. 
    # For now, let's just finish the ones we have.)

    print("\nGenerating Figures...")
    
    # Fig 1: Frontier
    plot_frontier_beta_sweep(frontier_agg, run_dir / "fig_frontier_beta_sweep.png")
    plot_frontier_beta_sweep(frontier_agg, run_dir / "fig_frontier_band.png")
    
    # Fig 2: Robust Curves
    plot_robust_risk_vs_eta(curves_export, run_dir / "fig_robust_risk_vs_eta.png", use_bands=False)
    plot_robust_risk_vs_eta(curves_export, run_dir / "fig_robust_risk_vs_eta_band.png", use_bands=True)
    
    # Fig 3: Regime Comparison
    plot_robust_compare_regime0(frontier_agg, run_dir / "fig_robust_compare_regime0.png")
    
    # Fig 4: Semantic Flip
    plot_semantic_flip_correlations(semantic_flips, run_dir / "fig_semantic_flip_correlations.png")

    # Fig 5: Turnover Concentration (Skip if we don't have diag_results saved)
    # run_paper didn't save diag_results to disk, only used it in memory.

    # Copy to figures/
    final_figures_dir = ROOT / "figures"
    final_figures_dir.mkdir(exist_ok=True)
    print("\nUpdating stable figures in figures/...")
    for fig_file in run_dir.glob("fig_*.png"):
        dest = final_figures_dir / fig_file.name
        shutil.copy2(fig_file, dest)
        print(f"  -> {dest.name}")

    # Print a summary table (R0 and R_stress)
    print("\nSummary of Results (Mean):")
    print(frontier_agg[["representation", "beta", "R0", "R_stress_eta0p1"]])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    finalize(args.run_dir)


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from pathlib import Path

def plot_frontier_beta_sweep(frontier_data: pd.DataFrame, out_path: Path):
    """
    Plots the robustness-information frontier (R0 vs R_stress) colored by info cost.
    
    frontier_data columns: representation, beta, R0, R_stress_eta0p1, info_cost, R0_std, R_stress_std
    """
    representations = frontier_data["representation"].unique()
    rep_markers = {"greeks": "o", "micro": "s", "combined": "^"}
    
    plt.figure(figsize=(6, 4))
    
    for rep in representations:
        subset = frontier_data[frontier_data["representation"] == rep]
        
        # If we have distinct beta points, we plot them as scatter
        plt.scatter(
            subset["R0"],
            subset["R_stress_eta0p1"],
            c=subset["info_cost"],
            cmap="viridis",
            marker=rep_markers.get(rep, "o"),
            label=rep,
            alpha=0.8,
            zorder=2
        )
        
        # If std columns exist and are non-zero, plot error bars
        if "R0_std" in subset.columns and "R_stress_std" in subset.columns:
            plt.errorbar(
                subset["R0"],
                subset["R_stress_eta0p1"],
                xerr=subset["R0_std"],
                yerr=subset["R_stress_std"],
                fmt='none',
                ecolor='gray',
                alpha=0.3,
                zorder=1
            )

    plt.colorbar(label="KL_inner (info cost)")
    plt.xlabel(r"$R_0$")
    plt.ylabel(r"$R_\eta$ ($\eta=0.1$)")
    plt.title("Robustness-information frontier")
    plt.legend(fontsize=8, loc="best")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved figure: {out_path}")

def plot_robust_risk_vs_eta(curves_data: list, out_path: Path, use_bands=False):
    """
    Plots robust risk curves (eta vs R_eta).
    
    curves_data: list of dicts, each having:
       representation, beta, etas, R_eta_mean, R_eta_std (optional)
    """
    # Organize by representation
    reps = sorted(list(set(d["representation"] for d in curves_data)))
    
    if not reps:
        print("No data for robustness curves plot.")
        return
    
    fig, axes = plt.subplots(1, len(reps), figsize=(4 * len(reps), 4), sharey=True, squeeze=False)
    axes = axes.flatten()
        
    for i, rep in enumerate(reps):
        ax = axes[i]
        rep_data = [d for d in curves_data if d["representation"] == rep]
        
        # Sort by beta for consistent legend
        rep_data.sort(key=lambda x: x["beta"])
        
        for item in rep_data:
            etas = np.array(item["etas"])
            means = np.array(item["R_eta_mean"])
            beta = item["beta"]
            
            ax.plot(etas, means, marker="o", label=f"beta={beta:g}", alpha=0.85)
            
            if use_bands and "R_eta_std" in item:
                stds = np.array(item["R_eta_std"])
                ax.fill_between(etas, means - stds, means + stds, alpha=0.2)

        ax.set_title(rep)
        ax.set_xlabel(r"$\eta$")
        if i == 0:
            ax.set_ylabel(r"$R_\eta$")
        ax.legend(fontsize=7, ncol=2, loc="best")
        ax.grid(True, linestyle='--', alpha=0.3)

    fig.suptitle("Robust risk curves (Regime 0)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved figure: {out_path}")

def plot_semantic_flip_correlations(flip_data: dict, out_path: Path):
    """
    Plots correlation histograms or bars for semantic flip.
    flip_data: {
        "n_trials": N,
        "corr_regime0": [...],
        "corr_regime1": [...]
    }
    """
    reg0 = flip_data["corr_regime0"]
    reg1 = flip_data["corr_regime1"]
    
    plt.figure(figsize=(6, 4))
    plt.hist(reg0, alpha=0.7, label="Regime 0 (Calm)", color='blue', bins=10)
    plt.hist(reg1, alpha=0.7, label="Regime 1 (Stress)", color='red', bins=10)
    
    plt.xlabel("Correlation (Volume vs |Returns|)") # Or whatever the semantic flip is
    plt.ylabel("Frequency")
    plt.title(f"Semantic Flip: Volume-Impact Correlation (N={flip_data.get('n_trials', len(reg0))})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved figure: {out_path}")

def plot_robust_compare_regime0(frontier_data: pd.DataFrame, out_path: Path):
    """
    Bar plot comparing baseline vs stressed risk for different representations (at beta=0 or similar).
    
    We'll pick the 'best' beta or beta=0 roughly.
    """
    # Filter for beta=0 (no penalty)
    subset = frontier_data[frontier_data["beta"] == 0.0]
    
    if subset.empty:
        print("No beta=0 data for comparison plot.")
        return

    representations = subset["representation"].unique()
    
    x = np.arange(len(representations))
    width = 0.35
    
    r0_vals = []
    rstress_vals = []
    r0_err = []
    rstress_err = []
    
    for rep in representations:
        row = subset[subset["representation"] == rep].iloc[0]
        r0_vals.append(row["R0"])
        rstress_vals.append(row["R_stress_eta0p1"])
        r0_err.append(row.get("R0_std", 0))
        rstress_err.append(row.get("R_stress_std", 0))
        
    plt.figure(figsize=(6, 4))
    plt.bar(x - width/2, r0_vals, width, label='Baseline R0', yerr=r0_err, capsize=5)
    plt.bar(x + width/2, rstress_vals, width, label='Stressed R_eta (0.1)', yerr=rstress_err, capsize=5)
    
    plt.ylabel('Risk (Expected Shortfall)')
    plt.title('Baseline vs Stressed Risk (beta=0)')
    plt.xticks(x, representations)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved figure: {out_path}")
    
def plot_turnover_concentration(diag_data: list, out_path: Path):
    """
    Plots turnover concentration by volume quintile under stress.
    diag_data: list of dicts {rep, beta, volume, turnover} (arrays)
    """
    import pandas as pd
    
    # Flatten data
    rows = []
    for item in diag_data:
        rep = item["representation"]
        beta = item["beta"]
        vols = np.array(item["volume"])
        turns = np.array(item["turnover"])
        
        # Create a DF
        df_tmp = pd.DataFrame({"vol": vols, "turn": turns})
        # Quintiles
        try:
            df_tmp["quintile"] = pd.qcut(df_tmp["vol"], 5, labels=False)
        except ValueError:
            # Handle degenerate case if vol constant
            df_tmp["quintile"] = 0
            
        # Mean turnover per quintile
        means = df_tmp.groupby("quintile")["turn"].mean()
        
        for q, val in means.items():
            rows.append({
                "representation": rep,
                "beta": beta,
                "quintile": q,
                "turnover": val
            })
            
    df_agg = pd.DataFrame(rows)
    # Average across seeds if multiple entries per (rep, beta)
    df_plot = df_agg.groupby(["representation", "beta", "quintile"])["turnover"].mean().reset_index()
    
    # Plotting
    reps = df_plot["representation"].unique()
    if len(reps) == 0:
        print("No data for turnover concentration plot.")
        return
    fig, axes = plt.subplots(1, len(reps), figsize=(4*len(reps), 4), sharey=True, squeeze=False)
    axes = axes.flatten()
    
    for i, rep in enumerate(reps):
        ax = axes[i]
        subset = df_plot[df_plot["representation"] == rep]
        
        # Plot line for each beta
        betas = sorted(subset["beta"].unique())
        for beta in betas:
            line = subset[subset["beta"] == beta]
            ax.plot(line["quintile"], line["turnover"], marker='o', label=f"beta={beta}")
            
        ax.set_title(rep)
        ax.set_xlabel("Volume Quintile (0=Low, 4=High)")
        if i==0: ax.set_ylabel("Mean Turnover")
        ax.legend(fontsize='small')
        ax.grid(True, alpha=0.3)
        
    fig.suptitle("Turnover Concentration by Volume (Stressed Regime)")
    fig.tight_layout()
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved figure: {out_path}")

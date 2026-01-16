
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from pathlib import Path
import torch

# --- Publication grade aesthetics ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "text.usetex": False, # Use True if user has TeX installed, but safer False with good fonts
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.titlesize": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 300,
})

def plot_frontier_beta_sweep(frontier_data: pd.DataFrame, out_path: Path):
    """
    Plots the robustness-information frontier (R0 vs R_stress) colored by info cost.
    """
    representations = frontier_data["representation"].unique()
    rep_markers = {"greeks": "o", "micro": "s", "combined": "^"}
    rep_colors = {"greeks": "#440154", "micro": "#21918c", "combined": "#fde725"} # Viridis extremes
    
    plt.figure(figsize=(5, 4))
    
    for rep in representations:
        subset = frontier_data[frontier_data["representation"] == rep]
        
        # Plot markers
        sc = plt.scatter(
            subset["R0"],
            subset["R_stress_eta0p1"],
            c=subset["info_cost"],
            cmap="magma", # Magma is sleeker for finance
            marker=rep_markers.get(rep, "o"),
            label=rep.capitalize(),
            alpha=0.9,
            edgecolors='k',
            linewidths=0.5,
            s=50,
            zorder=3
        )
        
        # Plot error bars
        if "R0_std" in subset.columns and "R_stress_std" in subset.columns:
            plt.errorbar(
                subset["R0"],
                subset["R_stress_eta0p1"],
                xerr=subset["R0_std"],
                yerr=subset["R_stress_std"],
                fmt='none',
                ecolor='gray',
                alpha=0.2,
                zorder=2
            )

    cbar = plt.colorbar(sc)
    cbar.set_label("Information Cost (KL bits)", rotation=270, labelpad=15)
    
    plt.xlabel(r"Baseline Risk ($R_0$)")
    plt.ylabel(r"Stressed Risk ($R_1$)")
    plt.title("The Occam Frontier")
    plt.legend(frameon=True, fancybox=True, shadow=True)
    
    # Add a diagonal line for parity
    lims = [
        min(plt.xlim()[0], plt.ylim()[0]),
        max(plt.xlim()[1], plt.ylim()[1]),
    ]
    plt.plot(lims, lims, 'k--', alpha=0.5, zorder=1, label='Regime Parity')
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {out_path}")

def plot_robust_risk_vs_eta(curves_data: list, out_path: Path, use_bands=False):
    """
    Plots robust risk curves (eta vs R_eta).
    """
    reps = sorted(list(set(d["representation"] for d in curves_data)))
    if not reps: return
    
    fig, axes = plt.subplots(1, len(reps), figsize=(4 * len(reps), 4), sharey=True, squeeze=False)
    axes = axes.flatten()
    
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, 6))
        
    for i, rep in enumerate(reps):
        ax = axes[i]
        rep_data = [d for d in curves_data if d["representation"] == rep]
        rep_data.sort(key=lambda x: x["beta"])
        
        for j, item in enumerate(rep_data):
            etas = np.array(item["etas"])
            means = np.array(item["R_eta_mean"])
            beta = item["beta"]
            
            color = colors[j % len(colors)]
            ax.plot(etas, means, marker=".", markersize=4, label=fr"$\beta={beta:g}$", alpha=0.9, linewidth=1.5, color=color)
            
            if use_bands and "R_eta_std" in item:
                stds = np.array(item["R_eta_std"])
                ax.fill_between(etas, means - stds, means + stds, alpha=0.15, color=color, linewidth=0)

        ax.set_title(rep.capitalize())
        ax.set_xlabel(r"Stress Severity ($\eta$)")
        if i == 0:
            ax.set_ylabel(r"Expected Shortfall ($ES_{0.95}$)")
        ax.legend(loc="upper left", ncol=2)
        ax.set_facecolor('#fdfdfd')

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def plot_semantic_flip_correlations(flip_data: dict, out_path: Path):
    """
    Plots correlation histograms for semantic flip with publication grade aesthetics.
    """
    reg0 = flip_data["corr_regime0"]
    reg1 = flip_data["corr_regime1"]
    
    plt.figure(figsize=(5, 3.5))
    plt.hist(reg0, alpha=0.6, label="Regime 0 (Calm)", color='#2166ac', bins=15, edgecolor='white', density=True)
    plt.hist(reg1, alpha=0.6, label="Regime 1 (Stress)", color='#b2182b', bins=15, edgecolor='white', density=True)
    
    plt.axvline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    plt.xlabel(r"Correlation (Volume $\leftrightarrow$ Marginal Impact)")
    plt.ylabel("Density")
    plt.title("Semantic Inversion Diagnostic")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def plot_policy_surface(model, representation: str, beta: float, out_path: Path, K=100.0, T=1.0, vol_hat=0.2, micro_lags=0, include_prev_action=False):
    """
    Generates a 'Thermomap' (Policy Surface) showing position as a function of Price and Volume.
    """
    from experiment_occam import occam_features_torch
    
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    
    # Grid
    n_grid = 50
    S_grid = np.linspace(80, 120, n_grid)
    V_grid = np.linspace(0.1, 5.0, n_grid)
    SS, VV = np.meshgrid(S_grid, V_grid)
    
    positions = np.zeros_like(SS)
    
    with torch.no_grad():
        for i in range(n_grid):
            for j in range(n_grid):
                s = SS[i, j]
                v = VV[i, j]
                
                # Zero out history/prev action for simplicity in surface plot
                tau = T/2 # Mid-point of episode
                
                if micro_lags > 0:
                    V_history = torch.zeros(1, micro_lags)
                else:
                    V_history = None
                    
                a_prev = torch.zeros(1) if include_prev_action else None
                
                feats = occam_features_torch(
                    representation,
                    torch.tensor([s], dtype=torch.float32),
                    torch.tensor([tau], dtype=torch.float32),
                    torch.tensor([v], dtype=torch.float32),
                    K,
                    vol_hat,
                    micro_lags=micro_lags,
                    include_prev_action=include_prev_action,
                    V_history=V_history,
                    a_prev=a_prev
                )
                
                action, _, _ = model(feats)
                positions[i, j] = action.item()

    plt.figure(figsize=(6, 4.5))
    im = plt.pcolormesh(SS, VV, positions, cmap='RdYlBu_r', shading='gouraud')
    plt.colorbar(im, label="Position ($a_t$)")
    
    plt.contour(SS, VV, positions, colors='k', alpha=0.2, linewidths=0.5)
    
    plt.xlabel("Stock Price ($S_t$)")
    plt.ylabel("Volume Proxy ($V_t$)")
    plt.title(fr"Policy Surface Heatmap ($\beta={beta:g}$)")
    
    # Add Strike line
    plt.axvline(K, color='k', linestyle=':', alpha=0.4, label='Strike')
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Thermomap position plot: {out_path}")

def plot_robust_compare_regime0(frontier_data: pd.DataFrame, out_path: Path):
    """Updated bar plot for comparing R0 vs R1 at beta=0."""
    subset = frontier_data[frontier_data["beta"] == 0.0]
    if subset.empty: return

    reps = subset["representation"].unique()
    x = np.arange(len(reps))
    width = 0.35
    
    plt.figure(figsize=(5, 4))
    plt.bar(x - width/2, subset["R0"], width, label='Normal ($R_0$)', color='#2c7bb6', alpha=0.8)
    plt.bar(x + width/2, subset["R_stress_eta0p1"], width, label='Stress ($R_1$)', color='#d7191c', alpha=0.8)
    
    plt.ylabel('Expected Shortfall ($ES_{0.95}$)')
    plt.title('Vulnerability Trace ($\beta=0$)')
    plt.xticks(x, [r.capitalize() for r in reps])
    plt.legend(frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_turnover_concentration(diag_data: list, out_path: Path):
    """Updated turnover concentration plot."""
    rows = []
    for item in diag_data:
        rep = item["representation"]
        beta = item["beta"]
        vols = np.array(item["volume"])
        turns = np.array(item["turnover"])
        df_tmp = pd.DataFrame({"vol": vols, "turn": turns})
        try: df_tmp["quintile"] = pd.qcut(df_tmp["vol"], 5, labels=False)
        except ValueError: df_tmp["quintile"] = 0
        means = df_tmp.groupby("quintile")["turn"].mean()
        for q, val in means.items():
            rows.append({"rep": rep, "beta": beta, "quintile": q, "turnover": val})
            
    df_plot = pd.DataFrame(rows).groupby(["rep", "beta", "quintile"])["turnover"].mean().reset_index()
    reps = df_plot["rep"].unique()
    if not len(reps): return
    
    fig, axes = plt.subplots(1, len(reps), figsize=(4*len(reps), 4), sharey=True, squeeze=False)
    axes = axes.flatten()
    colors = plt.cm.viridis(np.linspace(0, 0.8, 5))
    
    for i, rep in enumerate(reps):
        ax = axes[i]
        subset = df_plot[df_plot["rep"] == rep]
        betas = sorted(subset["beta"].unique())
        for j, beta in enumerate(betas):
            line = subset[subset["beta"] == beta]
            ax.plot(line["quintile"], line["turnover"], marker='o', label=fr"$\beta={beta:g}$", color=colors[j%len(colors)], linewidth=1.5)
            
        ax.set_title(rep.capitalize())
        ax.set_xlabel("Volume Quintile (L $\to$ H)")
        if i==0: ax.set_ylabel("Mean Trading Intensity")
        ax.legend(fontsize='x-small', ncol=1)

    fig.tight_layout()
    plt.savefig(out_path)
    plt.close()

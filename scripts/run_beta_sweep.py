import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from experiment_occam import hedge_on_paths, train_model
from world import simulate_heston_signflip
from risk import robust_es_kl
from paper_config import load_config, run_id_from_config
from utils import set_seeds
from policies import FactorizedVariationalPolicy, policy_delta_hedge
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "paper_run.json"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_id = run_id_from_config(cfg)

    base_seed = int(cfg["seed_train"])
    eval_seed = int(cfg["seed_eval"])
    set_seeds(base_seed)

    # Parse beta configuration: hierarchical or single-beta mode
    hierarchical_mode = cfg.get("hierarchical_beta", False)
    if hierarchical_mode:
        beta_price_anchor = float(cfg.get("beta_price_anchor", 1e-4))
        beta_micro_grid = np.array(cfg["beta_grid"], dtype=float)
        # Create beta configs as dicts for hierarchical mode
        beta_configs = [
            {"beta_price": beta_price_anchor, "beta_micro": bm} 
            for bm in beta_micro_grid
        ]
    else:
        # Original single-beta mode (backward compatible)
        beta_configs = list(np.array(cfg["beta_grid"], dtype=float))
    
    representations = list(cfg["representations"])
    gamma = float(cfg["gamma"])

    n_steps = int(cfg["n_steps"])
    T = float(cfg["T"])
    K = float(cfg["K"])
    vol_hat = float(cfg["vol_hat"])
    n_paths_train = int(cfg["n_paths_train"])
    n_paths_eval = int(cfg["n_paths_eval"])
    train_eta = float(cfg["train_eta"])
    stress_eta = float(cfg["stress_eta"])
    train_lambdas = np.array(cfg["train_lambdas"], dtype=float)

    S_train, _, V_train, lam_train, _ = simulate_heston_signflip(
        regime=0, n_paths=n_paths_train, n_steps=n_steps, T=T, seed=base_seed
    )
    S_eval, _, V_eval, lam_eval, _ = simulate_heston_signflip(
        regime=0, n_paths=n_paths_eval, n_steps=n_steps, T=T, seed=eval_seed
    )

    rows = []
    for rep in representations:
        for beta_cfg in beta_configs:
            # Setup device and model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            if rep == "combined":
                input_dim = 2
            elif rep == "micro":
                input_dim = 3
            elif rep == "oracle":
                input_dim = 4
            else:
                input_dim = 1
            
            model = FactorizedVariationalPolicy(input_dim, latent_dim_per_feature=2).to(device)
            
            # Convert to tensors
            S_t = torch.tensor(S_train, dtype=torch.float32, device=device)
            V_t = torch.tensor(V_train, dtype=torch.float32, device=device)
            lam_t = torch.tensor(lam_train, dtype=torch.float32, device=device)
            
            # Training config (beta_cfg can be float or dict)
            training_config = {
                "beta": beta_cfg,  # Now can be float (single-beta) or dict (hierarchical)
                "train_eta": train_eta,
                "train_lambdas": train_lambdas.tolist(),
                "gamma": gamma,
                "lr": 0.005,
                "n_epochs": 150,
                "warmup_epochs": 30
            }
            
            # Train and get full result (including history)
            result = train_model(
                model, S_t, V_t, lam_t, training_config,
                representation=rep, T=T, K=K, vol_hat=vol_hat
            )
            
            w = result["final_weights"]
            history = result["history"]
            
            # Save training history for information autopsy
            # Create directory name based on beta configuration
            if isinstance(beta_cfg, dict):
                beta_str = f"bp{beta_cfg['beta_price']:.4f}_bm{beta_cfg['beta_micro']:.4f}"
            else:
                beta_str = f"beta_{beta_cfg:.4f}"
            history_dir = ROOT / "runs" / f"{run_id}_{beta_str}_{rep}"
            history_dir.mkdir(parents=True, exist_ok=True)
            history_file = history_dir / "training_history.json"
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            losses, info_cost, turnover_rate, exec_cost = hedge_on_paths(
                S_eval,
                V_eval,
                lam_eval,
                T=T,
                K=K,
                vol_hat=vol_hat,
                representation=rep,
                weights_or_state_dict=w,
            )
            r0 = robust_es_kl(losses, eta=0.0, gamma=gamma)
            r_stress = robust_es_kl(losses, eta=stress_eta, gamma=gamma)
            
            # Extract beta values for CSV output
            if isinstance(beta_cfg, dict):
                beta_val = beta_cfg.get("beta_micro", 0.0)  # Use beta_micro as primary beta for plotting
                beta_price_val = beta_cfg.get("beta_price", 0.0)
                beta_micro_val = beta_cfg.get("beta_micro", 0.0)
            else:
                beta_val = float(beta_cfg)
                beta_price_val = float(beta_cfg)
                beta_micro_val = float(beta_cfg)
            
            rows.append(
                {
                    "representation": rep,
                    "beta": beta_val,
                    "beta_price": beta_price_val,
                    "beta_micro": beta_micro_val,
                    "R0": float(r0),
                    "R_stress_eta0p1": float(r_stress),
                    "info_cost": float(info_cost),
                    "turnover": float(turnover_rate),
                    "exec_cost": float(exec_cost),
                }
            )

    # Compute BS Delta Hedge Anchor Benchmark
    print("Computing BS Delta Hedge anchor...")
    tau_grid = np.linspace(T, 0.0, n_steps + 1)
    n_eval = S_eval.shape[0]
    
    # Simulate BS delta hedging
    a_bs = np.zeros(n_eval)
    pnl_bs = np.zeros(n_eval)
    cost_bs = np.zeros(n_eval)
    turnover_bs = np.zeros(n_eval)
    
    for t in range(n_steps):
        tau_t = np.full(n_eval, tau_grid[t])
        a_new = policy_delta_hedge(S_eval[:, t], K=K, tau_t=tau_t, vol_hat=vol_hat)
        a_new = np.clip(a_new, -5.0, 5.0)
        
        da = a_new - a_bs
        dS = S_eval[:, t + 1] - S_eval[:, t]
        
        pnl_bs += a_new * dS
        cost_bs += 0.5 * lam_eval[:, t] * (da ** 2)
        turnover_bs += np.abs(da)
        a_bs = a_new
    
    payoff_bs = np.maximum(S_eval[:, -1] - K, 0)
    losses_bs = payoff_bs - pnl_bs - cost_bs
    
    r0_bs = robust_es_kl(losses_bs, eta=0.0, gamma=gamma)
    r_stress_bs = robust_es_kl(losses_bs, eta=stress_eta, gamma=gamma)
    turnover_rate_bs = np.mean(turnover_bs) / n_steps
    exec_cost_bs = np.mean(cost_bs) / n_steps
    
    # Add BS anchor to results
    rows.append({
        "representation": "BS_Anchor",
        "beta": np.nan,  # Not applicable
        "R0": float(r0_bs),
        "R_stress_eta0p1": float(r_stress_bs),
        "info_cost": 0.0,  # No information bottleneck
        "turnover": float(turnover_rate_bs),
        "exec_cost": float(exec_cost_bs),
    })

    out_csv = ROOT / "runs" / f"paper_{run_id}_frontier.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "representation",
                "beta",
                "beta_price",
                "beta_micro",
                "R0",
                "R_stress_eta0p1",
                "info_cost",
                "turnover",
                "exec_cost",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", out_csv)

    reps = [r["representation"] for r in rows]
    r0s = np.array([r["R0"] for r in rows])
    rstress = np.array([r["R_stress_eta0p1"] for r in rows])
    info = np.array([r["info_cost"] for r in rows])

    rep_markers = {"greeks": "o", "micro": "s", "combined": "^"}
    plt.figure(figsize=(6, 4))
    for rep in representations:
        idx = [i for i, r in enumerate(reps) if r == rep]
        plt.scatter(
            r0s[idx],
            rstress[idx],
            c=info[idx],
            cmap="viridis",
            marker=rep_markers[rep],
            label=rep,
            alpha=0.8,
        )

    plt.colorbar(label="KL_inner (info cost)")
    
    # Add BS Anchor as red X
    bs_idx = [i for i, r in enumerate(rows) if r["representation"] == "BS_Anchor"]
    if bs_idx:
        plt.scatter(
            r0s[bs_idx],
            rstress[bs_idx],
            c='red',
            marker='X',
            s=200,
            label='BS Anchor',
            edgecolors='black',
            linewidths=2,
            zorder=10
        )
    
    plt.xlabel(r"$R_0$")
    plt.ylabel(r"$R_\eta$ ($\eta=0.1$)")
    plt.title("Robustness-information frontier")
    plt.legend(fontsize=8, loc="best")

    out_plot = ROOT / "figures" / f"paper_{run_id}_frontier_beta_sweep.png"
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_plot, dpi=200)
    print("Saved:", out_plot)


if __name__ == "__main__":
    main()

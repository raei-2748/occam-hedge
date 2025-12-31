import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from experiment_occam import hedge_on_paths, train_weights
from world import simulate_heston_signflip
from risk import robust_es_kl
from paper_config import load_config, run_id_from_config
from utils import set_seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "paper_run.json"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_id = run_id_from_config(cfg)

    base_seed = int(cfg["seed_train"])
    eval_seed = int(cfg["seed_eval"])
    set_seeds(base_seed)

    betas = np.array(cfg["beta_grid"], dtype=float)
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
        for beta in betas:
            w = train_weights(
                S_train,
                V_train,
                lam_train,
                T=T,
                K=K,
                vol_hat=vol_hat,
                representation=rep,
                beta=float(beta),
                train_eta=train_eta,
                train_lambdas=train_lambdas,
                gamma=gamma,
            )
            losses, info_cost, turnover_rate, exec_cost = hedge_on_paths(
                S_eval,
                V_eval,
                lam_eval,
                T=T,
                K=K,
                vol_hat=vol_hat,
                representation=rep,
                weights=w,
            )
            r0 = robust_es_kl(losses, eta=0.0, gamma=gamma)
            r_stress = robust_es_kl(losses, eta=stress_eta, gamma=gamma)
            rows.append(
                {
                    "representation": rep,
                    "beta": float(beta),
                    "R0": float(r0),
                    "R_stress_eta0p1": float(r_stress),
                    "info_cost": float(info_cost),
                    "turnover": float(turnover_rate),
                    "exec_cost": float(exec_cost),
                }
            )

    out_csv = ROOT / "runs" / f"paper_{run_id}_frontier.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "representation",
                "beta",
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

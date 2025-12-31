import argparse
import json
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "paper_run.json"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_id = run_id_from_config(cfg)

    base_seed = int(cfg["seed_train"])
    eval_seed = int(cfg["seed_eval"])
    np.random.seed(base_seed)

    etas = np.array(cfg["eta_grid"], dtype=float)
    betas = np.array(cfg["beta_grid_curves"], dtype=float)
    representations = list(cfg["representations"])
    gamma = float(cfg["gamma"])

    n_steps = int(cfg["n_steps"])
    T = float(cfg["T"])
    K = float(cfg["K"])
    vol_hat = float(cfg["vol_hat"])
    n_paths_train = int(cfg["n_paths_train"])
    n_paths_eval = int(cfg["n_paths_eval"])
    train_eta = float(cfg["train_eta"])
    train_lambdas = np.array(cfg["train_lambdas"], dtype=float)

    S_train, _, V_train, lam_train, _ = simulate_heston_signflip(
        regime=0, n_paths=n_paths_train, n_steps=n_steps, T=T, seed=base_seed
    )
    S_eval, _, V_eval, lam_eval, _ = simulate_heston_signflip(
        regime=0, n_paths=n_paths_eval, n_steps=n_steps, T=T, seed=eval_seed
    )

    trained = {}
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
            trained[(rep, float(beta))] = w

    results = []
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for i, rep in enumerate(representations):
        ax = axes[i]
        for beta in betas:
            w = trained[(rep, float(beta))]
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
            R = [robust_es_kl(losses, eta=e, gamma=gamma) for e in etas]
            ax.plot(etas, R, marker="o", label=f"beta={beta:g}", alpha=0.85)

            results.append(
                {
                    "representation": rep,
                    "beta": float(beta),
                    "etas": etas.tolist(),
                    "R_eta": [float(r) for r in R],
                    "info_cost": float(info_cost),
                    "turnover": float(turnover_rate),
                    "exec_cost": float(exec_cost),
                }
            )

        ax.set_title(rep)
        ax.set_xlabel(r"$\eta$")
        if i == 0:
            ax.set_ylabel(r"$R_\eta$")
        ax.legend(fontsize=7, ncol=2, loc="best")

    fig.suptitle("Robust risk curves (Regime 0)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    out_plot = ROOT / "figures" / f"paper_{run_id}_robust_risk_vs_eta.png"
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=200)
    print("Saved:", out_plot)

    out_json = ROOT / "runs" / f"paper_{run_id}_robust_curves.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": cfg,
        "run_id": run_id,
        "results": results,
    }
    with out_json.open("w") as f:
        json.dump(payload, f, indent=2)
    print("Saved:", out_json)


if __name__ == "__main__":
    main()

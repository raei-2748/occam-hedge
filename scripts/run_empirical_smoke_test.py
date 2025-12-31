import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from experiment_occam import hedge_on_paths, train_weights
from world import simulate_heston_signflip
from risk import robust_es_kl
from paper_config import load_config, run_id_from_config
from utils import set_seeds


def _train_and_eval(
    representation: str,
    beta: float,
    S_train: np.ndarray,
    V_train: np.ndarray,
    lam_train: np.ndarray,
    S_eval: np.ndarray,
    V_eval: np.ndarray,
    lam_eval: np.ndarray,
    T: float,
    K: float,
    vol_hat: float,
    gamma: float,
    train_eta: float,
    train_lambdas: np.ndarray,
    stress_eta: float,
) -> dict:
    w = train_weights(
        S_train,
        V_train,
        lam_train,
        T=T,
        K=K,
        vol_hat=vol_hat,
        representation=representation,
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
        representation=representation,
        weights=w,
    )
    r0 = robust_es_kl(losses, eta=0.0, gamma=gamma)
    r_stress = robust_es_kl(losses, eta=stress_eta, gamma=gamma)
    return {
        "representation": representation,
        "beta": float(beta),
        "R0": float(r0),
        "R_stress": float(r_stress),
        "info_cost": float(info_cost),
        "turnover": float(turnover_rate),
        "exec_cost": float(exec_cost),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "paper_run.json"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_id = run_id_from_config(cfg)

    smoke_cfg = cfg["smoke"]
    base_seed = int(smoke_cfg["seed"])
    eval_seed = int(cfg["seed_eval"])
    set_seeds(base_seed)

    gamma = float(cfg["gamma"])
    stress_eta = float(cfg["stress_eta"])
    train_eta = float(cfg["train_eta"])
    train_lambdas = np.array(cfg["train_lambdas"], dtype=float)

    n_steps = int(cfg["n_steps"])
    T = float(cfg["T"])
    K = float(cfg["K"])
    vol_hat = float(cfg["vol_hat"])
    n_paths_train = int(smoke_cfg["n_paths_train"])
    n_paths_eval = int(smoke_cfg["n_paths_eval"])

    S_train, _, V_train, lam_train, _ = simulate_heston_signflip(
        regime=0, n_paths=n_paths_train, n_steps=n_steps, T=T, seed=base_seed
    )
    S_eval, _, V_eval, lam_eval, _ = simulate_heston_signflip(
        regime=0, n_paths=n_paths_eval, n_steps=n_steps, T=T, seed=eval_seed
    )

    results = []
    results.append(
        _train_and_eval(
            "greeks",
            beta=float(smoke_cfg["beta_greeks"]),
            S_train=S_train,
            V_train=V_train,
            lam_train=lam_train,
            S_eval=S_eval,
            V_eval=V_eval,
            lam_eval=lam_eval,
            T=T,
            K=K,
            vol_hat=vol_hat,
            gamma=gamma,
            train_eta=train_eta,
            train_lambdas=train_lambdas,
            stress_eta=stress_eta,
        )
    )
    results.append(
        _train_and_eval(
            "micro",
            beta=float(smoke_cfg["beta_micro"]),
            S_train=S_train,
            V_train=V_train,
            lam_train=lam_train,
            S_eval=S_eval,
            V_eval=V_eval,
            lam_eval=lam_eval,
            T=T,
            K=K,
            vol_hat=vol_hat,
            gamma=gamma,
            train_eta=train_eta,
            train_lambdas=train_lambdas,
            stress_eta=stress_eta,
        )
    )

    out_json = ROOT / "runs" / f"paper_{run_id}_smoke_results.json"
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

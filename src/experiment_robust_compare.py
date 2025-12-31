# src/experiment_robust_compare.py
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from hedge_sim import run_hedge_paths
from risk import robust_es_kl
from paper_config import load_config, run_id_from_config

ROOT = Path(__file__).resolve().parents[1]


def robust_curve(losses: np.ndarray, etas: np.ndarray, gamma: float) -> np.ndarray:
    return np.array([robust_es_kl(losses, eta=e, gamma=gamma) for e in etas], dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "paper_run.json"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_id = run_id_from_config(cfg)

    etas = np.array(cfg["eta_grid_compare"], dtype=float)
    gamma = float(cfg["gamma"])
    n_paths_compare = int(cfg["n_paths_compare"])
    n_steps = int(cfg["n_steps"])
    seed = int(cfg["seed_train"])

    # KL stress is anchored on the training distribution (Regime 0).
    Y0_delta, _ = run_hedge_paths(
        regime=0,
        policy="delta",
        n_paths=n_paths_compare,
        n_steps=n_steps,
        seed=seed,
    )
    Y0_micro, _ = run_hedge_paths(
        regime=0,
        policy="micro",
        n_paths=n_paths_compare,
        n_steps=n_steps,
        seed=seed,
    )

    R_delta = robust_curve(Y0_delta, etas, gamma=gamma)
    R_micro = robust_curve(Y0_micro, etas, gamma=gamma)

    plt.figure()
    plt.plot(etas, R_delta, marker="o", label="Delta hedge (Greek-based)")
    plt.plot(etas, R_micro, marker="o", label="Microstructure-heavy (volume-reactive)")
    plt.xlabel(r"$\eta$")
    plt.ylabel(r"$R_\eta$")
    plt.title("KL-robust risk (Regime 0)")
    plt.legend()

    out = ROOT / "figures" / f"paper_{run_id}_robust_compare_regime0.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    print("Saved:", out)

    # Separate out-of-regime evaluation (Regime 1) for reference.
    Y1_delta, _ = run_hedge_paths(
        regime=1,
        policy="delta",
        n_paths=n_paths_compare,
        n_steps=n_steps,
        seed=seed,
    )
    Y1_micro, _ = run_hedge_paths(
        regime=1,
        policy="micro",
        n_paths=n_paths_compare,
        n_steps=n_steps,
        seed=seed,
    )
    print("Regime 1 mean loss: delta =", float(np.mean(Y1_delta)), "micro =", float(np.mean(Y1_micro)))


if __name__ == "__main__":
    main()

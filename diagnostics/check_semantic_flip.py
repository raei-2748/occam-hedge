import argparse
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from world import simulate_heston_signflip
from paper_config import load_config, run_id_from_config


def _corr_flat(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "paper_run.json"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_id = run_id_from_config(cfg)

    sem_cfg = cfg["semantic_flip"]
    seeds = list(sem_cfg["seeds"])
    n_paths = int(sem_cfg["n_paths"])
    n_steps = int(sem_cfg["n_steps"])
    T = float(sem_cfg["T"])

    corr_regime0 = []
    corr_regime1 = []
    flips = []

    for seed in seeds:
        _, _, V0, lam0, _ = simulate_heston_signflip(
            regime=0, n_paths=n_paths, n_steps=n_steps, T=T, seed=seed
        )
        _, _, V1, lam1, _ = simulate_heston_signflip(
            regime=1, n_paths=n_paths, n_steps=n_steps, T=T, seed=seed
        )
        c0 = _corr_flat(V0, lam0)
        c1 = _corr_flat(V1, lam1)
        corr_regime0.append(c0)
        corr_regime1.append(c1)
        flips.append(np.sign(c0) != np.sign(c1))

    flip_rate = float(np.mean(flips))
    if flip_rate < 0.9:
        raise AssertionError("Semantic sign flip not consistently detected")

    summary = {
        "n_trials": len(seeds),
        "flip_rate": flip_rate,
        "corr_regime0_mean": float(np.mean(corr_regime0)),
        "corr_regime1_mean": float(np.mean(corr_regime1)),
        "corr_regime0": [float(x) for x in corr_regime0],
        "corr_regime1": [float(x) for x in corr_regime1],
    }

    out_json = ROOT / "runs" / f"paper_{run_id}_semantic_flip_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)
    print("Saved:", out_json)

    plt.figure(figsize=(5, 3))
    x0 = np.zeros(len(corr_regime0))
    x1 = np.ones(len(corr_regime1))
    plt.scatter(x0, corr_regime0, color="tab:blue", alpha=0.7, label="Regime 0")
    plt.scatter(x1, corr_regime1, color="tab:orange", alpha=0.7, label="Regime 1")
    plt.axhline(0.0, color="k", linewidth=0.8)
    plt.xticks([0, 1], ["Regime 0", "Regime 1"])
    plt.ylabel("corr(Vol_t, impact)")
    plt.title("Semantic sign flip diagnostics")
    plt.legend(loc="best", fontsize=8)

    out_plot = ROOT / "figures" / f"paper_{run_id}_semantic_flip_correlations.png"
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_plot, dpi=200)
    print("Saved:", out_plot)


if __name__ == "__main__":
    main()

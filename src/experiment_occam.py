# src/experiment_occam.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from world import simulate_heston_signflip
from risk import robust_es_kl
from policies import bs_delta_call

ROOT = Path(__file__).resolve().parents[1]


def micro_signal(V_t: np.ndarray) -> np.ndarray:
    Vn = V_t / (np.median(V_t) + 1e-12)
    return np.tanh(np.log(Vn + 1e-12))


def occam_features(
    representation: str,
    S_t: np.ndarray,
    tau_t: np.ndarray,
    V_t: np.ndarray,
    K: float,
    vol_hat: float,
) -> np.ndarray:
    if representation == "greeks":
        delta = bs_delta_call(S_t, K=K, tau=tau_t, vol=vol_hat, r=0.0)
        return delta.reshape(-1, 1)
    if representation == "micro":
        micro = micro_signal(V_t)
        return micro.reshape(-1, 1)
    if representation == "combined":
        delta = bs_delta_call(S_t, K=K, tau=tau_t, vol=vol_hat, r=0.0)
        micro = micro_signal(V_t)
        return np.column_stack([delta, micro])
    raise ValueError("representation must be 'greeks', 'micro', or 'combined'")


def hedge_on_paths(
    S: np.ndarray,
    V: np.ndarray,
    lam: np.ndarray,
    T: float,
    K: float,
    vol_hat: float,
    representation: str,
    weights: np.ndarray,
    action_clip: float = 5.0,
) -> tuple[np.ndarray, float, float, float]:
    n_paths, n_steps = V.shape
    tau_grid = np.linspace(T, 0.0, n_steps + 1)

    a = np.zeros((n_paths,), dtype=np.float64)
    pnl = np.zeros((n_paths,), dtype=np.float64)
    cost = np.zeros((n_paths,), dtype=np.float64)
    info = np.zeros((n_paths,), dtype=np.float64)
    turnover = np.zeros((n_paths,), dtype=np.float64)

    for t in range(n_steps):
        S_t = S[:, t]
        tau_t = np.full((n_paths,), tau_grid[t], dtype=np.float64)

        feats = occam_features(representation, S_t, tau_t, V[:, t], K=K, vol_hat=vol_hat)
        mu = feats * weights
        a_new = np.sum(mu, axis=1)
        a_new = np.clip(a_new, -action_clip, action_clip)

        da = a_new - a
        dS = S[:, t + 1] - S[:, t]
        pnl += a_new * dS
        cost += 0.5 * lam[:, t] * (da ** 2)
        info += 0.5 * np.sum(mu ** 2, axis=1)
        turnover += np.abs(da)
        a = a_new

    payoff = np.maximum(S[:, -1] - K, 0.0)
    losses = payoff - pnl - cost
    info_cost = float(np.mean(info))
    turnover_rate = float(np.mean(turnover) / n_steps)
    exec_cost = float(np.mean(cost) / n_steps)
    return losses, info_cost, turnover_rate, exec_cost


def weight_grid(representation: str) -> list[np.ndarray]:
    if representation == "greeks":
        return [np.array([w]) for w in np.linspace(0.0, 1.4, 8)]
    if representation == "micro":
        return [np.array([w]) for w in np.linspace(-1.5, 1.5, 9)]
    if representation == "combined":
        w_delta = np.linspace(0.2, 1.4, 6)
        w_micro = np.linspace(-1.2, 1.2, 6)
        return [np.array([wd, wm]) for wd in w_delta for wm in w_micro]
    raise ValueError("representation must be 'greeks', 'micro', or 'combined'")


def train_weights(
    S: np.ndarray,
    V: np.ndarray,
    lam: np.ndarray,
    T: float,
    K: float,
    vol_hat: float,
    representation: str,
    beta: float,
    train_eta: float,
    train_lambdas: np.ndarray,
    gamma: float,
) -> np.ndarray:
    best_obj = np.inf
    best_w = None

    for w in weight_grid(representation):
        losses, info_cost, _, _ = hedge_on_paths(
            S,
            V,
            lam,
            T=T,
            K=K,
            vol_hat=vol_hat,
            representation=representation,
            weights=w,
        )
        risk = robust_es_kl(losses, eta=train_eta, gamma=gamma, lambdas=train_lambdas)
        obj = float(risk + beta * info_cost)
        if obj < best_obj:
            best_obj = obj
            best_w = w

    if best_w is None:
        raise RuntimeError("No weights found in grid")

    return best_w


def robust_curve(losses: np.ndarray, etas: np.ndarray, gamma: float) -> np.ndarray:
    return np.array([robust_es_kl(losses, eta=e, gamma=gamma) for e in etas], dtype=float)


def main():
    base_seed = 0
    np.random.seed(base_seed)

    etas = np.array([0.0, 0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6])
    betas = np.array([0.0, 0.01, 0.03, 0.06, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    representations = ["greeks", "micro", "combined"]

    n_steps = 100
    T = 1.0
    K = 100.0
    vol_hat = 0.20
    n_paths_train = 3000
    n_paths_eval = 12000
    train_eta = 0.05
    train_lambdas = np.logspace(-3, 1.2, 60)
    gamma = 0.95

    # Training and evaluation are both anchored on Regime 0 (P0).
    S_train, _, V_train, lam_train, _ = simulate_heston_signflip(
        regime=0, n_paths=n_paths_train, n_steps=n_steps, T=T, seed=base_seed
    )
    S_eval, _, V_eval, lam_eval, _ = simulate_heston_signflip(
        regime=0, n_paths=n_paths_eval, n_steps=n_steps, T=T, seed=base_seed + 777
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

    # Robust risk curves (Regime 0 anchor)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    curves = {}
    metrics = []
    stress_eta_1 = 0.1
    stress_eta_2 = 0.2
    eta_to_idx = {float(e): i for i, e in enumerate(etas)}

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
            R = robust_curve(losses, etas, gamma=gamma)
            curves[(rep, float(beta))] = R
            ax.plot(etas, R, marker="o", label=f"beta={beta:g}", alpha=0.85)

            R0 = float(R[eta_to_idx[0.0]])
            R_stress_1 = float(R[eta_to_idx[stress_eta_1]])
            R_stress_2 = float(R[eta_to_idx[stress_eta_2]])
            metrics.append(
                dict(
                    representation=rep,
                    beta=float(beta),
                    R0=R0,
                    R_stress_0p1=R_stress_1,
                    R_stress_0p2=R_stress_2,
                    info_cost=float(info_cost),
                    turnover=float(turnover_rate),
                    exec_cost=float(exec_cost),
                )
            )

        ax.set_title(rep)
        ax.set_xlabel(r"$\eta$")
        if i == 0:
            ax.set_ylabel(r"$R_\eta$")
        ax.legend(fontsize=7, ncol=2, loc="best")

    fig.suptitle("Robust risk curves (Regime 0)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    out_curves = ROOT / "figures" / "robust_curves_occam.png"
    out_curves.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_curves, dpi=200)
    print("Saved:", out_curves)

    # Normalized robust risk curves
    fig_norm, axes_norm = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for i, rep in enumerate(representations):
        ax = axes_norm[i]
        for beta in betas:
            R = curves[(rep, float(beta))]
            R0 = float(R[eta_to_idx[0.0]])
            denom = R0 if abs(R0) > 1e-12 else np.nan
            ax.plot(etas, R / denom, marker="o", label=f"beta={beta:g}", alpha=0.85)
        ax.set_title(rep)
        ax.set_xlabel(r"$\eta$")
        if i == 0:
            ax.set_ylabel(r"$R_\eta / R_0$")
        ax.legend(fontsize=7, ncol=2, loc="best")

    fig_norm.suptitle("Normalized robust risk (Regime 0)")
    fig_norm.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    out_norm = ROOT / "figures" / "robust_curves_occam_normalized.png"
    out_norm.parent.mkdir(parents=True, exist_ok=True)
    fig_norm.savefig(out_norm, dpi=200)
    print("Saved:", out_norm)

    # Robustness-information frontier
    reps = [m["representation"] for m in metrics]
    R0s = np.array([m["R0"] for m in metrics])
    Rstress_1 = np.array([m["R_stress_0p1"] for m in metrics])
    Rstress_2 = np.array([m["R_stress_0p2"] for m in metrics])
    info = np.array([m["info_cost"] for m in metrics])
    betas_all = np.array([m["beta"] for m in metrics])

    rep_markers = {"greeks": "o", "micro": "s", "combined": "^"}
    highlight = {0.0, 0.1, 0.4, 1.0}
    plt.figure(figsize=(6, 4))
    for rep in representations:
        idx = [j for j, r in enumerate(reps) if r == rep]
        plt.scatter(
            R0s[idx],
            Rstress_1[idx],
            c=info[idx],
            cmap="viridis",
            marker=rep_markers[rep],
            label=rep,
            alpha=0.75,
        )
        idx_hi = [j for j in idx if betas_all[j] in highlight]
        if idx_hi:
            plt.scatter(
                R0s[idx_hi],
                Rstress_1[idx_hi],
                c=info[idx_hi],
                cmap="viridis",
                marker=rep_markers[rep],
                edgecolor="k",
                linewidth=0.6,
                s=50,
            )

    plt.colorbar(label="C")
    plt.xlabel(r"$R_0$")
    plt.ylabel(r"$R_\eta$ ($\eta=0.1$)")
    plt.title(r"Frontier ($\eta=0.1$)")
    plt.legend(fontsize=8, loc="best")

    out_frontier = ROOT / "figures" / "robust_frontier_occam_eta0p1.png"
    out_frontier.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_frontier, dpi=200)
    print("Saved:", out_frontier)

    plt.figure(figsize=(6, 4))
    for rep in representations:
        idx = [j for j, r in enumerate(reps) if r == rep]
        plt.scatter(
            R0s[idx],
            Rstress_2[idx],
            c=info[idx],
            cmap="viridis",
            marker=rep_markers[rep],
            label=rep,
            alpha=0.75,
        )
        idx_hi = [j for j in idx if betas_all[j] in highlight]
        if idx_hi:
            plt.scatter(
                R0s[idx_hi],
                Rstress_2[idx_hi],
                c=info[idx_hi],
                cmap="viridis",
                marker=rep_markers[rep],
                edgecolor="k",
                linewidth=0.6,
                s=50,
            )

    plt.colorbar(label="C")
    plt.xlabel(r"$R_0$")
    plt.ylabel(r"$R_\eta$ ($\eta=0.2$)")
    plt.title(r"Frontier ($\eta=0.2$)")
    plt.legend(fontsize=8, loc="best")

    out_frontier_2 = ROOT / "figures" / "robust_frontier_occam_eta0p2.png"
    out_frontier_2.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_frontier_2, dpi=200)
    print("Saved:", out_frontier_2)

    # Diagnostic: turnover and execution cost vs beta
    plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    for rep in ["micro", "combined"]:
        rep_rows = [m for m in metrics if m["representation"] == rep]
        rep_rows = sorted(rep_rows, key=lambda x: x["beta"])
        beta_vals = np.array([m["beta"] for m in rep_rows])
        turn_vals = np.array([m["turnover"] for m in rep_rows])
        exec_vals = np.array([m["exec_cost"] for m in rep_rows])
        ax1.plot(beta_vals, turn_vals, marker="o", label=rep)
        ax2.plot(beta_vals, exec_vals, marker="o", label=rep)

    ax1.set_ylabel("Turnover")
    ax1.legend(fontsize=8, loc="best")
    ax1.set_title(r"Behavior vs $\beta$")
    ax2.set_ylabel("Exec cost")
    ax2.set_xlabel(r"$\beta$")
    ax2.legend(fontsize=8, loc="best")

    out_diag = ROOT / "figures" / "occam_diagnostic_turnover.png"
    out_diag.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_diag, dpi=200)
    print("Saved:", out_diag)

    # Tabulate matched-baseline subset
    target = float(np.median([m["R0"] for m in metrics]))
    tol = 0.05 * abs(target)
    matched = [m for m in metrics if abs(m["R0"] - target) <= tol]
    matched = sorted(matched, key=lambda x: x["info_cost"])

    out_table = ROOT / "runs" / "occam_frontier.csv"
    out_table.parent.mkdir(parents=True, exist_ok=True)
    with out_table.open("w") as f:
        f.write(
            "representation,beta,R0,R_stress_eta0p1,R_stress_eta0p2,"
            "info_cost,turnover,exec_cost,stress_lift_eta0p1\n"
        )
        for m in metrics:
            lift = m["R_stress_0p1"] - m["R0"]
            f.write(
                f"{m['representation']},{m['beta']:.3f},{m['R0']:.6f},"
                f"{m['R_stress_0p1']:.6f},{m['R_stress_0p2']:.6f},"
                f"{m['info_cost']:.6f},{m['turnover']:.6f},{m['exec_cost']:.6f},"
                f"{lift:.6f}\n"
            )
    print("Saved:", out_table)

    print("Matched-baseline subset (|R0-target| <=", tol, ")")
    for m in matched:
        lift = m["R_stress_0p1"] - m["R0"]
        print(
            "rep=", m["representation"],
            "beta=", m["beta"],
            "R0=", round(m["R0"], 4),
            "R_stress=", round(m["R_stress_0p1"], 4),
            "C=", round(m["info_cost"], 4),
            "lift=", round(lift, 4),
        )


if __name__ == "__main__":
    main()

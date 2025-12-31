# src/hedge_sim.py
from __future__ import annotations
import numpy as np

from world import simulate_heston_signflip
from policies import policy_delta_hedge, policy_volume_reactive


def call_payoff(ST: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(ST - K, 0.0)


def run_hedge_paths(
    regime: int,
    n_paths: int = 20_000,
    n_steps: int = 100,
    T: float = 1.0,
    seed: int = 0,
    K: float = 100.0,
    vol_hat: float = 0.20,
    policy: str = "delta",
    k_vol: float = 1.5,
    lam_mult: float = 1.0,
):
    """
    Simulate hedging and return per-path loss samples.

    Loss definition (simple and consistent):
      Y = payoff - hedge_PnL - impact_cost
    where hedge_PnL = sum a_t * (S_{t+1}-S_t)
          impact_cost = sum 0.5 * lambda_t * (Δa_t)^2

    Note: This is intentionally minimal; we add spreads etc later.
    """
    S, v, V, lam, meta = simulate_heston_signflip(
        regime=regime,
        n_paths=n_paths,
        n_steps=n_steps,
        T=T,
        seed=seed,
    )
    lam = lam_mult * lam

    dt = T / n_steps
    tau_grid = np.linspace(T, 0.0, n_steps + 1)  # remaining time at each step

    # actions a_t (position in underlying)
    a = np.zeros((n_paths,), dtype=np.float64)
    pnl = np.zeros((n_paths,), dtype=np.float64)
    cost = np.zeros((n_paths,), dtype=np.float64)

    for t in range(n_steps):
        S_t = S[:, t]
        tau_t = np.full((n_paths,), tau_grid[t], dtype=np.float64)

        if policy == "delta":
            a_new = policy_delta_hedge(S_t, K=K, tau_t=tau_t, vol_hat=vol_hat)
        elif policy == "micro":
            a_new = policy_volume_reactive(S_t, K=K, tau_t=tau_t, vol_hat=vol_hat, V_t=V[:, t], k=k_vol)
        else:
            raise ValueError("policy must be 'delta' or 'micro'")

        da = a_new - a

        # hedge PnL from holding a over (t,t+1)
        dS = S[:, t + 1] - S[:, t]
        pnl += a_new * dS

        # impact cost
        cost += 0.5 * lam[:, t] * (da ** 2)

        a = a_new

    payoff = call_payoff(S[:, -1], K=K)
    Y = payoff - pnl - cost
    return Y, dict(meta=meta, K=K, vol_hat=vol_hat, policy=policy, k_vol=k_vol, lam_mult=lam_mult)

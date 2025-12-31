from __future__ import annotations
import numpy as np
import math

def _norm_cdf(x):
    # vectorized math.erf, works everywhere
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def bs_delta_call(S: np.ndarray, K: float, tau: np.ndarray, vol: float, r: float = 0.0) -> np.ndarray:
    """
    Black–Scholes delta for a European call.
    S: (n_paths,) current price
    tau: (n_paths,) time-to-maturity in years
    """
    eps = 1e-12
    tau = np.maximum(tau, eps)
    vol = max(vol, 1e-12)

    d1 = (np.log((S + eps) / K) + (r + 0.5 * vol * vol) * tau) / (vol * np.sqrt(tau))
    return _norm_cdf(d1)


def policy_delta_hedge(S_t: np.ndarray, K: float, tau_t: np.ndarray, vol_hat: float) -> np.ndarray:
    """
    Payoff-anchored policy: hold delta shares.
    """
    return bs_delta_call(S_t, K=K, tau=tau_t, vol=vol_hat, r=0.0)


def policy_volume_reactive(
    S_t: np.ndarray,
    K: float,
    tau_t: np.ndarray,
    vol_hat: float,
    V_t: np.ndarray,
    k: float = 1.5,
) -> np.ndarray:
    """
    Microstructure-heavy policy: take delta, then scale aggressiveness by volume proxy.

    Idea:
      In Regime 0, high V_t => cheap trading => scale up adjustments.
      In Regime 1, that logic becomes wrong because high V_t => high impact.

    k controls how strongly volume changes the position.
    """
    base = bs_delta_call(S_t, K=K, tau=tau_t, vol=vol_hat, r=0.0)

    # Normalize volume within the cross-section at time t to a stable scale.
    # This prevents 'crazy' scaling when volume distribution changes.
    V = V_t
    Vn = V / (np.median(V) + 1e-12)

    # Smooth, bounded scaling so it doesn't explode
    scale = 1.0 + k * np.tanh(np.log(Vn + 1e-12))
    return np.clip(base * scale, -5.0, 5.0)

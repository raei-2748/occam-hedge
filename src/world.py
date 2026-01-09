from __future__ import annotations

import numpy as np


def simulate_heston_signflip(
    regime: int,
    n_paths: int = 20_000,
    n_steps: int = 100,
    T: float = 1.0,
    seed: int = 0,
    # Heston-like parameters (risk-neutral-ish, but we just need realistic paths)
    s0: float = 100.0,
    v0: float = 0.04,          # variance, so vol is sqrt(v0)=20%
    mu: float = 0.00,          # drift is not that important for hedging experiments
    kappa: float = 2.0,        # mean reversion speed
    theta: float = 0.04,       # long-run variance
    xi: float = 0.60,          # vol-of-vol
    rho: float = -0.60,        # correlation between returns and variance shocks
    # Microstructure / impact parameters
    lam0: float = 0.45,        # base impact scale (final tune for >10% gap)
    kappa_alpha: float = 2.5,  # sensitivity to toxicity (sharpened)
    kappa_m: float = 0.8,      # sensitivity to margin/funding tightness
    # Latent microstructure dynamics (log-AR(1))
    alpha_bar_0: float = 0.0,  # log baseline in Regime 0
    alpha_bar_1: float = 0.6,  # log baseline in Regime 1
    m_bar_0: float = 0.0,      # log baseline in Regime 0
    m_bar_1: float = 0.5,      # log baseline in Regime 1
    kappa_alpha_lat: float = 2.5,
    kappa_m_lat: float = 2.0,
    sigma_alpha_0: float = 0.35,
    sigma_alpha_1: float = 0.60,
    sigma_m_0: float = 0.30,
    sigma_m_1: float = 0.65,
    m_feedback: float = 1.0,
    vol_noise_scale: float | None = None, # Optional variance control
    # TASK B: Leaky simulator - AR(1) noise for temporal regime signal
    leak_phi_r0: float = 0.0,  # AR(1) coefficient for regime 0 (default: i.i.d.)
    leak_phi_r1: float = 0.0,  # AR(1) coefficient for regime 1 (default: i.i.d.)
):
    """
    Simulate a controlled regime-switching environment with:
      - Heston-like stochastic volatility for prices
      - a volume proxy V_t
      - a regime-dependent sign flip in the mapping V_t -> impact lambda_t

    Regime meaning:
      regime = 0: high volume => lower impact (lambda ~ 1 / volume)
      regime = 1: high volume => higher impact (lambda ~ volume)

    Returns:
      S: (n_paths, n_steps+1) price paths
      v: (n_paths, n_steps+1) variance paths (nonnegative)
      vol_proxy: (n_paths, n_steps) volume proxy V_t (positive)
      lam: (n_paths, n_steps) impact coefficient lambda_t (positive)
      meta: dict of params used
    """
    if regime not in (0, 1):
        raise ValueError("regime must be 0 or 1")

    rng = np.random.default_rng(seed + 10_000 * regime)
    dt = T / n_steps
    eps = 1e-10

    # Correlated Brownian increments for price and variance
    z1 = rng.standard_normal((n_paths, n_steps))
    z2 = rng.standard_normal((n_paths, n_steps))
    w_s = z1
    w_v = rho * z1 + np.sqrt(max(1.0 - rho * rho, 0.0)) * z2

    # Allocate arrays
    S = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    v = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    S[:, 0] = s0
    v[:, 0] = max(v0, eps)

    # Latent states for microstructure semantics (log-AR(1))
    # alpha_t: toxicity proxy (higher in stress)
    # m_t: funding tightness proxy (higher in stress)
    alpha_bar = alpha_bar_0 if regime == 0 else alpha_bar_1
    m_bar = m_bar_0 if regime == 0 else m_bar_1
    sigma_alpha = sigma_alpha_0 if regime == 0 else sigma_alpha_1
    sigma_m = sigma_m_0 if regime == 0 else sigma_m_1

    log_alpha = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    log_m = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    log_alpha[:, 0] = alpha_bar + 0.1 * rng.standard_normal(n_paths)
    log_m[:, 0] = m_bar + 0.1 * rng.standard_normal(n_paths)
    z_alpha = rng.standard_normal((n_paths, n_steps))
    z_m = rng.standard_normal((n_paths, n_steps))

    # Simulate Heston with a stable "full truncation" style discretization
    for t in range(n_steps):
        vt = np.maximum(v[:, t], eps)

        # variance update: v_{t+1} = v_t + kappa(theta - v_t) dt + xi sqrt(v_t) sqrt(dt) * w_v
        v_next = vt + kappa * (theta - vt) * dt + xi * np.sqrt(vt * dt) * w_v[:, t]
        v_next = np.maximum(v_next, eps)
        v[:, t + 1] = v_next

        # price update (log-Euler):
        # log S_{t+1} = log S_t + (mu - 0.5 v_t) dt + sqrt(v_t dt) * w_s
        S[:, t + 1] = S[:, t] * np.exp((mu - 0.5 * vt) * dt + np.sqrt(vt * dt) * w_s[:, t])

        # update latent states with mild feedback from absolute returns
        log_ret_t = np.log((S[:, t + 1] + eps) / (S[:, t] + eps))
        log_alpha[:, t + 1] = (
            log_alpha[:, t]
            + kappa_alpha_lat * (alpha_bar - log_alpha[:, t]) * dt
            + sigma_alpha * np.sqrt(dt) * z_alpha[:, t]
        )
        log_m[:, t + 1] = (
            log_m[:, t]
            + kappa_m_lat * (m_bar - log_m[:, t]) * dt
            + sigma_m * np.sqrt(dt) * z_m[:, t]
            + m_feedback * np.abs(log_ret_t)
        )

    # Build a volume proxy
    # Intuition: more volatility and larger absolute returns => more volume.
    log_ret = np.log(S[:, 1:] / (S[:, :-1] + eps))
    inst_vol = np.sqrt(np.maximum(v[:, :-1], eps))

    # Positive proxy with some noise, then exponentiate to ensure positivity
    vol_signal = 0.7 * np.abs(log_ret) / (np.mean(np.abs(log_ret)) + eps) + 0.7 * inst_vol / (np.mean(inst_vol) + eps)
    
    # Variance Matching Logic
    # RESEARCH HARDENING: Force identical noise across regimes
    # This isolates semantic inversion from noise levels
    if vol_noise_scale is not None:
        noise_scale = vol_noise_scale
    else:
        noise_scale = 0.30  # Constant for both regimes
    
    # TASK B: AR(1) noise for temporal regime signal
    # Select phi based on regime
    phi = leak_phi_r0 if regime == 0 else leak_phi_r1
    
    if abs(phi) < 1e-8:
        # Original i.i.d. behavior (backward compatible)
        vol_noise = rng.normal(loc=0.0, scale=noise_scale, size=(n_paths, n_steps))
    else:
        # AR(1) noise with variance matching
        # For stationary AR(1): Var(X) = σ²_innov / (1 - φ²)
        # To match stationary variance = noise_scale², set:
        # σ²_innov = noise_scale² * (1 - φ²)
        sigma_innov = noise_scale * np.sqrt(max(1.0 - phi**2, 1e-8))
        
        vol_noise = np.zeros((n_paths, n_steps))
        # Initialize from stationary distribution
        vol_noise[:, 0] = rng.normal(0, noise_scale, n_paths)
        for t in range(1, n_steps):
            innovation = rng.normal(0, sigma_innov, n_paths)
            vol_noise[:, t] = phi * vol_noise[:, t-1] + innovation
    
    vol_proxy = np.exp(np.log(vol_signal + 0.2) + vol_noise)  # always positive

    # Regime-dependent sign flip for impact lambda
    # regime 0: g(V)=1/V (slope < 0), regime 1: g(V)=V (slope > 0)
    # The conflict set is the set of states where signs differ. Here, it is the entire domain.
    is_conflict_set = np.ones(n_paths, dtype=bool)

    if regime == 0:
        g = 1.0 / (vol_proxy + eps)
    else:
        g = vol_proxy

    log_alpha_t = log_alpha[:, :-1]
    log_m_t = log_m[:, :-1]
    lam = lam0 * g * np.exp(kappa_alpha * log_alpha_t) * np.exp(kappa_m * log_m_t)

    meta = dict(
        regime=regime,
        n_paths=n_paths,
        n_steps=n_steps,
        T=T,
        dt=dt,
        s0=s0,
        v0=v0,
        mu=mu,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
        lam0=lam0,
        kappa_alpha=kappa_alpha,
        kappa_m=kappa_m,
        alpha_bar_0=alpha_bar_0,
        alpha_bar_1=alpha_bar_1,
        m_bar_0=m_bar_0,
        m_bar_1=m_bar_1,
        kappa_alpha_lat=kappa_alpha_lat,
        kappa_m_lat=kappa_m_lat,
        sigma_alpha_0=sigma_alpha_0,
        sigma_alpha_1=sigma_alpha_1,
        sigma_m_0=sigma_m_0,
        sigma_m_1=sigma_m_1,
        m_feedback=m_feedback,
        is_conflict_set=is_conflict_set, # Global semantic inversion
        leak_phi_r0=leak_phi_r0,
        leak_phi_r1=leak_phi_r1,
    )

    return S, v, vol_proxy, lam, meta

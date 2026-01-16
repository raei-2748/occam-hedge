"""
Controlled simulator for semantic inversion experiments.

This module implements the minimal simulator construction that produces latent fragility
under semantic inversion, as described in Section 4 (Methodology) of the paper.

Key Simulator Functions:
    - simulate_heston_signflip(): Primary variance-matched simulator with semantic inversion
    - simulate_heston_markov_switching(): Markov regime-switching variant
    - simulate_heston_adversarial(): Stress test with tail jumps

Semantic Inversion Mechanism (Section 4.3, Eq. 4.3):
    The volume-impact relationship flips sign across regimes:
    - Regime 0: λ₀(Vol) ∝ 1/Vol  (high volume → low impact, liquid regime)
    - Regime 1: λ₁(Vol) ∝ Vol    (high volume → high impact, constrained regime)

Variance-Matched Control (Section 4.4):
    Volume proxy Vol_t is constructed as independent lognormal with matched variance
    across regimes (σ_v,0 = σ_v,1 = 0.30), eliminating distributional leakage and
    ensuring degradation is purely attributable to semantic inversion.

Paper References:
    - Section 4.2: Underlying dynamics (Heston SV)
    - Section 4.3: Execution costs and semantic inversion
    - Section 4.4: Variance-matched control
"""

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


def simulate_heston_markov_switching(
    p_01: float = 0.05,  # Probability 0 -> 1 per step
    p_10: float = 0.10,  # Probability 1 -> 0 per step
    prior_0: float = 1.0, # Probability of starting in state 0
    n_paths: int = 10_000,
    n_steps: int = 100,
    T: float = 1.0,
    seed: int = 42,
    # Heston shared
    s0: float = 100.0,
    v0: float = 0.04,
    mu: float = 0.00,
    kappa: float = 2.0,
    theta: float = 0.04,
    xi: float = 0.60,
    rho: float = -0.60,
    # Impact shared
    lam0: float = 0.45,
    kappa_alpha: float = 2.5,
    kappa_m: float = 0.8,
    # Regime-dependent params
    alpha_bar_0: float = 0.0,
    alpha_bar_1: float = 0.6,
    m_bar_0: float = 0.0,
    m_bar_1: float = 0.5,
    leak_phi_r0: float = 0.0,
    leak_phi_r1: float = 0.0,
    vol_noise_scale: float = 0.30,
    # Latent dynamics
    kappa_alpha_lat: float = 2.5,
    kappa_m_lat: float = 2.0,
    sigma_alpha_0: float = 0.35,
    sigma_alpha_1: float = 0.60,
    sigma_m_0: float = 0.30,
    sigma_m_1: float = 0.65,
    m_feedback: float = 1.0,
):
    """
    Simulate Heston pathways with synchronous Markov switching of the regime.
    
    The regime R_t evolves: 
    P(R_{t+1}=1 | R_t=0) = p_01
    P(R_{t+1}=0 | R_t=1) = p_10
    
    Returns:
        S, v, vol_proxy, lam, R (regime path), meta
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    eps = 1e-10

    # 1. Simulate Regime Paths
    # R: (n_paths, n_steps) - regime effective at time t
    R = np.zeros((n_paths, n_steps), dtype=int)
    
    # Init R[:, 0]
    # If prior_0 = 1.0, all 0. If 0.5, coin flip.
    initial_r = rng.random(n_paths)
    R[:, 0] = (initial_r > prior_0).astype(int)
    
    for t in range(n_steps - 1):
        # Current regimes
        r_curr = R[:, t]
        
        # Transition probabilities
        # if r=0: flip to 1 with p_01
        # if r=1: flip to 0 with p_10
        flip_prob = np.where(r_curr == 0, p_01, p_10)
        
        # Draws
        flips = rng.random(n_paths) < flip_prob
        
        # Next state: if flip, 1-r, else r
        R[:, t+1] = np.where(flips, 1 - r_curr, r_curr)

    # 2. Heston Dynamics (S, v) - largely regime-independent in this setup
    z1 = rng.standard_normal((n_paths, n_steps))
    z2 = rng.standard_normal((n_paths, n_steps))
    w_s = z1
    w_v = rho * z1 + np.sqrt(max(1.0 - rho * rho, 0.0)) * z2

    S = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    v = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    S[:, 0] = s0
    v[:, 0] = max(v0, eps)

    for t in range(n_steps):
        vt = np.maximum(v[:, t], eps)
        # Euler discretization
        v_next = vt + kappa * (theta - vt) * dt + xi * np.sqrt(vt * dt) * w_v[:, t]
        v[:, t + 1] = np.maximum(v_next, eps)
        S[:, t + 1] = S[:, t] * np.exp((mu - 0.5 * vt) * dt + np.sqrt(vt * dt) * w_s[:, t])

    # 3. Microstructure & Impact
    # These depend on R[:, t]
    
    # Latent Factors
    log_alpha = np.zeros((n_paths, n_steps + 1))
    log_m = np.zeros((n_paths, n_steps + 1))
    
    # Init based on R[:, 0]
    # We'll just define current-step parameters
    for t in range(n_steps):
        r_t = R[:, t]
        # Map regime to params
        alpha_bar_t = np.where(r_t==0, alpha_bar_0, alpha_bar_1)
        m_bar_t = np.where(r_t==0, m_bar_0, m_bar_1)
        sigma_alpha_t = np.where(r_t==0, sigma_alpha_0, sigma_alpha_1)
        sigma_m_t = np.where(r_t==0, sigma_m_0, sigma_m_1)
        
        z_alpha = rng.standard_normal(n_paths)
        z_m = rng.standard_normal(n_paths)
        
        # Log returns for feedback
        if t > 0:
            log_ret = np.log(S[:, t]/(S[:, t-1]+eps))
        else:
            log_ret = np.zeros(n_paths)
            
        # Update
        # Careful: t is index. log_alpha has size n_steps+1.
        # This is a simplification; we update state t to t+1
        # using regime parameters of t 
        log_alpha[:, t+1] = (
            log_alpha[:, t] + 
            kappa_alpha_lat * (alpha_bar_t - log_alpha[:, t]) * dt + 
            sigma_alpha_t * np.sqrt(dt) * z_alpha
        )
        
        log_m[:, t+1] = (
            log_m[:, t] + 
            kappa_m_lat * (m_bar_t - log_m[:, t]) * dt + 
            sigma_m_t * np.sqrt(dt) * z_m + 
            m_feedback * np.abs(log_ret)
        )

    # 4. Volume Proxy
    # We need temporal correlation (AR noise), but phi changes with regime!
    # "Heteroskedastic AR(1)"
    
    # Base signal
    log_ret_all = np.log(S[:, 1:] / (S[:, :-1] + eps))
    inst_vol = np.sqrt(np.maximum(v[:, :-1], eps))
    vol_signal = 0.7 * np.abs(log_ret_all) / (np.mean(np.abs(log_ret_all)) + eps) + \
                 0.7 * inst_vol / (np.mean(inst_vol) + eps)
    
    # Noise gen
    vol_noise = np.zeros((n_paths, n_steps))
    vol_noise[:, 0] = rng.normal(0, vol_noise_scale, n_paths)
    
    for t in range(1, n_steps):
        r_current = R[:, t]
        # If any leakage, use AR prop.
        # But 'phi' depends on path-specific regime
        curr_phi = np.where(r_current==0, leak_phi_r0, leak_phi_r1)
        
        # Variance matching for innovation
        sigma_innov = vol_noise_scale * np.sqrt(np.maximum(1.0 - curr_phi**2, 1e-8))
        z_innov = rng.standard_normal(n_paths)
        
        vol_noise[:, t] = curr_phi * vol_noise[:, t-1] + sigma_innov * z_innov

    vol_proxy = np.exp(np.log(vol_signal + 0.2) + vol_noise)

    # 5. Impact Lambda (Semantic Inversion)
    # R=0 => 1/V, R=1 => V
    
    # Construct g(V)
    # R is (n_paths, n_steps), vol_proxy is (n_paths, n_steps)
    g_R0 = 1.0 / (vol_proxy + eps)
    g_R1 = vol_proxy
    g = np.where(R == 0, g_R0, g_R1)
    
    lam = lam0 * g * np.exp(kappa_alpha * log_alpha[:, :-1]) * np.exp(kappa_m * log_m[:, :-1])
    
    meta = dict(
        p_01=p_01, p_10=p_10,
        leak_phi_r0=leak_phi_r0, leak_phi_r1=leak_phi_r1,
        seed=seed
    )
    
    return S, v, vol_proxy, lam, R, meta

def simulate_heston_adversarial(
    p_01: float = 0.05,
    p_10: float = 0.10,
    prior_0: float = 1.0,
    n_paths: int = 10_000,
    n_steps: int = 100,
    T: float = 1.0,
    seed: int = 42,
    # Heston
    s0: float = 100.0,
    v0: float = 0.04,
    kappa: float = 2.0,
    theta: float = 0.04,
    xi: float = 0.60,
    rho: float = -0.60,
    # Jumps (Section 931)
    jump_intensity: float = 0.1, # Lambda for Poisson
    jump_size_mean: float = 0.0,
    jump_size_std: float = 0.02, # 2% jumps
    # Toxicity Jumps
    alpha_jump_intensity: float = 0.05,
    alpha_jump_size: float = 0.5, # Sudden spike in log_alpha
    # Impact
    lam0: float = 0.45,
    kappa_alpha: float = 2.5,
    kappa_m: float = 0.8,
    vol_noise_scale: float = 0.30
):
    """
    Combines Markov-switching, Heston Vol, and Tail Jumps.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    eps = 1e-10

    # 1. Regimes
    R = np.zeros((n_paths, n_steps), dtype=int)
    R[:, 0] = (rng.random(n_paths) > prior_0).astype(int)
    for t in range(n_steps - 1):
        flips = rng.random(n_paths) < np.where(R[:, t] == 0, p_01, p_10)
        R[:, t+1] = np.where(flips, 1 - R[:, t], R[:, t])

    # 2. Heston + Jumps
    S = np.zeros((n_paths, n_steps + 1))
    v = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = s0
    v[:, 0] = v0
    
    # Pre-gen Poisson jumps
    # Price Jumps
    n_jumps = rng.poisson(jump_intensity * dt, (n_paths, n_steps))
    jump_sizes = rng.normal(jump_size_mean, jump_size_std, (n_paths, n_steps))
    
    # Toxicity Jumps
    n_alpha_jumps = rng.poisson(alpha_jump_intensity * dt, (n_paths, n_steps))
    
    log_alpha = np.zeros((n_paths, n_steps + 1))
    log_m = np.zeros((n_paths, n_steps + 1))

    for t in range(n_steps):
        vt = np.maximum(v[:, t], eps)
        z1, z2 = rng.standard_normal(n_paths), rng.standard_normal(n_paths)
        ws = z1
        wv = rho * z1 + np.sqrt(1 - rho**2) * z2
        
        # Variance update
        v[:, t+1] = np.maximum(vt + kappa*(theta - vt)*dt + xi*np.sqrt(vt*dt)*wv, eps)
        
        # Price update with jumps
        jump_impact = n_jumps[:, t] * jump_sizes[:, t]
        S[:, t+1] = S[:, t] * np.exp((-0.5*vt)*dt + np.sqrt(vt*dt)*ws + jump_impact)
        
        # Latent updates with toxicity jumps
        r_t = R[:, t]
        a_bar = np.where(r_t==0, 0.0, 0.6)
        m_bar = np.where(r_t==0, 0.0, 0.5)
        
        log_alpha[:, t+1] = (log_alpha[:, t] + 2.5*(a_bar - log_alpha[:, t])*dt + 
                             0.35 * np.sqrt(dt) * rng.standard_normal(n_paths) + 
                             n_alpha_jumps[:, t] * alpha_jump_size)
        
        log_m[:, t+1] = log_m[:, t] + 2.0*(m_bar - log_m[:, t])*dt + 0.3 * np.sqrt(dt) * rng.standard_normal(n_paths)

    # 3. Volume & Impact
    log_ret = np.log(S[:, 1:] / (S[:, :-1] + eps))
    inst_vol = np.sqrt(np.maximum(v[:, :-1], eps))
    vol_signal = np.abs(log_ret) + inst_vol
    vol_noise = rng.normal(0, vol_noise_scale, (n_paths, n_steps))
    vol_proxy = np.exp(np.log(vol_signal + 0.1) + vol_noise)
    
    g = np.where(R == 0, 1.0/(vol_proxy + eps), vol_proxy)
    lam = lam0 * g * np.exp(kappa_alpha * log_alpha[:, :-1]) * np.exp(kappa_m * log_m[:, :-1])
    
    return S, v, vol_proxy, lam, R, {}

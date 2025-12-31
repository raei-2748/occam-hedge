
import numpy as np
from scipy.stats import norm

def evaluate_path_diagnostics(
    S: np.ndarray,
    V: np.ndarray,
    lam: np.ndarray,
    T: float,
    K: float,
    vol_hat: float,
    representation: str,
    weights: np.ndarray,
) -> dict:
    """
    Evaluates policy and returns path-level diagnostics (Mean Volume, Total Turnover, Total Cost).
    Used for mechanism concentration analysis.
    """
    dt = T / (S.shape[1] - 1)
    n_paths = S.shape[0]
    n_steps = V.shape[1] 
    
    # Calculate Features
    X_list = []
    # Vectorized loop over steps
    for t in range(n_steps):
        S_t = S[:, t]
        
        # 1. Structural (delta)
        # Avoid div by zero at maturity
        ttl = max(T - t*dt, 1e-4)
        d1 = (np.log(S_t/K) + 0.5 * vol_hat**2 * ttl) / (vol_hat * np.sqrt(ttl))
        delta_bs = norm.cdf(d1)
        
        # 2. Microstructure (volume proxy)
        micro_feat = V[:, t]
        
        if representation == "greeks":
            features = delta_bs[:, None]
            
        elif representation == "micro":
            # For micro-only, typically we might include moneyness or just V.
            # Based on standard logic in experiment_occam.occam_features_numpy:
            # if rep == "micro": feat = [V_t]
            features = micro_feat[:, None]
            
        elif representation == "combined":
            features = np.stack([delta_bs, micro_feat], axis=1)
        
        else:
            raise ValueError(f"Unknown rep {representation}")
            
        X_list.append(features)
        
    X = np.stack(X_list, axis=1) # (N, T, Dim)
    
    # Compute Actions: a_t = X_t @ weights
    # weights: (Dim,)
    actions = np.einsum('ntd,d->nt', X, weights)
    
    # Compute Trades: Delta a_t = a_t - a_{t-1}, a_{-1}=0
    actions_padded = np.concatenate([np.zeros((n_paths, 1)), actions], axis=1)
    trades = np.diff(actions_padded, axis=1) # (N, T)
    
    # Metrics per path
    turnover_per_path = np.sum(np.abs(trades), axis=1)
    
    # lam is usually 2D (N, T) matching steps. If V is (N, T), lam is (N, T).
    # ensure shape match. S is (N, N_STEPS), V is (N, N_STEPS).
    # Cost = 0.5 * lambda * trade^2
    costs_per_path = np.sum(0.5 * lam * trades**2, axis=1)
    
    # Mean volume per path
    mean_volume_per_path = np.mean(V, axis=1)
    
    return {
        "turnover": turnover_per_path,
        "cost": costs_per_path,
        "volume": mean_volume_per_path
    }

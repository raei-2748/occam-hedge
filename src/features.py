
import torch
import numpy as np
import math

def get_feature_dim(representation: str, include_prev_action: bool = False, micro_lags: int = 0) -> int:
    """
    Returns the input dimension for a given representation.
    
    Args:
        representation: Feature set to use ('greeks', 'micro', 'combined', 'oracle')
        include_prev_action: Whether to append previous action a_{t-1}
        micro_lags: Number of lagged micro signals to append (K means [V_{t-1}, ..., V_{t-K}])
    """
    if representation == "greeks":
        base_dim = 1
    elif representation == "micro":
        base_dim = 3
    elif representation == "combined":
        base_dim = 2
    elif representation == "oracle":
        base_dim = 4
    else:
        raise ValueError(f"Unknown representation: {representation}")
    
    # Lagged micro signals add K extra features for representations that include micro
    micro_expansion = micro_lags if representation in ["micro", "combined"] else 0
    
    return base_dim + micro_expansion + (1 if include_prev_action else 0)

def bs_delta_call_torch(S: torch.Tensor, K: float, tau: torch.Tensor, vol: float, r: float = 0.0) -> torch.Tensor:
    eps = 1e-12
    tau = torch.clamp(tau, min=eps)
    vol = max(vol, 1e-12)
    
    d1 = (torch.log((S + eps) / K) + (r + 0.5 * vol * vol) * tau) / (vol * torch.sqrt(tau))
    return 0.5 * (1.0 + torch.erf(d1 / math.sqrt(2.0)))

def micro_signal_torch(V_t: torch.Tensor) -> torch.Tensor:
    med = torch.median(V_t)
    Vn = V_t / (med + 1e-12)
    return torch.tanh(torch.log(Vn + 1e-12))

def occam_features_torch(
    representation: str,
    S_t: torch.Tensor,
    tau_t: torch.Tensor,
    V_t: torch.Tensor,
    K: float,
    vol_hat: float,
    R_t: torch.Tensor | None = None,
    a_prev: torch.Tensor | None = None,
    include_prev_action: bool = False,
    micro_lags: int = 0,
    V_history: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Construct observation vector for policy.
    
    Args:
        representation: Feature set to use
        S_t: Current stock price (B,)
        tau_t: Time to maturity (B,)
        V_t: Current volume proxy (B,)
        K: Strike price
        vol_hat: Estimated volatility
        R_t: Regime label (B,) - only for 'oracle' representation
        a_prev: Previous action (B,) - for Markovity
        include_prev_action: Whether to append a_prev
        micro_lags: Number of lagged micro signals K (appends V_{t-1}, ..., V_{t-K})
        V_history: Tensor of lagged micro signals (B, micro_lags), already transformed
    """
    # Build base features
    if representation == "greeks":
        delta = bs_delta_call_torch(S_t, K=K, tau=tau_t, vol=vol_hat)
        features = delta.unsqueeze(1)  # (B, 1)
    elif representation == "micro":
        eps = 1e-6
        log_mon = torch.log((S_t + eps) / K)
        micro = micro_signal_torch(V_t)
        features = torch.stack([log_mon, tau_t, micro], dim=1)  # (B, 3)
    elif representation == "combined":
        delta = bs_delta_call_torch(S_t, K=K, tau=tau_t, vol=vol_hat)
        micro = micro_signal_torch(V_t)
        features = torch.stack([delta, micro], dim=1)  # (B, 2)
    elif representation == "oracle":
        log_mon = torch.log(S_t / K)
        micro = micro_signal_torch(V_t)
        if R_t is None:
            raise ValueError("representation='oracle' requires R_t")
        features = torch.stack([log_mon, tau_t, micro, R_t], dim=1)  # (B, 4)
    else:
        raise ValueError(f"Unknown representation: {representation}")
    
    # Append lagged micro signals for temporal regime inference
    # V_history should be (B, micro_lags) with already-transformed micro signals
    if micro_lags > 0 and representation in ["micro", "combined"]:
        if V_history is not None:
            # V_history contains lagged micro signals [V_{t-1}, V_{t-2}, ..., V_{t-K}]
            features = torch.cat([features, V_history], dim=1)
        else:
            # Pad with zeros (for t < micro_lags)
            batch_size = S_t.shape[0]
            padding = torch.zeros(batch_size, micro_lags, device=S_t.device)
            features = torch.cat([features, padding], dim=1)
    
    # Append previous action if requested
    if include_prev_action:
        if a_prev is None:
            # Default to zero for t=0
            a_prev = torch.zeros(S_t.shape[0], device=S_t.device)
        features = torch.cat([features, a_prev.unsqueeze(1)], dim=1)
    
    return features


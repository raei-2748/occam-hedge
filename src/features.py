
import torch
import numpy as np
import math

def get_feature_dim(representation: str) -> int:
    """Returns the input dimension for a given representation."""
    if representation == "greeks":
        return 1
    if representation == "micro":
        return 3
    if representation == "combined":
        return 2
    if representation == "oracle":
        return 4
    raise ValueError(f"Unknown representation: {representation}")

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
) -> torch.Tensor:
    if representation == "greeks":
        delta = bs_delta_call_torch(S_t, K=K, tau=tau_t, vol=vol_hat)
        return delta.unsqueeze(1) # (B, 1)
    if representation == "micro":
        # Base state + Micro signal
        # Base: log_moneyness, tau
        eps = 1e-6
        log_mon = torch.log((S_t + eps) / K)
        micro = micro_signal_torch(V_t)
        return torch.stack([log_mon, tau_t, micro], dim=1) # (B, 3)
    if representation == "combined":
        delta = bs_delta_call_torch(S_t, K=K, tau=tau_t, vol=vol_hat)
        micro = micro_signal_torch(V_t)
        return torch.stack([delta, micro], dim=1)
    if representation == "oracle":
        # Includes true regime label R_t
        log_mon = torch.log(S_t / K)
        micro = micro_signal_torch(V_t)
        if R_t is None:
            raise ValueError("representation='oracle' requires R_t")
        return torch.stack([log_mon, tau_t, micro, R_t], dim=1)
    raise ValueError(f"Unknown representation: {representation}")

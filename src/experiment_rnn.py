
import torch
import torch.nn as nn
from risk import es_loss_torch, robust_risk_torch
from features import occam_features_torch

def compute_hedging_losses_rnn(
    model, S, V, lam, T, K, vol_hat, representation, action_clip=0.20,
    R=None, impact_power=2.0
):
    """
    Vectorized version of compute_hedging_losses for RNN policies.
    Assumes S, V are exogenous.
    
    Args:
        S: (B, T+1)
        V: (B, T)
        lam: (B, T)
    """
    device = S.device
    B, n_steps = V.shape
    dt = T / n_steps
    tau = torch.linspace(T, 0, n_steps+1, device=device) # (T+1)
    
    # 1. Feature Construction (Vectorized)
    # We need to construct (B, T, D)
    # occam_features_torch is mostly elementwise.
    # Let's import a helper or simply perform the stack here.
    
    # Time to maturity matrix: (B, T)
    tau_mat = tau[:-1].unsqueeze(0).expand(B, n_steps) 
    
    # Construct features using the vectorized helper (to be implemented/verified)
    from features import occam_features_torch
    
    # NOTE: To perform fully vectorized feature construction efficiently, 
    # we might need to modify features.py. 
    # But strictly speaking, we can just map the existing function? No, slow.
    # Let's assume we can construct columns easily.
    
    # For 'combined': [Money-ness, Time, Vol_Signal, Time] (Simplified)
    # Moneyness: log(S_t/K)
    log_moneyness = torch.log(S[:, :-1] / K)
    
    # Signal: V_t (standardized likely)
    # The standardizer is inside the policy? No, features.py assumes raw? 
    # Actually features.py uses raw V_t usually.
    
    # Let's replicate `occam_features_torch` logic here inline for vectorization speed
    # or Assume `features.py` has a vector mode.
    # Existing `occam_features_torch` takes `S_t` (B,) and `V_t` (B,).
    # We can pass `S[:, :-1]` (B, T) and `V` (B, T) if it supports broadcasting?
    # Checking `features.py` (assumed based on memory): usually operations are elementwise.
    
    # ...
    # Call features on flattened tensors (B*T) then reshape?
    S_flat = S[:, :-1].reshape(-1)
    tau_flat = tau_mat.reshape(-1)
    V_flat = V.reshape(-1)
    
    if R is not None:
        R_flat = R.reshape(-1)
    else:
        R_flat = None
        
    feat_flat = occam_features_torch(
        representation, S_flat, tau_flat, V_flat, K, vol_hat, R_t=R_flat
    )
    # feat_flat is (B*T, D)
    
    # Reshape to (B, T, D) for RNN
    # We need input dim from feat_flat
    D = feat_flat.shape[-1]
    X = feat_flat.view(B, n_steps, D)
    
    # 2. Forward Pass
    # actions: (B, T), mus: list[(B, T, Latent)], logvars: list[(B, T, Latent)]
    actions, mus, logvars = model(X)
    
    actions = torch.clamp(actions, -action_clip, action_clip)
    
    # 3. PnL / Loss Calculation (Vectorized)
    # PnL = sum( a_t * dS_t )
    dS = S[:, 1:] - S[:, :-1] # (B, T)
    pnl_path = torch.sum(actions * dS, dim=1)
    
    # Cost = sum( 0.5 * lam_t * |da_t|^p )
    # da_t = a_t - a_{t-1}. a_{-1} = 0.
    # Pad actions with 0 at left
    a_padded = torch.cat([torch.zeros(B, 1, device=device), actions], dim=1) # (B, T+1)
    da = a_padded[:, 1:] - a_padded[:, :-1] # (B, T)
    
    if abs(impact_power - 2.0) < 1e-6:
        cost_path = torch.sum(0.5 * lam * (da**2), dim=1)
    else:
        cost_path = torch.sum(0.5 * lam * (torch.abs(da)**impact_power), dim=1)
        
    # Payoff
    payoff = torch.relu(S[:, -1] - K)
    
    losses = (payoff - pnl_path + cost_path) / S[:, 0]
    
    # 4. Info Cost (Average over T and Channels)
    # Each mu/logvar is (B, T, LatentDim)
    total_kl_per_channel = []
    
    for mu, logvar in zip(mus, logvars):
        # KL per step per batch
        # kld = -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) dim=-1
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1) # (B, T)
        mean_kl = torch.mean(kld) # Mean over B and T
        total_kl_per_channel.append(mean_kl)
        
    total_info_cost = sum(total_kl_per_channel)
    
    return losses, total_info_cost, total_kl_per_channel



import json
import time
import shutil
import logging
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import io
import math

from utils import set_seeds, get_git_revision_hash
from paper_config import Config, run_id_from_config
from world import simulate_heston_signflip
from risk import robust_es_kl, robust_risk_torch, es_loss_torch
from policies import bs_delta_call, FactorizedVariationalPolicy

ROOT = Path(__file__).resolve().parents[1]

from features import occam_features_torch, get_feature_dim

def bs_delta_call_torch(S, K, tau, vol, r=0.0):
    device = S.device
    eps = 1e-12
    # Ensure tau matches S shape or is broadcastable
    tau = torch.maximum(tau, torch.tensor(eps, device=device))
    vol = max(vol, 1e-12)
    
    d1 = (torch.log((S + eps) / K) + (r + 0.5 * vol * vol) * tau) / (vol * torch.sqrt(tau))
    return 0.5 * (1.0 + torch.erf(d1 / math.sqrt(2.0)))

def compute_hedging_losses_torch(
    model: nn.Module,
    S: torch.Tensor,
    V: torch.Tensor,
    lam: torch.Tensor,
    T: float,
    K: float,
    vol_hat: float,
    representation: str,
    action_clip: float = 5.0,
    impact_power: float = 2.0,
    R: torch.Tensor | None = None,
    micro_lags: int = 0,
    include_prev_action: bool = False,
):
    """
    Differentiable simulation of hedging with VIB.
    Updated to return per-channel information costs.
    
    TASK 1: Supports lagged micro signals via micro_lags parameter.
    TASK 2: Supports previous action via include_prev_action parameter.
    """
    n_paths = S.shape[0]
    actual_steps = S.shape[1] - 1
    tau_grid = torch.linspace(T, 0.0, actual_steps + 1, device=S.device)
    
    a = torch.zeros(n_paths, device=S.device)
    pnl = torch.zeros(n_paths, device=S.device)
    cost = torch.zeros(n_paths, device=S.device)
    
    # Initialize separate KL accumulators for each feature channel
    # We need to know input_dim to init this, or we can infer it from the first pass.
    # We'll initialize it as None and create it on the first step.
    total_kl_per_channel = None 
    
    # TASK 1: Maintain a rolling buffer of transformed micro signals for temporal inference
    # Buffer stores the K most recent micro signals (already transformed via micro_signal_torch)
    micro_buffer = []  # List of (n_paths,) tensors
    
    for t in range(actual_steps):
        S_t = S[:, t]
        tau_t = torch.full((n_paths,), tau_grid[t], device=S.device)
        
        R_t = R[:, t] if R is not None else None
        
        # TASK 1: Build V_history from micro buffer
        # V_history should be (n_paths, micro_lags) with [V_{t-1}, V_{t-2}, ..., V_{t-K}]
        if micro_lags > 0 and representation in ["micro", "combined"]:
            if len(micro_buffer) >= micro_lags:
                # Take the most recent micro_lags entries (reversed so t-1 is first)
                V_history = torch.stack(micro_buffer[-micro_lags:][::-1], dim=1)
            elif len(micro_buffer) > 0:
                # Partial history - pad with zeros
                available = torch.stack(micro_buffer[::-1], dim=1)  # (n_paths, len(micro_buffer))
                padding = torch.zeros(n_paths, micro_lags - len(micro_buffer), device=S.device)
                V_history = torch.cat([available, padding], dim=1)
            else:
                # No history yet - all zeros
                V_history = None  # Will be handled by occam_features_torch
        else:
            V_history = None
        
        # TASK 2: Pass previous action for Markovity
        a_prev = a if include_prev_action else None
        
        feats = occam_features_torch(
            representation, S_t, tau_t, V[:, t], K, vol_hat, 
            R_t=R_t, 
            a_prev=a_prev,
            include_prev_action=include_prev_action,
            micro_lags=micro_lags,
            V_history=V_history
        )
        
        # VIB Forward
        # mus, logvars are lists of length input_dim
        action, mus, logvars = model(feats)
        
        # Initialize accumulator if first step
        if total_kl_per_channel is None:
            total_kl_per_channel = torch.zeros(len(mus), device=S.device)

        a_new = torch.clamp(action, -action_clip, action_clip)
        da = a_new - a
        dS = S[:, t + 1] - S_t
        
        pnl = pnl + a_new * dS
        
        # Cost with optional power law
        if abs(impact_power - 2.0) < 1e-6:
             cost = cost + 0.5 * lam[:, t] * (da ** 2)
        else:
             cost = cost + 0.5 * lam[:, t] * (torch.abs(da) ** impact_power)
        
        # Analytic KL Divergence Loss per channel
        # KL(N(mu, sigma) || N(0, 1)) = 0.5 * sum(sigma^2 + mu^2 - 1 - log(sigma^2))
        for i, (mu, logvar) in enumerate(zip(mus, logvars)):
            # Sum over batch dimension, we will average later
            # Shape of mu: (B, latent_dim_per_feature)
            kld_batch = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            total_kl_per_channel[i] += torch.mean(kld_batch) # Average over batch immediately to keep scale
        
        # TASK 1: Update micro buffer for next step
        if micro_lags > 0 and representation in ["micro", "combined"]:
            from features import micro_signal_torch
            current_micro = micro_signal_torch(V[:, t])
            micro_buffer.append(current_micro)
            # Keep buffer size bounded
            if len(micro_buffer) > micro_lags:
                micro_buffer.pop(0)
        
        a = a_new
        
    payoff = torch.relu(S[:, -1] - K)
    losses = payoff - pnl + cost  # Cost is a PENALTY (positive)
    
    # Normalize losses by initial stock price to match dimensionless beta
    losses = losses / S[:, 0]

    # Average KL over time steps
    avg_kl_per_channel = total_kl_per_channel / actual_steps
    total_info_cost = torch.sum(avg_kl_per_channel)
        
    return losses, total_info_cost, avg_kl_per_channel

# --- Configuration & Main ---

def train_model(
    model: nn.Module,
    S: torch.Tensor,
    V: torch.Tensor,
    lam: torch.Tensor,
    training_config: dict,
    representation: str,
    T: float,
    K: float,
    vol_hat: float,
    q_init: float = 0.0,
    R: torch.Tensor | None = None,
) -> dict:
    
    beta = training_config["beta"]
    train_eta = training_config["train_eta"]
    train_lambdas = torch.tensor(training_config["train_lambdas"], device=S.device, dtype=torch.float32)
    gamma = training_config["gamma"]
    
    # Parse beta parameter: can be float (uniform) or dict (hierarchical)
    if isinstance(beta, dict):
        beta_price = beta.get("beta_price", 0.0)
        beta_micro = beta.get("beta_micro", 0.0)
        hierarchical = True
    else:
        # Backward compatible: single beta applied to all channels
        beta_price = beta
        beta_micro = beta
        hierarchical = False
    
    lr = training_config.get("lr", 0.001) # Reduced default LR for stability
    n_epochs = training_config.get("n_epochs", 200)
    warmup_epochs = training_config.get("warmup_epochs", 50)
    
    # TASK 1 & 2: Extract lag and previous action settings
    micro_lags = training_config.get("micro_lags", 0)
    include_prev_action = training_config.get("include_prev_action", False)
    
    q_param = nn.Parameter(torch.tensor(float(q_init), device=S.device))
    optimizer = optim.Adam(list(model.parameters()) + [q_param], lr=lr)
    
    history = []
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        losses, info_cost, info_components = compute_hedging_losses_torch(
            model, S, V, lam, T, K, vol_hat, representation, 
            R=R, micro_lags=micro_lags, include_prev_action=include_prev_action
        )
        
        # Apply hierarchical or uniform information penalty
        if hierarchical:
            # Differential penalties: beta_price for channel 0 (Delta), beta_micro for channel 1 (Micro)
            # For "combined" representation: info_components[0] = Delta, info_components[1] = Micro
            info_penalty = beta_price * info_components[0] + beta_micro * info_components[1]
        else:
            # Original behavior: uniform penalty across all channels
            info_penalty = beta * info_cost
        
        if epoch < warmup_epochs:
            loss_obj = torch.mean(losses**2) + info_penalty
            mode = "warmup"
        else:
            es_vals = es_loss_torch(losses, q_param, gamma)
            risk = robust_risk_torch(es_vals, train_eta, train_lambdas)
            loss_obj = risk + info_penalty
            mode = "robust"
            
        # --- STABILIZATION: NaN Guardrail ---
        if torch.isnan(loss_obj):
            print(f"CRITICAL: NaN detected at epoch {epoch}. Aborting.")
            break
            
        loss_obj.backward()
        
        # --- STABILIZATION: Gradient Clipping ---
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if epoch % 10 == 0:
            # Create dynamic labels for the components based on representation
            # This helps downstream plotting know what index 0, 1, etc. are.
            info_dict = {f"info_dim_{i}": v.item() for i, v in enumerate(info_components)}
            
            log_entry = {
                "epoch": epoch,
                "loss_obj": float(loss_obj.item()),
                "q": float(q_param.item()),
                "info_total": float(info_cost.item()),
                "mode": mode,
                "hierarchical": hierarchical,
                **info_dict
            }
            if hierarchical:
                log_entry["beta_price"] = float(beta_price)
                log_entry["beta_micro"] = float(beta_micro)
            
            history.append(log_entry)
            
    return {
        "final_weights": model.state_dict(),
        "final_q": q_param.item(),
        "history": history
    }

def hedge_on_paths(
    S, V, lam, T, K, vol_hat, representation, 
    weights_or_state_dict, 
    action_clip=5.0,
    R=None,
    micro_lags=0,
    include_prev_action=False,
):
    """
    Evaluates policy on path using Torch model (CPU/GPU) for consistency with VIB.
    
    TASK 1 & 2: Supports lagged micro signals and previous action.
    """
    # Setup Torch environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S_t = torch.tensor(S, dtype=torch.float32, device=device)
    V_t = torch.tensor(V, dtype=torch.float32, device=device)
    lam_t = torch.tensor(lam, dtype=torch.float32, device=device)
    
    # Reconstruct Model with correct input dimension accounting for lags
    input_dim = get_feature_dim(representation, include_prev_action=include_prev_action, micro_lags=micro_lags)
        
    # Note: latent_dim must match training. User request = 8.
    model = FactorizedVariationalPolicy(input_dim, latent_dim_per_feature=8).to(device)
    
    if isinstance(weights_or_state_dict, dict):
        model.load_state_dict(weights_or_state_dict)
    else:
        # Fallback for legacy numpy weights? Not supported in VIB.
        raise ValueError("hedge_on_paths requires state_dict for VIB policy")
    
    model.eval()
    
    with torch.no_grad():
        losses_t, info_cost_t, _ = compute_hedging_losses_torch(
            model, S_t, V_t, lam_t, T, K, vol_hat, representation, action_clip, 
            R=R, micro_lags=micro_lags, include_prev_action=include_prev_action
        )
        # Quick loop for metrics (using model)
        # We need to flatten both V and da (absolute changes) for wrong-way score
        
        n_paths = S.shape[0]
        n_steps = S.shape[1] - 1
        
        tau_grid = torch.linspace(T, 0.0, n_steps + 1, device=device)

        a = torch.zeros(n_paths, device=device)
        path_cost = torch.zeros(n_paths, device=device)
        path_turnover = torch.zeros(n_paths, device=device)
        
        da_list = []
        V_list = []

 # ... inside hedge_on_paths ...
    
    # Initialize state for the metrics loop
    micro_buffer = []
    a = torch.zeros(n_paths, device=device) # Reset action for the loop
    
    for t in range(n_steps):
        R_t = R[:, t] if R is not None else None
        
        # --- FIX START: Reconstruct History & Prev Action ---
        # 1. Build V_history
        if micro_lags > 0 and representation in ["micro", "combined"]:
            if len(micro_buffer) >= micro_lags:
                V_history = torch.stack(micro_buffer[-micro_lags:][::-1], dim=1)
            elif len(micro_buffer) > 0:
                available = torch.stack(micro_buffer[::-1], dim=1)
                padding = torch.zeros(n_paths, micro_lags - len(micro_buffer), device=device)
                V_history = torch.cat([available, padding], dim=1)
            else:
                V_history = None
        else:
            V_history = None
            
        # 2. Prepare Previous Action
        a_prev = a if include_prev_action else None
        
        # 3. Correct Function Call
        feats = occam_features_torch(
            representation, 
            S_t[:, t], 
            torch.full((n_paths,), tau_grid[t], device=device), 
            V_t[:, t], 
            K, 
            vol_hat, 
            R_t=R_t,
            micro_lags=micro_lags,           # Pass new arg
            include_prev_action=include_prev_action, # Pass new arg
            V_history=V_history,             # Pass history
            a_prev=a_prev                    # Pass prev action
        )
        
        # 4. Update Buffer
        if micro_lags > 0 and representation in ["micro", "combined"]:
            from features import micro_signal_torch
            current_micro = micro_signal_torch(V_t[:, t])
            micro_buffer.append(current_micro)
            if len(micro_buffer) > micro_lags:
                micro_buffer.pop(0)
        # --- FIX END ---

        action, _, _ = model(feats) 
            
        turnover_rate = torch.mean(path_turnover) / n_steps
        exec_cost = torch.mean(path_cost) / n_steps
        
        # Stack and Correlation
        stack_da = torch.stack(da_list, dim=1).flatten().abs()
        stack_V = torch.stack(V_list, dim=1).flatten()
        
        # Pearson Correlation
        vx = stack_da - torch.mean(stack_da)
        vy = stack_V - torch.mean(stack_V)
        da_vol_corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
        
    return losses_t.cpu().numpy(), info_cost_t.item(), turnover_rate.item(), exec_cost.item(), da_vol_corr.item()


def train_weights(
    S, V, lam, T, K, vol_hat, representation, beta, train_eta, train_lambdas, gamma,
    R=None,
    micro_lags=0,
    include_prev_action=False,
    **kwargs
):
    """
    Convenience function to train a VIB policy.
    
    TASK 1 & 2: Supports lagged micro signals and previous action.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get correct input dimension accounting for lags and prev action
    input_dim = get_feature_dim(representation, include_prev_action=include_prev_action, micro_lags=micro_lags)
    
    model = FactorizedVariationalPolicy(input_dim, latent_dim_per_feature=8).to(device)
    
    cfg = {
        "beta": beta,
        "train_eta": train_eta,
        "train_lambdas": train_lambdas,
        "gamma": gamma,
        "lr": 0.001,
        "n_epochs": 150, 
        "warmup_epochs": 50,
        "micro_lags": micro_lags,
        "include_prev_action": include_prev_action,
    }
    cfg.update(kwargs)
    
    S_t = torch.tensor(S, dtype=torch.float32, device=device)
    V_t = torch.tensor(V, dtype=torch.float32, device=device)
    lam_t = torch.tensor(lam, dtype=torch.float32, device=device)
    
    res = train_model(model, S_t, V_t, lam_t, cfg, representation, T, K, vol_hat, R=R)
    return res["final_weights"]

def main():
    # Placeholder for main logic if needed (usually run via run_paper.py)
    pass 

if __name__ == "__main__":
    main()

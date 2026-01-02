
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

from utils import set_seeds, get_git_revision_hash
from paper_config import Config, run_id_from_config
from world import simulate_heston_signflip
from risk import robust_es_kl, robust_risk_torch, es_loss_torch
from policies import bs_delta_call, FactorizedVariationalPolicy

ROOT = Path(__file__).resolve().parents[1]

# --- Torch Helpers ---

def bs_delta_call_torch(S: torch.Tensor, K: float, tau: torch.Tensor, vol: float, r: float = 0.0) -> torch.Tensor:
    eps = 1e-12
    tau = torch.clamp(tau, min=eps)
    vol = max(vol, 1e-12)
    
    d1 = (torch.log((S + eps) / K) + (r + 0.5 * vol * vol) * tau) / (vol * torch.sqrt(tau))
    return 0.5 * (1.0 + torch.erf(d1 / np.sqrt(2.0)))

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
) -> torch.Tensor:
    if representation == "greeks":
        delta = bs_delta_call_torch(S_t, K=K, tau=tau_t, vol=vol_hat)
        return delta.unsqueeze(1) # (B, 1)
    if representation == "micro":
        micro = micro_signal_torch(V_t)
        return micro.unsqueeze(1)
    if representation == "combined":
        delta = bs_delta_call_torch(S_t, K=K, tau=tau_t, vol=vol_hat)
        micro = micro_signal_torch(V_t)
        return torch.stack([delta, micro], dim=1)
    raise ValueError(f"Unknown representation: {representation}")

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
    impact_power: float = 2.0
):
    """
    Differentiable simulation of hedging with VIB.
    """
    n_paths = S.shape[0]
    actual_steps = S.shape[1] - 1
    tau_grid = torch.linspace(T, 0.0, actual_steps + 1, device=S.device)
    
    a = torch.zeros(n_paths, device=S.device)
    pnl = torch.zeros(n_paths, device=S.device)
    cost = torch.zeros(n_paths, device=S.device)
    info = torch.zeros(n_paths, device=S.device)
    
    # Track SNR statistics if evaluating? (Optional)
    
    for t in range(actual_steps):
        S_t = S[:, t]
        tau_t = torch.full((n_paths,), tau_grid[t], device=S.device)
        
        feats = occam_features_torch(representation, S_t, tau_t, V[:, t], K, vol_hat)
        
        # VIB Forward
        # action: (B,), mus: list of (B, L_i), logvars: list of (B, L_i)
        action, mus, logvars = model(feats)
        
        a_new = torch.clamp(action, -action_clip, action_clip)
        
        da = a_new - a
        dS = S[:, t + 1] - S_t
        
        pnl = pnl + a_new * dS
        
        # Cost with optional power law
        if abs(impact_power - 2.0) < 1e-6:
             cost = cost + 0.5 * lam[:, t] * (da ** 2)
        else:
             cost = cost + 0.5 * lam[:, t] * (torch.abs(da) ** impact_power)
        
        # Analytic KL Divergence Loss
        # Sum KL over all factorized encoders
        kl_step = torch.zeros(n_paths, device=S.device)
        for mu, logvar in zip(mus, logvars):
            # KL(N(mu, sigma) || N(0, 1)) = 0.5 * sum(sigma^2 + mu^2 - 1 - log(sigma^2))
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kl_step = kl_step + kld
            
        info = info + kl_step
        
        a = a_new
        
    payoff = torch.relu(S[:, -1] - K)
    losses = payoff - pnl - cost
    info_cost = torch.mean(info)
        
    return losses, info_cost

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
    q_init: float = 0.0
) -> dict:
    
    beta = training_config["beta"]
    train_eta = training_config["train_eta"]
    train_lambdas = torch.tensor(training_config["train_lambdas"], device=S.device, dtype=torch.float32)
    gamma = training_config["gamma"]
    
    lr = training_config.get("lr", 0.005) # Lower LR for VIB stability?
    n_epochs = training_config.get("n_epochs", 200)
    warmup_epochs = training_config.get("warmup_epochs", 50)
    
    q_param = nn.Parameter(torch.tensor(float(q_init), device=S.device))
    optimizer = optim.Adam(list(model.parameters()) + [q_param], lr=lr)
    
    history = []
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        losses, info_cost = compute_hedging_losses_torch(
            model, S, V, lam, T, K, vol_hat, representation
        )
        
        if epoch < warmup_epochs:
            loss_obj = torch.mean(losses**2) + beta * info_cost
            mode = "warmup"
        else:
            es_vals = es_loss_torch(losses, q_param, gamma)
            risk = robust_risk_torch(es_vals, train_eta, train_lambdas)
            loss_obj = risk + beta * info_cost
            mode = "robust"
            
        loss_obj.backward()
        optimizer.step()
        
        # Log basic stats
        if epoch % 10 == 0:
            history.append({
                "epoch": epoch,
                "loss_obj": float(loss_obj.item()),
                "q": float(q_param.item()),
                "info": float(info_cost.item()),
                "mode": mode
            })
            
    return {
        "final_weights": model.state_dict(),
        "final_q": q_param.item(),
        "history": history
    }

def hedge_on_paths(
    S, V, lam, T, K, vol_hat, representation, 
    weights_or_state_dict, 
    action_clip=5.0
):
    """
    Evaluates policy on path using Torch model (CPU/GPU) for consistency with VIB.
    """
    # Setup Torch environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S_t = torch.tensor(S, dtype=torch.float32, device=device)
    V_t = torch.tensor(V, dtype=torch.float32, device=device)
    lam_t = torch.tensor(lam, dtype=torch.float32, device=device)
    
    # Reconstruct Model
    input_dim = 2 if representation == "combined" else 1
    # Note: latent_dim must match training. We assume 2 here per user preference or config
    # Ideally config is passed, but we'll assume default 2 for now
    model = FactorizedVariationalPolicy(input_dim, latent_dim_per_feature=2).to(device)
    
    if isinstance(weights_or_state_dict, dict):
        model.load_state_dict(weights_or_state_dict)
    else:
        # Fallback for legacy numpy weights? Not supported in VIB.
        raise ValueError("hedge_on_paths requires state_dict for VIB policy")
    
    model.eval()
    
    with torch.no_grad():
        losses_t, info_cost_t = compute_hedging_losses_torch(
            model, S_t, V_t, lam_t, T, K, vol_hat, representation, action_clip
        )
        # We need other metrics: turnover, exec_cost
        # Re-run loop or extract? 
        # compute_hedging_losses_torch doesn't return turnover.
        # Let's add turnover to compute_hedging_losses_torch? 
        # Or hack it here using numpy loop if we want exact same metrics?
        # Actually, let's just stick to "Robust ES" logic. 
        # But run_paper.py expects (losses, info_cost, turnover, exec_cost).
        
        # Quick loop for metrics (using model)
        n_paths = S.shape[0]
        n_steps = S.shape[1] - 1
        a = torch.zeros(n_paths, device=device)
        cost = torch.zeros(n_paths, device=device)
        turnover = torch.zeros(n_paths, device=device)
        
        tau_grid = torch.linspace(T, 0.0, n_steps + 1, device=device)
        
        for t in range(n_steps):
            feats = occam_features_torch(representation, S_t[:, t], torch.full((n_paths,), tau_grid[t], device=device), V_t[:, t], K, vol_hat)
            # Use MEAN for evaluation (deterministic policy mode)? 
            # Or sample? VIB usually samples. But for evaluation/hedging we might want Expected Action.
            # "The policy maps the SAMPLED z".
            # If we want deterministic eval, we should use mu.
            # But the policy is trained on samples. 
            # Strict VIB eval should sample.
            action, _, _ = model(feats) 
            a_new = torch.clamp(action, -action_clip, action_clip)
            da = a_new - a
            cost += 0.5 * lam_t[:, t] * da**2
            turnover += torch.abs(da)
            a = a_new
            
        turnover_rate = torch.mean(turnover) / n_steps
        exec_cost = torch.mean(cost) / n_steps
        
    return losses_t.cpu().numpy(), info_cost_t.item(), turnover_rate.item(), exec_cost.item()


def train_weights(
    S, V, lam, T, K, vol_hat, representation, beta, train_eta, train_lambdas, gamma,
    **kwargs
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 2 if representation == "combined" else 1
    
    model = FactorizedVariationalPolicy(input_dim, latent_dim_per_feature=2).to(device)
    
    cfg = {
        "beta": beta,
        "train_eta": train_eta,
        "train_lambdas": train_lambdas,
        "gamma": gamma,
        "lr": 0.005,
        "n_epochs": 150, 
        "warmup_epochs": 30
    }
    cfg.update(kwargs)
    
    S_t = torch.tensor(S, dtype=torch.float32, device=device)
    V_t = torch.tensor(V, dtype=torch.float32, device=device)
    lam_t = torch.tensor(lam, dtype=torch.float32, device=device)
    
    res = train_model(model, S_t, V_t, lam_t, cfg, representation, T, K, vol_hat)
    return res["final_weights"]

def main():
    # Placeholder for main logic if needed (usually run via run_paper.py)
    pass 

if __name__ == "__main__":
    main()

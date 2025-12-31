
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

from utils import set_seeds, get_git_revision_hash
from paper_config import Config, run_id_from_config
from world import simulate_heston_signflip
from risk import robust_es_kl, robust_risk_torch, es_loss_torch
from policies import bs_delta_call

ROOT = Path(__file__).resolve().parents[1]

# --- Torch Helpers ---

def bs_delta_call_torch(S: torch.Tensor, K: float, tau: torch.Tensor, vol: float, r: float = 0.0) -> torch.Tensor:
    eps = 1e-12
    tau = torch.clamp(tau, min=eps)
    vol = max(vol, 1e-12)
    
    d1 = (torch.log((S + eps) / K) + (r + 0.5 * vol * vol) * tau) / (vol * torch.sqrt(tau))
    # CDF of normal
    return 0.5 * (1.0 + torch.erf(d1 / np.sqrt(2.0)))

def micro_signal_torch(V_t: torch.Tensor) -> torch.Tensor:
    # V_t: (B,)
    # median over batch
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
    action_clip: float = 5.0
):
    """
    Differentiable simulation of hedging.
    S, V, lam: (B, T_steps)
    """
    n_paths, n_steps_data = S.shape
    # We iterate t from 0 to n_steps-1
    # tau at step t
    tau_grid = torch.linspace(T, 0.0, n_steps_data + 1, device=S.device)
    
    a = torch.zeros(n_paths, device=S.device)
    pnl = torch.zeros(n_paths, device=S.device)
    cost = torch.zeros(n_paths, device=S.device)
    info = torch.zeros(n_paths, device=S.device) # info cost accumulation
    
    # We assume S has S_0 ... S_T (cols 0..T)? 
    # Usually S has n_steps+1 points?
    # world.py simulate returns (n_paths, n_steps+1)?
    # Let's check world.py or usage. 
    # usage: for t in range(n_steps): S[:, t].
    # S[:, t+1] - S[:, t].
    # So S shape is (B, n_steps+1) theoretically, but usage implied S[:,t] and S[:, t+1].
    # Wait, in original code:
    # S[:, t]  ... dS = S[:, t+1] - S[:, t]
    # So S must have at least n_steps + 1 columns?
    # Let's check world.py.
    # It calls geometric_brownian_motion ...
    # We will assume S is (B, N+1).
    
    actual_steps = S.shape[1] - 1
    
    for t in range(actual_steps):
        S_t = S[:, t]
        tau_t = torch.full((n_paths,), tau_grid[t], device=S.device)
        
        feats = occam_features_torch(representation, S_t, tau_t, V[:, t], K, vol_hat)
        # Model forward
        mu = model(feats) # (B, 1) or (B,)
        mu = mu.squeeze()
        
        a_new = mu # simple linear model outputs target position directly?
        # Original code: `mu = feats * weights`; `a_new = sum(mu)`
        # `weights` was shape (D,). `feats` (B, D). `mu` (B, D). `sum` -> (B,).
        # Yes.
        
        a_new = torch.clamp(a_new, -action_clip, action_clip)
        
        da = a_new - a
        dS = S[:, t + 1] - S_t
        
        pnl = pnl + a_new * dS
        cost = cost + 0.5 * lam[:, t] * (da ** 2)
        
        # Info cost: 0.5 * sum(mu**2) 
        # mu here corresponds to `feats * weights`.
        # Note: In original code `mu = feats * weights` (elementwise).
        # `info += 0.5 * np.sum(mu ** 2, axis=1)`
        # If representation is combined, we have w_delta * delta + w_micro * micro.
        # Original code logic: `mu` is the vector of contributions?
        # `mu = feats * weights`. `a_new = sum(mu)`.
        # `info` term depends on contributions.
        # Yes, `0.5 * sum((w_i * f_i)^2)`.
        # So we need the elementwise contributions.
        
        # In our Torch model, we can implement `forward` to return contributions?
        # Or just compute it here explicitly if model is just weights.
        
        # We will extract weights from model to compute elementwise.
        w = model.weights # Assuming simple linear model
        contribs = feats * w
        info = info + 0.5 * torch.sum(contribs ** 2, dim=1)
        
        a = a_new
        
    payoff = torch.relu(S[:, -1] - K)
    losses = payoff - pnl - cost
    info_cost = torch.mean(info)
        
    return losses, info_cost

class LinearPolicy(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(input_dim))
        
    def forward(self, x):
        return torch.sum(x * self.weights, dim=1)

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
    
    # Config
    beta = training_config["beta"]
    train_eta = training_config["train_eta"]
    train_lambdas = torch.tensor(training_config["train_lambdas"], device=S.device, dtype=torch.float32)
    gamma = training_config["gamma"]
    
    lr = training_config.get("lr", 0.05)
    n_epochs = training_config.get("n_epochs", 200) # Replaces simple grid search
    warmup_epochs = training_config.get("warmup_epochs", 50)
    
    # Parameters
    # q is trainable
    q_param = nn.Parameter(torch.tensor(float(q_init), device=S.device))
    
    optimizer = optim.Adam(list(model.parameters()) + [q_param], lr=lr)
    
    history = []
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        losses, info_cost = compute_hedging_losses_torch(
            model, S, V, lam, T, K, vol_hat, representation
        )
        
        # Objectives
        # 1. Warmup: Minimize Variance/MSE of hedging error?
        #    Target is Payoff. Error = Payoff - (PnL - Cost).
        #    Wait, Losses = Payoff - PnL - Cost.
        #    So minimizing E[Losses^2] or Var[Losses] is good proxy.
        #    Or just Mean Squared Loss?
        #    Usually we want to minimize E[Loss]. But E[Loss] is just -E[PnL].
        #    Minimizing Variance is standard for hedging.
        
        if epoch < warmup_epochs:
            # Warmup: Minimize MSE of Loss (i.e. Loss^2) + Info penalty
            # or Variance.
            loss_obj = torch.mean(losses**2) + beta * info_cost
            mode = "warmup"
        else:
            # Robust + ES
            # L = R_eta(ES_q(losses)) + beta * info
            
            # Step 1: ES loss
            # q_param is optimized to minimize the objective too
            # ES = q + 1/(1-gamma) * ReLU(L - q)
            es_vals = es_loss_torch(losses, q_param, gamma)
            
            # Step 2: Robust Risk
            risk = robust_risk_torch(es_vals, train_eta, train_lambdas)
            
            loss_obj = risk + beta * info_cost
            mode = "robust"
            
        loss_obj.backward()
        optimizer.step()
        
        history.append({
            "epoch": epoch,
            "loss_obj": float(loss_obj.item()),
            "q": float(q_param.item()),
            "weights": model.weights.detach().cpu().numpy().tolist(),
            "mode": mode
        })
        
    return {
        "final_weights": model.weights.detach().cpu().numpy(),
        "final_q": q_param.item(),
        "history": history
    }

# Re-implement eval in numpy (wrapper around original or new)
# We can use the original hedge_on_paths from experiment_occam.py
# But we need to define it or import it. Since I am overwriting the file, I must redefine it.

def occam_features_numpy(representation, S_t, tau_t, V_t, K, vol_hat):
    # Copy from original
    # We can import policies.bs_delta_call which is numpy
    from policies import bs_delta_call
    
    def micro_fn(x):
        return np.tanh(np.log(x / (np.median(x) + 1e-12) + 1e-12))

    if representation == "greeks":
        delta = bs_delta_call(S_t, K=K, tau=tau_t, vol=vol_hat, r=0.0)
        return delta.reshape(-1, 1)
    if representation == "micro":
        micro = micro_fn(V_t)
        return micro.reshape(-1, 1)
    if representation == "combined":
        delta = bs_delta_call(S_t, K=K, tau=tau_t, vol=vol_hat, r=0.0)
        micro = micro_fn(V_t)
        return np.column_stack([delta, micro])
    raise ValueError

def hedge_on_paths_numpy(S, V, lam, T, K, vol_hat, representation, weights, action_clip=5.0):
    n_paths, n_steps = V.shape
    tau_grid = np.linspace(T, 0.0, n_steps + 1)

    a = np.zeros((n_paths,), dtype=np.float64)
    pnl = np.zeros((n_paths,), dtype=np.float64)
    cost = np.zeros((n_paths,), dtype=np.float64)
    info = np.zeros((n_paths,), dtype=np.float64)
    turnover = np.zeros((n_paths,), dtype=np.float64)

    for t in range(n_steps):
        S_t = S[:, t]
        tau_t = np.full((n_paths,), tau_grid[t], dtype=np.float64)

        feats = occam_features_numpy(representation, S_t, tau_t, V[:, t], K=K, vol_hat=vol_hat)
        mu = feats * weights # Elementwise broadcast
        a_new = np.sum(mu, axis=1)
        a_new = np.clip(a_new, -action_clip, action_clip)

        da = a_new - a
        dS = S[:, t + 1] - S[:, t]
        pnl += a_new * dS
        cost += 0.5 * lam[:, t] * (da ** 2)
        info += 0.5 * np.sum(mu ** 2, axis=1)
        turnover += np.abs(da)
        a = a_new

    payoff = np.maximum(S[:, -1] - K, 0.0)
    losses = payoff - pnl - cost
    info_cost = float(np.mean(info))
    turnover_rate = float(np.mean(turnover) / n_steps)
    exec_cost = float(np.mean(cost) / n_steps)
    return losses, info_cost, turnover_rate, exec_cost

# Alias for backward compatibility
hedge_on_paths = hedge_on_paths_numpy

def train_weights(
    S: np.ndarray,
    V: np.ndarray,
    lam: np.ndarray,
    T: float,
    K: float,
    vol_hat: float,
    representation: str,
    beta: float,
    train_eta: float,
    train_lambdas: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Backward-compatible wrapper for Torch-based `train_model`.
    """
    # 1. Setup Torch
    # Ensure seeds are set if not already, but usually caller does it.
    # set_seeds() 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S_t = torch.tensor(S, dtype=torch.float32, device=device)
    V_t = torch.tensor(V, dtype=torch.float32, device=device)
    lam_t = torch.tensor(lam, dtype=torch.float32, device=device)
    
    # 2. Setup Model
    input_dim = 2 if representation == "combined" else 1
    model = LinearPolicy(input_dim).to(device)
    
    # 3. Config dict
    cfg = {
        "beta": beta,
        "train_eta": train_eta,
        "train_lambdas": train_lambdas,
        "gamma": gamma,
        # Default hyperparams for the wrapper
        "lr": 0.05,
        "n_epochs": 150, 
        "warmup_epochs": 30
    }
    
    # 4. Train
    res = train_model(model, S_t, V_t, lam_t, cfg, representation, T, K, vol_hat)
    return res["final_weights"]

# --- Main ---

def main():
    cfg = Config.load()
    set_seeds(cfg["seed_train"])
    
    # Setup Runs Directory
    run_id = cfg.get_run_id()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "runs" / f"{timestamp}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Resolved Config
    with (run_dir / "config_resolved.json").open("w") as f:
        json.dump(cfg.resolve(), f, indent=2)
        
    # Save Metadata
    metadata = {
        "git_commit": get_git_revision_hash(),
        "timestamp": timestamp,
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    }
    with (run_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    # Simulation Params
    T = float(cfg["T"])
    K = float(cfg["K"])
    vol_hat = float(cfg["vol_hat"])
    n_steps = int(cfg["n_steps"])
    
    # Load Data
    # Note: simulate_heston_signflip returns numpy arrays
    S_train, _, V_train, lam_train, _ = simulate_heston_signflip(
        regime=0, n_paths=int(cfg["n_paths_train"]), n_steps=n_steps, T=T, seed=int(cfg["seed_train"])
    )
    S_eval, _, V_eval, lam_eval, _ = simulate_heston_signflip(
        regime=0, n_paths=int(cfg["n_paths_eval"]), n_steps=n_steps, T=T, seed=int(cfg["seed_eval"])
    )
    
    # Convert Train to Torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S_train_t = torch.tensor(S_train, dtype=torch.float32, device=device)
    V_train_t = torch.tensor(V_train, dtype=torch.float32, device=device)
    lam_train_t = torch.tensor(lam_train, dtype=torch.float32, device=device)
    
    betas = cfg["beta_grid"]
    representations = cfg["representations"]
    
    metrics_log = []
    
    trained_weights = {} # (rep, beta) -> weights
    
    # Training Loop
    for rep in representations:
        input_dim = 2 if rep == "combined" else 1
        
        for beta in betas:
            print(f"Training {rep} beta={beta}")
            model = LinearPolicy(input_dim).to(device)
            
            # Training Config
            train_cfg = {
                "beta": float(beta),
                "train_eta": float(cfg["train_eta"]),
                "train_lambdas": cfg["train_lambdas"],
                "gamma": float(cfg["gamma"]),
                "lr": 0.05,
                "n_epochs": 150,
                "warmup_epochs": 30
            }
            
            # Run Training
            result = train_model(
                model, S_train_t, V_train_t, lam_train_t,
                train_cfg, rep, T, K, vol_hat
            )
            
            w_star = result["final_weights"]
            trained_weights[(rep, float(beta))] = w_star
            
            # Log training metrics
            for step in result["history"]:
                metrics_log.append({
                    "run_id": run_id,
                    "rep": rep,
                    "beta": beta,
                    **step
                })
                
    # Save Metrics Log
    with (run_dir / "metrics.jsonl").open("w") as f:
        for m in metrics_log:
            f.write(json.dumps(m) + "\n")
            
    # Evaluation & Plotting (using existing robust_curve logic but with learned weights)
    # We use numpy evaluation for consistency
    
    results = []
    etas = cfg["eta_grid_compare"] # or define
    
    for rep in representations:
        for beta in betas:
            w = trained_weights[(rep, float(beta))]
            losses, info_cost, turnover, exec_cost = hedge_on_paths_numpy(
                S_eval, V_eval, lam_eval, T, K, vol_hat, rep, w
            )
            
            # Calculate Robust Curve
            # using src/risk.py robust_es_kl
            R_curve = []
            for e in etas:
                val = robust_es_kl(losses, eta=float(e), gamma=float(cfg["gamma"]))
                R_curve.append(val)
                
            results.append({
                "rep": rep,
                "beta": beta,
                "weights": w.tolist(),
                "info_cost": info_cost,
                "R_curve": R_curve
            })
            
    # Save Results
    with (run_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Run {run_id} completed. Artifacts in {run_dir}")

if __name__ == "__main__":
    main()

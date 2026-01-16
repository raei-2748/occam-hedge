
import torch
import torch.nn as nn
import torch.autograd as autograd


def compute_irm_penalty(losses, dummy_w):
    """
    Computes IRMv1 penalty: grad_norm(loss * dummy_w)^2
    """
    # dummy_w is a scalar tensor with value 1.0, requiring grad
    # We multiply losses by dummy_w
    # But wait, losses is a vector (B,). We need mean loss for the environment.
    
    mean_loss = torch.mean(losses * dummy_w)
    
    grad = autograd.grad(
        mean_loss, dummy_w, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    
    return grad.pow(2).sum()

def train_irm(
    model, optimizer, 
    S_r0, V_r0, lam_r0,
    S_r1, V_r1, lam_r1,
    irm_penalty_weight=100.0,
    steps=100
):
    """
    Train using IRM objective: Mean(Loss_e) + lambda * IRM_Penalty
    """
    dummy_w = torch.tensor(1.0, requires_grad=True, device=S_r0.device)
    
    for step in range(steps):
        model.train()
        optimizer.zero_grad()
        
        # 1. Forward Env 0
        # We need to compute losses. We'll reuse logic from experiment_occam but stripped down.
        # Actually, let's just abstract the 'loss_function(model, S, V, lam)'
        
        # ... Implementation detail ...
        pass

# Helper to avoid circular imports / massive copy-paste
# We need `compute_hedging_losses_torch` but it returns many things.
# Let's import it.
from experiment_occam import compute_hedging_losses_torch

def train_irm_step(
    model, optimizer,
    S0, V0, lam0,
    S1, V1, lam1,
    T, K, vol_hat, representation,
    irm_penalty_weight
):
    """
    Single optimization step for IRM.
    """
    optimizer.zero_grad()
    
    # Env 0
    losses0, _, _ = compute_hedging_losses_torch(
        model, S0, V0, lam0, T, K, vol_hat, representation
    )
    # We need to make this differentiable w.r.t dummy_w PER ENV?
    # IRM trick: multiply the *unreduced* loss or just the mean?
    # Mean(loss * w)
    
    # We need a dummy `w` for EACH environment? Or just one?
    # Usually one dummy w=1.0 applied to the scalar loss of each env.
    
    w = torch.tensor(1.0, device=S0.device, requires_grad=True)
    
    loss0_mean = torch.mean(losses0 * w)
    loss1_mean = torch.mean(loss_function_helper(model, S1, V1, lam1, T, K, vol_hat, representation) * w)
    
    # Total Risk (Mean of environments)
    risk = 0.5 * (loss0_mean + loss1_mean) # But wait, we don't differentiate w.r.t w for the risk.
    # We differentiate w.r.t model weights for the RISK.
    
    # Penalty
    # g0 = grad(loss0_mean, w)
    g0 = autograd.grad(loss0_mean, w, create_graph=True)[0]
    g1 = autograd.grad(loss1_mean, w, create_graph=True)[0]
    
    penalty = g0.pow(2) + g1.pow(2)
    
    obj = risk + irm_penalty_weight * penalty
    
    obj.backward()
    optimizer.step()
    
    return risk.item(), penalty.item()

def loss_function_helper(model, S, V, lam, T, K, vol_hat, representation):
    l, _, _ = compute_hedging_losses_torch(model, S, V, lam, T, K, vol_hat, representation)
    return l

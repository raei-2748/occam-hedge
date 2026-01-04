"""
Forensic Audit of Seed 2 Anomaly at β=10.0

Investigation Objective:
At β=10.0, Seed 2 produced an anomalous profit of +5.96 in R0, contradicting
the expected stability of a 'Simplified' model. This script performs a detailed
path-level analysis to determine if this profit represents:
1. A systematic success (genuine simplicity bias working)
2. A failure to hedge (model stopped trading and got lucky on a trending path)

Analysis Components:
- Load specific price (S_t) and volume (V_t) paths for Seed 2, Regime 0
- Visualize cumulative PnL evolution for β=10 policy
- Compare β=10 policy actions against standard BS delta hedge
- Diagnose if high-β model learned a simpler hedge or just outputs ~0
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from world import simulate_heston_signflip
from policies import FactorizedVariationalPolicy, policy_delta_hedge
from experiment_occam import occam_features_torch, bs_delta_call_torch
from paper_config import load_config
from utils import set_seeds


def load_seed2_paths(n_paths=5000, n_steps=100, T=1.0, seed=2):
    """
    Generate the exact paths that Seed 2 produced in Regime 0.
    """
    S, v, V, lam, meta = simulate_heston_signflip(
        regime=0,
        n_paths=n_paths,
        n_steps=n_steps,
        T=T,
        seed=seed
    )
    return S, v, V, lam, meta


def load_trained_model(beta, representation="combined", config_path=None):
    """
    Load the trained model weights for a specific beta value.
    """
    if config_path is None:
        config_path = ROOT / "configs" / "paper_run.json"
    
    cfg = load_config(str(config_path))
    run_id = f"variance_matched_combined"  # From paper_config
    
    history_dir = ROOT / "runs" / f"{run_id}_beta_{beta:.4f}_{representation}"
    weights_file = history_dir / "final_weights.pth"
    
    if not weights_file.exists():
        raise FileNotFoundError(f"No trained model found at {weights_file}")
    
    # Load state dict
    state_dict = torch.load(weights_file, map_location='cpu')
    
    # Reconstruct model
    input_dim = 2 if representation == "combined" else 1
    model = FactorizedVariationalPolicy(input_dim, latent_dim_per_feature=2)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def evaluate_policy_on_paths(model, S, V, lam, T, K, vol_hat, representation="combined"):
    """
    Evaluate policy step-by-step and return detailed diagnostics.
    
    Returns:
        actions: (n_paths, n_steps) array of policy actions
        pnl: (n_paths, n_steps) cumulative PnL
        costs: (n_paths, n_steps) cumulative execution costs
        turnover: (n_paths, n_steps) cumulative turnover
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    n_paths, n_steps_plus1 = S.shape
    n_steps = n_steps_plus1 - 1
    tau_grid = np.linspace(T, 0.0, n_steps + 1)
    
    # Storage
    actions = np.zeros((n_paths, n_steps))
    pnl = np.zeros((n_paths, n_steps))
    costs = np.zeros((n_paths, n_steps))
    turnover = np.zeros((n_paths, n_steps))
    
    a = np.zeros(n_paths)
    cum_pnl = np.zeros(n_paths)
    cum_cost = np.zeros(n_paths)
    cum_turnover = np.zeros(n_paths)
    
    with torch.no_grad():
        for t in range(n_steps):
            S_t = torch.tensor(S[:, t], dtype=torch.float32, device=device)
            V_t = torch.tensor(V[:, t], dtype=torch.float32, device=device)
            tau_t = torch.full((n_paths,), tau_grid[t], dtype=torch.float32, device=device)
            
            # Get features
            feats = occam_features_torch(representation, S_t, tau_t, V_t, K, vol_hat)
            
            # Get action (deterministic: use mean)
            action, mus, logvars = model(feats)
            a_new = torch.clamp(action, -5.0, 5.0).cpu().numpy()
            
            # Compute step metrics
            da = a_new - a
            dS = S[:, t + 1] - S[:, t]
            
            step_pnl = a_new * dS
            step_cost = 0.5 * lam[:, t] * (da ** 2)
            step_turnover = np.abs(da)
            
            cum_pnl += step_pnl
            cum_cost += step_cost
            cum_turnover += step_turnover
            
            # Store
            actions[:, t] = a_new
            pnl[:, t] = cum_pnl.copy()
            costs[:, t] = cum_cost.copy()
            turnover[:, t] = cum_turnover.copy()
            
            a = a_new
    
    return actions, pnl, costs, turnover


def evaluate_bs_delta_on_paths(S, V, lam, T, K, vol_hat):
    """
    Evaluate standard BS delta hedge on the same paths.
    """
    n_paths, n_steps_plus1 = S.shape
    n_steps = n_steps_plus1 - 1
    tau_grid = np.linspace(T, 0.0, n_steps + 1)
    
    actions = np.zeros((n_paths, n_steps))
    pnl = np.zeros((n_paths, n_steps))
    costs = np.zeros((n_paths, n_steps))
    turnover = np.zeros((n_paths, n_steps))
    
    a = np.zeros(n_paths)
    cum_pnl = np.zeros(n_paths)
    cum_cost = np.zeros(n_paths)
    cum_turnover = np.zeros(n_paths)
    
    for t in range(n_steps):
        tau_t = np.full(n_paths, tau_grid[t])
        a_new = policy_delta_hedge(S[:, t], K=K, tau_t=tau_t, vol_hat=vol_hat)
        a_new = np.clip(a_new, -5.0, 5.0)
        
        da = a_new - a
        dS = S[:, t + 1] - S[:, t]
        
        step_pnl = a_new * dS
        step_cost = 0.5 * lam[:, t] * (da ** 2)
        step_turnover = np.abs(da)
        
        cum_pnl += step_pnl
        cum_cost += step_cost
        cum_turnover += step_turnover
        
        actions[:, t] = a_new
        pnl[:, t] = cum_pnl.copy()
        costs[:, t] = cum_cost.copy()
        turnover[:, t] = cum_turnover.copy()
        
        a = a_new
    
    return actions, pnl, costs, turnover


def plot_forensic_analysis(S, V, actions_beta10, actions_bs, pnl_beta10, pnl_bs, 
                           costs_beta10, costs_bs, T, K, output_dir):
    """
    Create comprehensive forensic visualization.
    """
    n_paths, n_steps = actions_beta10.shape
    time_grid = np.linspace(0, T, n_steps)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. Price Paths (sample)
    ax1 = fig.add_subplot(gs[0, :])
    sample_paths = min(50, n_paths)
    for i in range(sample_paths):
        ax1.plot(np.linspace(0, T, S.shape[1]), S[i, :], alpha=0.3, linewidth=0.5, color='gray')
    ax1.axhline(K, color='red', linestyle='--', label=f'Strike K={K}')
    ax1.set_xlabel('Time (years)')
    ax1.set_ylabel('Price $S_t$')
    ax1.set_title(f'Seed 2 Price Paths (Regime 0, n={n_paths})')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Policy Actions Comparison (mean ± std)
    ax2 = fig.add_subplot(gs[1, 0])
    mean_beta10 = np.mean(actions_beta10, axis=0)
    std_beta10 = np.std(actions_beta10, axis=0)
    mean_bs = np.mean(actions_bs, axis=0)
    std_bs = np.std(actions_bs, axis=0)
    
    ax2.plot(time_grid, mean_beta10, label=r'$\beta=10$ Policy', color='blue', linewidth=2)
    ax2.fill_between(time_grid, mean_beta10 - std_beta10, mean_beta10 + std_beta10, 
                     alpha=0.3, color='blue')
    ax2.plot(time_grid, mean_bs, label='BS Delta', color='orange', linewidth=2)
    ax2.fill_between(time_grid, mean_bs - std_bs, mean_bs + std_bs, 
                     alpha=0.3, color='orange')
    ax2.set_xlabel('Time (years)')
    ax2.set_ylabel('Position $a_t$')
    ax2.set_title('Policy Actions: β=10 vs BS Delta')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Action Distribution at t=50
    ax3 = fig.add_subplot(gs[1, 1])
    mid_step = n_steps // 2
    ax3.hist(actions_beta10[:, mid_step], bins=50, alpha=0.6, label=r'$\beta=10$', color='blue')
    ax3.hist(actions_bs[:, mid_step], bins=50, alpha=0.6, label='BS Delta', color='orange')
    ax3.set_xlabel('Position $a_t$')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Action Distribution at t={time_grid[mid_step]:.2f}')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Action Variance Over Time
    ax4 = fig.add_subplot(gs[1, 2])
    var_beta10 = np.var(actions_beta10, axis=0)
    var_bs = np.var(actions_bs, axis=0)
    ax4.plot(time_grid, var_beta10, label=r'$\beta=10$', color='blue', linewidth=2)
    ax4.plot(time_grid, var_bs, label='BS Delta', color='orange', linewidth=2)
    ax4.set_xlabel('Time (years)')
    ax4.set_ylabel('Var($a_t$)')
    ax4.set_title('Policy Variance (Cross-Path)')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # 5. Cumulative PnL Distribution
    ax5 = fig.add_subplot(gs[2, 0])
    final_pnl_beta10 = pnl_beta10[:, -1]
    final_pnl_bs = pnl_bs[:, -1]
    ax5.hist(final_pnl_beta10, bins=50, alpha=0.6, label=r'$\beta=10$', color='blue')
    ax5.hist(final_pnl_bs, bins=50, alpha=0.6, label='BS Delta', color='orange')
    ax5.axvline(np.mean(final_pnl_beta10), color='blue', linestyle='--', linewidth=2)
    ax5.axvline(np.mean(final_pnl_bs), color='orange', linestyle='--', linewidth=2)
    ax5.set_xlabel('Final PnL')
    ax5.set_ylabel('Frequency')
    ax5.set_title(f'Final PnL Distribution\nβ=10: {np.mean(final_pnl_beta10):.2f} | BS: {np.mean(final_pnl_bs):.2f}')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # 6. Cumulative PnL Evolution (percentiles)
    ax6 = fig.add_subplot(gs[2, 1])
    p50_beta10 = np.median(pnl_beta10, axis=0)
    p25_beta10 = np.percentile(pnl_beta10, 25, axis=0)
    p75_beta10 = np.percentile(pnl_beta10, 75, axis=0)
    
    p50_bs = np.median(pnl_bs, axis=0)
    p25_bs = np.percentile(pnl_bs, 25, axis=0)
    p75_bs = np.percentile(pnl_bs, 75, axis=0)
    
    ax6.plot(time_grid, p50_beta10, label=r'$\beta=10$ (median)', color='blue', linewidth=2)
    ax6.fill_between(time_grid, p25_beta10, p75_beta10, alpha=0.3, color='blue')
    ax6.plot(time_grid, p50_bs, label='BS Delta (median)', color='orange', linewidth=2)
    ax6.fill_between(time_grid, p25_bs, p75_bs, alpha=0.3, color='orange')
    ax6.axhline(0, color='black', linestyle='--', linewidth=1)
    ax6.set_xlabel('Time (years)')
    ax6.set_ylabel('Cumulative PnL')
    ax6.set_title('PnL Evolution (25th-75th percentile)')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    # 7. Execution Costs
    ax7 = fig.add_subplot(gs[2, 2])
    final_cost_beta10 = costs_beta10[:, -1]
    final_cost_bs = costs_bs[:, -1]
    ax7.hist(final_cost_beta10, bins=50, alpha=0.6, label=r'$\beta=10$', color='blue')
    ax7.hist(final_cost_bs, bins=50, alpha=0.6, label='BS Delta', color='orange')
    ax7.axvline(np.mean(final_cost_beta10), color='blue', linestyle='--', linewidth=2)
    ax7.axvline(np.mean(final_cost_bs), color='orange', linestyle='--', linewidth=2)
    ax7.set_xlabel('Total Execution Cost')
    ax7.set_ylabel('Frequency')
    ax7.set_title(f'Execution Costs\nβ=10: {np.mean(final_cost_beta10):.2f} | BS: {np.mean(final_cost_bs):.2f}')
    ax7.legend()
    ax7.grid(alpha=0.3)
    
    # 8. "Policy Abandonment" Diagnostic
    ax8 = fig.add_subplot(gs[3, 0])
    # Measure: fraction of paths where |a_t| < 0.1 (near-zero position)
    abandonment_beta10 = np.mean(np.abs(actions_beta10) < 0.1, axis=0)
    abandonment_bs = np.mean(np.abs(actions_bs) < 0.1, axis=0)
    ax8.plot(time_grid, abandonment_beta10, label=r'$\beta=10$', color='blue', linewidth=2)
    ax8.plot(time_grid, abandonment_bs, label='BS Delta', color='orange', linewidth=2)
    ax8.set_xlabel('Time (years)')
    ax8.set_ylabel('Fraction |$a_t$| < 0.1')
    ax8.set_title('Policy Abandonment Score')
    ax8.legend()
    ax8.grid(alpha=0.3)
    
    # 9. Delta-Neutrality Check
    ax9 = fig.add_subplot(gs[3, 1])
    # Compute BS delta at each step
    n_steps_full = S.shape[1]
    tau_grid_full = np.linspace(T, 0.0, n_steps_full)
    bs_deltas = np.zeros((n_paths, n_steps))
    for t in range(n_steps):
        tau_t = np.full(n_paths, tau_grid_full[t])
        bs_deltas[:, t] = policy_delta_hedge(S[:, t], K=K, tau_t=tau_t, vol_hat=0.2)
    
    # Deviation from delta
    delta_dev_beta10 = np.abs(actions_beta10 - bs_deltas)
    delta_dev_mean = np.mean(delta_dev_beta10, axis=0)
    delta_dev_std = np.std(delta_dev_beta10, axis=0)
    
    ax9.plot(time_grid, delta_dev_mean, color='blue', linewidth=2)
    ax9.fill_between(time_grid, delta_dev_mean - delta_dev_std, 
                     delta_dev_mean + delta_dev_std, alpha=0.3, color='blue')
    ax9.set_xlabel('Time (years)')
    ax9.set_ylabel('|$a_t - \\Delta_{BS}$|')
    ax9.set_title(r'Deviation from BS Delta ($\beta=10$)')
    ax9.grid(alpha=0.3)
    
    # 10. Wrong-Way Trading Score
    ax10 = fig.add_subplot(gs[3, 2])
    # Compute correlation between position changes and subsequent price moves
    wrong_way_scores = []
    for t in range(n_steps - 1):
        da = actions_beta10[:, t+1] - actions_beta10[:, t]
        dS = S[:, t+2] - S[:, t+1]
        # Wrong-way: increasing position when price about to fall (negative corr)
        corr = np.corrcoef(da, dS)[0, 1]
        wrong_way_scores.append(-corr)  # Negative corr = wrong-way
    
    ax10.plot(time_grid[:-1], wrong_way_scores, color='red', linewidth=2)
    ax10.axhline(0, color='black', linestyle='--', linewidth=1)
    ax10.set_xlabel('Time (years)')
    ax10.set_ylabel('Wrong-Way Score')
    ax10.set_title(r'Wrong-Way Trading ($\beta=10$)')
    ax10.grid(alpha=0.3)
    
    plt.suptitle('Forensic Audit: Seed 2 Anomaly at β=10.0', fontsize=16, fontweight='bold')
    
    # Save
    output_path = output_dir / "seed2_forensic_audit.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved forensic plot: {output_path}")
    plt.close()


def generate_summary_report(S, V, lam, actions_beta10, actions_bs, pnl_beta10, pnl_bs,
                            costs_beta10, costs_bs, turnover_beta10, turnover_bs, 
                            K, output_dir):
    """
    Generate a text summary report of the forensic findings.
    """
    n_paths = S.shape[0]
    
    # Compute final metrics
    final_pnl_beta10 = pnl_beta10[:, -1]
    final_pnl_bs = pnl_bs[:, -1]
    final_cost_beta10 = costs_beta10[:, -1]
    final_cost_bs = costs_bs[:, -1]
    final_turnover_beta10 = turnover_beta10[:, -1]
    final_turnover_bs = turnover_bs[:, -1]
    
    # Payoff
    payoff = np.maximum(S[:, -1] - K, 0)
    
    # Hedging error (loss)
    loss_beta10 = payoff - final_pnl_beta10 - final_cost_beta10
    loss_bs = payoff - final_pnl_bs - final_cost_bs
    
    # Policy statistics
    mean_action_beta10 = np.mean(np.abs(actions_beta10))
    mean_action_bs = np.mean(np.abs(actions_bs))
    std_action_beta10 = np.std(actions_beta10)
    std_action_bs = np.std(actions_bs)
    
    # Abandonment rate (fraction of time |a| < 0.1)
    abandonment_rate_beta10 = np.mean(np.abs(actions_beta10) < 0.1)
    abandonment_rate_bs = np.mean(np.abs(actions_bs) < 0.1)
    
    # Price trend analysis
    final_price = S[:, -1]
    initial_price = S[:, 0]
    price_trend = np.mean(final_price - initial_price)
    trending_paths = np.mean(np.abs(final_price - initial_price) > 5.0)
    
    report = f"""
================================================================================
                    FORENSIC AUDIT REPORT: SEED 2 ANOMALY
================================================================================

INVESTIGATION CONTEXT:
----------------------
Seed 2 at β=10.0 produced an anomalous profit of +5.96 in R0 (paper_frontier.csv),
contradicting the expected behavior of a simplified, high-β model.

HYPOTHESIS:
-----------
H1: Model learned a genuinely simpler, effective hedge (Occam's Razor success)
H2: Model stopped trading and got lucky on trending paths (failure mode)

DATA SUMMARY:
-------------
Number of Paths: {n_paths}
Strike Price: {K}
Mean Initial Price: {np.mean(S[:, 0]):.2f}
Mean Final Price: {np.mean(S[:, -1]):.2f}
Price Trend (mean Δ): {price_trend:.2f}
Trending Paths (|ΔS| > 5): {trending_paths*100:.1f}%

POLICY BEHAVIOR ANALYSIS:
-------------------------

β=10 Policy:
  Mean |Action|: {mean_action_beta10:.4f}
  Std(Action): {std_action_beta10:.4f}
  Abandonment Rate (|a| < 0.1): {abandonment_rate_beta10*100:.2f}%
  Mean Turnover: {np.mean(final_turnover_beta10):.4f}
  Mean Execution Cost: {np.mean(final_cost_beta10):.4f}

BS Delta Hedge:
  Mean |Action|: {mean_action_bs:.4f}
  Std(Action): {std_action_bs:.4f}
  Abandonment Rate (|a| < 0.1): {abandonment_rate_bs*100:.2f}%
  Mean Turnover: {np.mean(final_turnover_bs):.4f}
  Mean Execution Cost: {np.mean(final_cost_bs):.4f}

HEDGING PERFORMANCE:
--------------------

β=10 Policy:
  Mean PnL: {np.mean(final_pnl_beta10):.4f}
  Mean Loss (Hedging Error): {np.mean(loss_beta10):.4f}
  Std(Loss): {np.std(loss_beta10):.4f}
  95th Percentile Loss: {np.percentile(loss_beta10, 95):.4f}

BS Delta Hedge:
  Mean PnL: {np.mean(final_pnl_bs):.4f}
  Mean Loss (Hedging Error): {np.mean(loss_bs):.4f}
  Std(Loss): {np.std(loss_bs):.4f}
  95th Percentile Loss: {np.percentile(loss_bs, 95):.4f}

DIAGNOSTIC FINDINGS:
--------------------

1. POLICY SIMPLIFICATION:
   β=10 reduces mean |action| by: {(1 - mean_action_beta10/mean_action_bs)*100:.1f}%
   β=10 reduces turnover by: {(1 - np.mean(final_turnover_beta10)/np.mean(final_turnover_bs))*100:.1f}%
   
2. POLICY ABANDONMENT:
   β=10 abandonment rate is {abandonment_rate_beta10/abandonment_rate_bs:.2f}x higher than BS Delta
   {'⚠️  WARNING: High abandonment suggests model stopped trading' if abandonment_rate_beta10 > 0.3 else '✓ Abandonment within normal range'}

3. HEDGING EFFECTIVENESS:
   β=10 mean loss vs BS Delta: {np.mean(loss_beta10) - np.mean(loss_bs):.4f}
   {'✓ β=10 outperforms BS Delta' if np.mean(loss_beta10) < np.mean(loss_bs) else '⚠️  β=10 underperforms BS Delta'}
   
4. PROFIT SOURCE ANALYSIS:
   Mean payoff: {np.mean(payoff):.4f}
   β=10 PnL as % of payoff: {np.mean(final_pnl_beta10)/np.mean(payoff)*100:.1f}%
   BS PnL as % of payoff: {np.mean(final_pnl_bs)/np.mean(payoff)*100:.1f}%

VERDICT:
--------
"""
    
    # Determine verdict based on diagnostics
    if abandonment_rate_beta10 > 0.5:
        verdict = """
⚠️  FAILURE MODE DETECTED (H2 supported)
The β=10 model exhibits severe policy abandonment (>50% of time near zero position).
The positive profit appears to be LUCK from trending paths, not systematic hedging.
This is a sign of INFORMATION BOTTLENECK COLLAPSE, not Occam's Razor success.

RECOMMENDATION: Increase β penalty or add policy activity regularization.
"""
    elif np.mean(loss_beta10) < np.mean(loss_bs) and abandonment_rate_beta10 < 0.3:
        verdict = """
✓ SYSTEMATIC SUCCESS (H1 supported)
The β=10 model learned a genuinely simpler hedge that outperforms BS Delta.
Low abandonment rate indicates active trading, not policy collapse.
This supports the Occam's Razor hypothesis: simplicity bias leads to robustness.

RECOMMENDATION: This is the desired behavior. Document as evidence for the paper.
"""
    else:
        verdict = """
⚠️  AMBIGUOUS RESULT
The β=10 model shows mixed signals. Further investigation needed:
- Check if profit is concentrated in specific market regimes
- Analyze correlation between position and subsequent price moves
- Compare against oracle policy with regime information

RECOMMENDATION: Run additional diagnostics before drawing conclusions.
"""
    
    report += verdict
    report += "\n" + "="*80 + "\n"
    
    # Save report
    report_path = output_dir / "seed2_forensic_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\nSaved report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Forensic audit of Seed 2 anomaly")
    parser.add_argument("--beta", type=float, default=10.0, help="Beta value to investigate")
    parser.add_argument("--seed", type=int, default=2, help="Seed to investigate")
    parser.add_argument("--n_paths", type=int, default=5000, help="Number of paths to generate")
    parser.add_argument("--representation", type=str, default="combined", help="Feature representation")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    
    # Setup
    output_dir = ROOT / "diagnostics" / f"seed{args.seed}_beta{args.beta}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    if args.config is None:
        args.config = str(ROOT / "configs" / "paper_run.json")
    cfg = load_config(args.config)
    
    T = float(cfg["T"])
    K = float(cfg["K"])
    vol_hat = float(cfg["vol_hat"])
    n_steps = int(cfg["n_steps"])
    
    print(f"\n{'='*80}")
    print(f"FORENSIC AUDIT: Seed {args.seed} at β={args.beta}")
    print(f"{'='*80}\n")
    
    # Step 1: Load paths
    print("Step 1: Loading Seed 2 paths from simulate_heston_signflip...")
    S, v, V, lam, meta = load_seed2_paths(
        n_paths=args.n_paths,
        n_steps=n_steps,
        T=T,
        seed=args.seed
    )
    print(f"  ✓ Generated {S.shape[0]} paths with {S.shape[1]-1} steps")
    
    # Step 2: Load trained model
    print(f"\nStep 2: Loading trained β={args.beta} model...")
    try:
        model = load_trained_model(args.beta, representation=args.representation, config_path=args.config)
        print(f"  ✓ Model loaded successfully")
    except FileNotFoundError as e:
        print(f"  ✗ {e}")
        print("\n  NOTE: You may need to train the model first using scripts/run_beta_sweep.py")
        return
    
    # Step 3: Evaluate β=10 policy
    print(f"\nStep 3: Evaluating β={args.beta} policy on paths...")
    actions_beta10, pnl_beta10, costs_beta10, turnover_beta10 = evaluate_policy_on_paths(
        model, S, V, lam, T, K, vol_hat, representation=args.representation
    )
    print(f"  ✓ Policy evaluation complete")
    
    # Step 4: Evaluate BS delta hedge
    print("\nStep 4: Evaluating BS Delta hedge baseline...")
    actions_bs, pnl_bs, costs_bs, turnover_bs = evaluate_bs_delta_on_paths(
        S, V, lam, T, K, vol_hat
    )
    print(f"  ✓ Baseline evaluation complete")
    
    # Step 5: Generate visualizations
    print("\nStep 5: Generating forensic visualizations...")
    plot_forensic_analysis(
        S, V, actions_beta10, actions_bs, pnl_beta10, pnl_bs,
        costs_beta10, costs_bs, T, K, output_dir
    )
    print(f"  ✓ Visualizations saved")
    
    # Step 6: Generate summary report
    print("\nStep 6: Generating forensic report...")
    generate_summary_report(
        S, V, lam, actions_beta10, actions_bs, pnl_beta10, pnl_bs,
        costs_beta10, costs_bs, turnover_beta10, turnover_bs,
        K, output_dir
    )
    
    print(f"\n{'='*80}")
    print(f"FORENSIC AUDIT COMPLETE")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

"""
Simplified Forensic Audit of Seed 2 Anomaly

This script performs a path-level investigation WITHOUT requiring trained model weights.
Instead, it analyzes the raw price/volume paths to understand WHY Seed 2 produced
anomalous results.

Key Questions:
1. Are Seed 2 paths systematically different from other seeds?
2. Did the paths trend in a way that rewards "do nothing" strategies?
3. What would a simple "constant position" policy achieve on these paths?
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from world import simulate_heston_signflip
from policies import policy_delta_hedge
from paper_config import load_config


def generate_seed_paths(seed, n_paths=5000, n_steps=100, T=1.0):
    """Generate paths for a specific seed in Regime 0."""
    S, v, V, lam, meta = simulate_heston_signflip(
        regime=0,
        n_paths=n_paths,
        n_steps=n_steps,
        T=T,
        seed=seed
    )
    return S, v, V, lam, meta


def evaluate_constant_policy(S, V, lam, constant_position=0.0):
    """
    Evaluate a policy that holds a constant position.
    This tests if "doing nothing" would be profitable.
    """
    n_paths, n_steps_plus1 = S.shape
    n_steps = n_steps_plus1 - 1
    
    a = constant_position
    pnl = np.zeros(n_paths)
    cost = np.zeros(n_paths)
    
    # Initial trade to reach constant position
    da_initial = a
    cost += 0.5 * lam[:, 0] * (da_initial ** 2)
    
    # Hold position and accumulate PnL
    for t in range(n_steps):
        dS = S[:, t + 1] - S[:, t]
        pnl += a * dS
    
    return pnl, cost


def evaluate_bs_delta(S, V, lam, T, K, vol_hat):
    """Evaluate standard BS delta hedge."""
    n_paths, n_steps_plus1 = S.shape
    n_steps = n_steps_plus1 - 1
    tau_grid = np.linspace(T, 0.0, n_steps + 1)
    
    a = np.zeros(n_paths)
    pnl = np.zeros(n_paths)
    cost = np.zeros(n_paths)
    turnover = np.zeros(n_paths)
    
    for t in range(n_steps):
        tau_t = np.full(n_paths, tau_grid[t])
        a_new = policy_delta_hedge(S[:, t], K=K, tau_t=tau_t, vol_hat=vol_hat)
        a_new = np.clip(a_new, -5.0, 5.0)
        
        da = a_new - a
        dS = S[:, t + 1] - S[:, t]
        
        pnl += a_new * dS
        cost += 0.5 * lam[:, t] * (da ** 2)
        turnover += np.abs(da)
        
        a = a_new
    
    return pnl, cost, turnover


def analyze_path_characteristics(S, V, lam):
    """
    Compute statistical characteristics of the paths.
    """
    n_paths = S.shape[0]
    
    # Price statistics
    initial_price = S[:, 0]
    final_price = S[:, -1]
    price_change = final_price - initial_price
    price_trend = np.mean(price_change)
    
    # Volatility
    log_returns = np.diff(np.log(S), axis=1)
    realized_vol = np.std(log_returns, axis=1) * np.sqrt(252)  # Annualized
    
    # Volume statistics
    mean_volume = np.mean(V, axis=1)
    vol_volatility = np.std(V, axis=1)
    
    # Impact statistics
    mean_lambda = np.mean(lam, axis=1)
    
    return {
        'price_trend': price_trend,
        'price_change_std': np.std(price_change),
        'mean_realized_vol': np.mean(realized_vol),
        'mean_volume': np.mean(mean_volume),
        'mean_lambda': np.mean(mean_lambda),
        'trending_fraction': np.mean(np.abs(price_change) > 5.0)
    }


def compare_seeds(seeds, n_paths=5000, n_steps=100, T=1.0, K=100.0, vol_hat=0.2):
    """
    Compare path characteristics across different seeds.
    """
    results = []
    
    for seed in seeds:
        print(f"Analyzing Seed {seed}...")
        S, v, V, lam, meta = generate_seed_paths(seed, n_paths, n_steps, T)
        
        # Path characteristics
        chars = analyze_path_characteristics(S, V, lam)
        
        # Evaluate policies
        pnl_zero, cost_zero = evaluate_constant_policy(S, V, lam, constant_position=0.0)
        pnl_half, cost_half = evaluate_constant_policy(S, V, lam, constant_position=0.5)
        pnl_bs, cost_bs, turnover_bs = evaluate_bs_delta(S, V, lam, T, K, vol_hat)
        
        # Compute losses (hedging error)
        payoff = np.maximum(S[:, -1] - K, 0)
        loss_zero = payoff - pnl_zero - cost_zero
        loss_half = payoff - pnl_half - cost_half
        loss_bs = payoff - pnl_bs - cost_bs
        
        results.append({
            'seed': seed,
            'price_trend': chars['price_trend'],
            'price_change_std': chars['price_change_std'],
            'realized_vol': chars['mean_realized_vol'],
            'mean_volume': chars['mean_volume'],
            'mean_lambda': chars['mean_lambda'],
            'trending_fraction': chars['trending_fraction'],
            'loss_zero_mean': np.mean(loss_zero),
            'loss_zero_std': np.std(loss_zero),
            'loss_half_mean': np.mean(loss_half),
            'loss_half_std': np.std(loss_half),
            'loss_bs_mean': np.mean(loss_bs),
            'loss_bs_std': np.std(loss_bs),
            'pnl_zero_mean': np.mean(pnl_zero),
            'pnl_half_mean': np.mean(pnl_half),
            'pnl_bs_mean': np.mean(pnl_bs),
        })
    
    return pd.DataFrame(results)


def plot_seed_comparison(df, output_dir):
    """
    Visualize differences across seeds.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Price Trend
    ax = axes[0, 0]
    ax.bar(df['seed'], df['price_trend'], color='steelblue')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Seed')
    ax.set_ylabel('Mean Price Change')
    ax.set_title('Price Trend by Seed')
    ax.grid(alpha=0.3)
    
    # 2. Realized Volatility
    ax = axes[0, 1]
    ax.bar(df['seed'], df['realized_vol'], color='coral')
    ax.set_xlabel('Seed')
    ax.set_ylabel('Mean Realized Vol')
    ax.set_title('Realized Volatility by Seed')
    ax.grid(alpha=0.3)
    
    # 3. Trending Fraction
    ax = axes[0, 2]
    ax.bar(df['seed'], df['trending_fraction'] * 100, color='green')
    ax.set_xlabel('Seed')
    ax.set_ylabel('% Paths with |ΔS| > 5')
    ax.set_title('Trending Paths by Seed')
    ax.grid(alpha=0.3)
    
    # 4. Loss Comparison (Zero Position)
    ax = axes[1, 0]
    ax.bar(df['seed'], df['loss_zero_mean'], color='red', alpha=0.7, label='Zero Position')
    ax.errorbar(df['seed'], df['loss_zero_mean'], yerr=df['loss_zero_std'], 
                fmt='none', color='black', capsize=5)
    ax.set_xlabel('Seed')
    ax.set_ylabel('Mean Loss')
    ax.set_title('Loss: Zero Position Policy')
    ax.grid(alpha=0.3)
    ax.legend()
    
    # 5. Loss Comparison (BS Delta)
    ax = axes[1, 1]
    ax.bar(df['seed'], df['loss_bs_mean'], color='blue', alpha=0.7, label='BS Delta')
    ax.errorbar(df['seed'], df['loss_bs_mean'], yerr=df['loss_bs_std'], 
                fmt='none', color='black', capsize=5)
    ax.set_xlabel('Seed')
    ax.set_ylabel('Mean Loss')
    ax.set_title('Loss: BS Delta Hedge')
    ax.grid(alpha=0.3)
    ax.legend()
    
    # 6. PnL Comparison
    ax = axes[1, 2]
    width = 0.25
    x = np.arange(len(df))
    ax.bar(x - width, df['pnl_zero_mean'], width, label='Zero Pos', color='red', alpha=0.7)
    ax.bar(x, df['pnl_half_mean'], width, label='Half Pos', color='orange', alpha=0.7)
    ax.bar(x + width, df['pnl_bs_mean'], width, label='BS Delta', color='blue', alpha=0.7)
    ax.set_xlabel('Seed')
    ax.set_ylabel('Mean PnL')
    ax.set_title('PnL by Policy')
    ax.set_xticks(x)
    ax.set_xticklabels(df['seed'])
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.suptitle('Seed Comparison: Path Characteristics & Policy Performance', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / "seed_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot: {output_path}")
    plt.close()


def detailed_seed2_analysis(n_paths=5000, n_steps=100, T=1.0, K=100.0, vol_hat=0.2, output_dir=None):
    """
    Deep dive into Seed 2 specifically.
    """
    print("\n" + "="*80)
    print("DETAILED SEED 2 ANALYSIS")
    print("="*80 + "\n")
    
    S, v, V, lam, meta = generate_seed_paths(2, n_paths, n_steps, T)
    
    # Evaluate multiple constant positions
    positions_to_test = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    constant_results = []
    
    for pos in positions_to_test:
        pnl, cost = evaluate_constant_policy(S, V, lam, constant_position=pos)
        payoff = np.maximum(S[:, -1] - K, 0)
        loss = payoff - pnl - cost
        constant_results.append({
            'position': pos,
            'mean_pnl': np.mean(pnl),
            'mean_cost': np.mean(cost),
            'mean_loss': np.mean(loss),
            'std_loss': np.std(loss)
        })
    
    constant_df = pd.DataFrame(constant_results)
    
    # BS Delta for comparison
    pnl_bs, cost_bs, turnover_bs = evaluate_bs_delta(S, V, lam, T, K, vol_hat)
    payoff = np.maximum(S[:, -1] - K, 0)
    loss_bs = payoff - pnl_bs - cost_bs
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Loss vs Constant Position
    ax = axes[0, 0]
    ax.plot(constant_df['position'], constant_df['mean_loss'], 'o-', linewidth=2, markersize=8)
    ax.axhline(np.mean(loss_bs), color='red', linestyle='--', linewidth=2, label='BS Delta')
    ax.set_xlabel('Constant Position')
    ax.set_ylabel('Mean Hedging Loss')
    ax.set_title('Loss vs Constant Position (Seed 2)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. PnL vs Constant Position
    ax = axes[0, 1]
    ax.plot(constant_df['position'], constant_df['mean_pnl'], 'o-', linewidth=2, markersize=8, color='green')
    ax.axhline(np.mean(pnl_bs), color='red', linestyle='--', linewidth=2, label='BS Delta')
    ax.set_xlabel('Constant Position')
    ax.set_ylabel('Mean PnL')
    ax.set_title('PnL vs Constant Position (Seed 2)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Cost vs Constant Position
    ax = axes[1, 0]
    ax.plot(constant_df['position'], constant_df['mean_cost'], 'o-', linewidth=2, markersize=8, color='orange')
    ax.axhline(np.mean(cost_bs), color='red', linestyle='--', linewidth=2, label='BS Delta')
    ax.set_xlabel('Constant Position')
    ax.set_ylabel('Mean Execution Cost')
    ax.set_title('Cost vs Constant Position (Seed 2)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Price Path Distribution
    ax = axes[1, 1]
    final_prices = S[:, -1]
    initial_prices = S[:, 0]
    price_changes = final_prices - initial_prices
    ax.hist(price_changes, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(np.mean(price_changes), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(price_changes):.2f}')
    ax.set_xlabel('Final Price - Initial Price')
    ax.set_ylabel('Frequency')
    ax.set_title('Price Change Distribution (Seed 2)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.suptitle('Seed 2 Deep Dive: Constant Position Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_dir:
        output_path = output_dir / "seed2_constant_position_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved Seed 2 analysis: {output_path}")
    plt.close()
    
    # Print summary
    print("\nCONSTANT POSITION ANALYSIS:")
    print("-" * 80)
    print(constant_df.to_string(index=False))
    print("\nBS DELTA BASELINE:")
    print(f"  Mean Loss: {np.mean(loss_bs):.4f}")
    print(f"  Mean PnL: {np.mean(pnl_bs):.4f}")
    print(f"  Mean Cost: {np.mean(cost_bs):.4f}")
    print(f"  Mean Turnover: {np.mean(turnover_bs):.4f}")
    
    # Find optimal constant position
    optimal_idx = constant_df['mean_loss'].idxmin()
    optimal_pos = constant_df.loc[optimal_idx, 'position']
    optimal_loss = constant_df.loc[optimal_idx, 'mean_loss']
    
    print(f"\nOPTIMAL CONSTANT POSITION: {optimal_pos:.2f}")
    print(f"  Achieves loss: {optimal_loss:.4f}")
    print(f"  vs BS Delta: {np.mean(loss_bs):.4f}")
    print(f"  Improvement: {np.mean(loss_bs) - optimal_loss:.4f}")
    
    # Diagnosis
    print("\n" + "="*80)
    print("DIAGNOSIS:")
    print("="*80)
    
    if optimal_pos < 0.1:
        print("⚠️  SMOKING GUN: Optimal constant position is near ZERO!")
        print("   This suggests Seed 2 paths reward 'doing nothing'.")
        print("   The β=10 model likely collapsed to a~0 and got LUCKY.")
        print("   This is NOT evidence of Occam's Razor working.")
    elif optimal_loss < np.mean(loss_bs):
        print("⚠️  WARNING: A simple constant position outperforms BS Delta!")
        print(f"   Optimal position: {optimal_pos:.2f}")
        print("   This suggests the paths have a systematic bias.")
        print("   The β=10 model may have learned this bias (legitimate simplicity).")
    else:
        print("✓ BS Delta outperforms all constant positions.")
        print("  Seed 2 paths appear normal. Further investigation needed.")
    
    print("="*80 + "\n")
    
    return constant_df


def main():
    parser = argparse.ArgumentParser(description="Simplified forensic audit of Seed 2")
    parser.add_argument("--seeds", nargs='+', type=int, default=[0, 1, 2, 3, 4],
                       help="Seeds to compare")
    parser.add_argument("--n_paths", type=int, default=5000, help="Number of paths per seed")
    parser.add_argument("--detailed", action='store_true', help="Run detailed Seed 2 analysis")
    args = parser.parse_args()
    
    # Setup
    output_dir = ROOT / "diagnostics" / "seed_forensics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    cfg = load_config(str(ROOT / "configs" / "paper_run.json"))
    T = float(cfg["T"])
    K = float(cfg["K"])
    vol_hat = float(cfg["vol_hat"])
    n_steps = int(cfg["n_steps"])
    
    print("\n" + "="*80)
    print("FORENSIC AUDIT: Seed Comparison")
    print("="*80 + "\n")
    
    # Compare seeds
    print("Comparing seeds:", args.seeds)
    df = compare_seeds(args.seeds, n_paths=args.n_paths, n_steps=n_steps, T=T, K=K, vol_hat=vol_hat)
    
    # Save results
    csv_path = output_dir / "seed_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved comparison data: {csv_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("SEED COMPARISON SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    # Plot comparison
    plot_seed_comparison(df, output_dir)
    
    # Detailed Seed 2 analysis
    if args.detailed or 2 in args.seeds:
        detailed_seed2_analysis(n_paths=args.n_paths, n_steps=n_steps, T=T, K=K, 
                               vol_hat=vol_hat, output_dir=output_dir)
    
    print(f"\nAll results saved to: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

import csv
import statistics
import math
import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="Path to results directory containing paper_frontier.csv")
    args = parser.parse_args()

    csv_path = Path(args.results_dir) / 'paper_frontier.csv'
    
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        return

    data = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                beta = float(row['beta'])
                if beta not in data:
                    data[beta] = {'R0':[], 'R1':[], 'turnover':[], 'info_cost':[]}
                data[beta]['R0'].append(float(row['R0']))
                data[beta]['R1'].append(float(row['R_stress_eta0p1']))
                data[beta]['turnover'].append(float(row['turnover']))
                data[beta]['info_cost'].append(float(row['info_cost']))
    
        print(f"{'Beta':<10} {'R0':<20} {'R1':<20} {'Deg':<10} {'InfoCost':<20}")
        keys = sorted(data.keys())
        for beta in keys:
            d = data[beta]
            r0_mean = statistics.mean(d['R0'])
            r0_sem = statistics.stdev(d['R0'])/math.sqrt(len(d['R0'])) if len(d['R0'])>1 else 0
            r1_mean = statistics.mean(d['R1'])
            r1_sem = statistics.stdev(d['R1'])/math.sqrt(len(d['R1'])) if len(d['R1'])>1 else 0
            info_mean = statistics.mean(d['info_cost'])
            deg = r1_mean/r0_mean if r0_mean != 0 else 0
            print(f"{beta:<10.4f} {r0_mean:.2f}+-{r0_sem:.2f}       {r1_mean:.2f}+-{r1_sem:.2f}       {deg:.2f}       {info_mean:.4f}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

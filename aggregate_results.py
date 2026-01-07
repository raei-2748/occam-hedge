
import csv
import statistics
import math
import os

csv_path = 'runs/occam_frontier_20260104_212826/paper_frontier.csv'

if not os.path.exists(csv_path):
    # Try finding it in other directories if the specific run ID is uncertain 
    # but based on previous cat command it should be there.
    # The user might have deleted it? No, cat worked.
    pass

data = {}
try:
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            beta = float(row['beta'])
            if beta not in data:
                data[beta] = {'R0':[], 'R1':[], 'turnover_0':[], 'turnover_1':[], 'info_cost':[]}
            data[beta]['R0'].append(float(row['R0']))
            data[beta]['R1'].append(float(row['R1']))
            data[beta]['turnover_0'].append(float(row['turnover_0']))
            data[beta]['turnover_1'].append(float(row['turnover_1']))
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

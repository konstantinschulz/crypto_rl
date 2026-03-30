#!/usr/bin/env python
"""
Quick experiment runner to test different configurations.
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_experiment(exp_num, name, cmd_args):
    """Run a single experiment."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT {exp_num}: {name}")
    print(f"{'='*80}")
    
    cmd = ["python", "rl_trader.py"] + cmd_args + ["--mode", "backtest_3way"]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Started: {datetime.now().isoformat()}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    
    output = result.stdout + result.stderr
    
    # Save output
    output_file = f"exp{exp_num}_output.log"
    with open(output_file, 'w') as f:
        f.write(output)
    print(f"Output saved to: {output_file}")
    
    # Extract key metrics
    metrics = {}
    for line in output.split('\n'):
        if 'Win Rate' in line and ('%' in line or 'rate' in line.lower()):
            metrics['win_rate_line'] = line.strip()
        if 'Return' in line and '%' in line:
            metrics['return_line'] = line.strip()
        if 'Trades' in line:
            metrics['trades_line'] = line.strip()
        if 'PnL' in line and '$' in line:
            metrics['pnl_line'] = line.strip()
    
    print(f"\nKey Results:")
    for key, val in metrics.items():
        if val:
            print(f"  {key}: {val}")
    
    return result.returncode == 0, metrics


# Design experiments
experiments = [
    # Baseline: 14d, 3sym, 50k steps, small model
    (1, "Baseline (14d, 3sym, 50k, small model)", [
        "--days", "14",
        "--max-symbols", "3",
        "--train-steps", "50000",
    ]),
    
    # Exp 2: Larger model
    (2, "Larger model [64, 32]", [
        "--days", "14",
        "--max-symbols", "3",
        "--train-steps", "50000",
        "--model-arch", "[64, 32]",
    ]),
    
    # Exp 3: Even larger model [128, 64]
    (3, "Larger model [128, 64]", [
        "--days", "14",
        "--max-symbols", "3",
        "--train-steps", "50000",
        "--model-arch", "[128, 64]",
    ]),
    
    # Exp 4: More training steps
    (4, "More training (100k steps)", [
        "--days", "14",
        "--max-symbols", "3",
        "--train-steps", "100000",
    ]),
    
    # Exp 5: More data (21 days)
    (5, "More data (21 days)", [
        "--days", "21",
        "--max-symbols", "3",
        "--train-steps", "50000",
    ]),
    
    # Exp 6: More symbols (5)
    (6, "More symbols (5)", [
        "--days", "14",
        "--max-symbols", "5",
        "--train-steps", "50000",
    ]),
    
    # Exp 7: More positions (5)
    (7, "More positions (5 max)", [
        "--days", "14",
        "--max-symbols", "3",
        "--train-steps", "50000",
        "--max-positions", "5",
    ]),
    
    # Exp 8: Better batch settings
    (8, "Better batch (batch=16, n_steps=512)", [
        "--days", "14",
        "--max-symbols", "3",
        "--train-steps", "50000",
        "--batch-size", "16",
        "--n-steps", "512",
    ]),
    
    # Exp 9: Combination: larger model + more training
    (9, "Combo: [64,32] + 100k steps", [
        "--days", "14",
        "--max-symbols", "3",
        "--train-steps", "100000",
        "--model-arch", "[64, 32]",
    ]),
    
    # Exp 10: Combination: larger model + more data
    (10, "Combo: [64,32] + 21 days", [
        "--days", "21",
        "--max-symbols", "3",
        "--train-steps", "50000",
        "--model-arch", "[64, 32]",
    ]),
]

if __name__ == '__main__':
    results_summary = []
    
    for exp_num, name, args in experiments:
        try:
            success, metrics = run_experiment(exp_num, name, args)
            results_summary.append({
                'exp': exp_num,
                'name': name,
                'success': success,
                'metrics': metrics,
            })
        except subprocess.TimeoutExpired:
            print(f"✗ Experiment {exp_num} timed out")
            results_summary.append({
                'exp': exp_num,
                'name': name,
                'success': False,
                'error': 'timeout',
            })
        except Exception as e:
            print(f"✗ Experiment {exp_num} failed: {e}")
            results_summary.append({
                'exp': exp_num,
                'name': name,
                'success': False,
                'error': str(e),
            })
    
    # Save summary
    with open('EXPERIMENT_SUMMARY.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Successful: {sum(1 for r in results_summary if r['success'])}")
    print(f"Failed: {sum(1 for r in results_summary if not r['success'])}")
    print(f"\nSummary saved to: EXPERIMENT_SUMMARY.json")
    print(f"Individual logs: exp*_output.log")


#!/usr/bin/env python
"""
Systematic experiment runner for model optimization.
Tests different hyperparameters and logs results.
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import psutil
import os

class ExperimentRunner:
    def __init__(self, log_file: str = "EXPERIMENT_RESULTS.json"):
        self.log_file = Path(log_file)
        self.results: List[Dict[str, Any]] = []
        self.load_existing_results()
        
    def load_existing_results(self):
        """Load any existing results from previous runs."""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    self.results = json.load(f)
                print(f"✓ Loaded {len(self.results)} existing results")
            except Exception as e:
                print(f"⚠ Could not load results: {e}")
    
    def save_results(self):
        """Save all results to JSON file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Saved {len(self.results)} results to {self.log_file}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get current system resource info."""
        process = psutil.Process(os.getpid())
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'timestamp': datetime.now().isoformat(),
        }
    
    def run_experiment(
        self,
        exp_name: str,
        days: int = 14,
        max_symbols: int = 3,
        train_steps: int = 50_000,
        model_arch: str = "[32, 16]",
        max_positions: int = 3,
        initial_cash: float = 100.0,
        position_duration: int = 1440,
        device: str = "cpu",
        extra_params: str = "",
    ) -> Dict[str, Any]:
        """
        Run a single experiment with specified parameters.
        Returns metrics extracted from the run.
        """
        print(f"\n{'='*70}")
        print(f"Experiment: {exp_name}")
        print(f"{'='*70}")
        print(f"Days: {days} | Symbols: {max_symbols} | Steps: {train_steps}")
        print(f"Model: {model_arch} | Positions: {max_positions} | Device: {device}")
        
        start_time = time.time()
        start_info = self.get_system_info()
        
        # Build command
        cmd = [
            "python", "rl_trader.py",
            "--days", str(days),
            "--max-symbols", str(max_symbols),
            "--train-steps", str(train_steps),
            "--mode", "backtest_3way",
            "--device", device,
        ]
        
        if extra_params:
            cmd.extend(extra_params.split())
        
        print(f"Running: {' '.join(cmd)}")
        
        try:
            # Run the experiment
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            
            elapsed = time.time() - start_time
            end_info = self.get_system_info()
            
            # Parse output for key metrics
            output = result.stdout + result.stderr
            metrics = self._extract_metrics(output)
            
            # Compile result
            exp_result = {
                'timestamp': datetime.now().isoformat(),
                'experiment': exp_name,
                'parameters': {
                    'days': days,
                    'max_symbols': max_symbols,
                    'train_steps': train_steps,
                    'model_arch': model_arch,
                    'max_positions': max_positions,
                    'initial_cash': initial_cash,
                    'position_duration': position_duration,
                    'device': device,
                },
                'metrics': metrics,
                'elapsed_seconds': elapsed,
                'system': {
                    'start': start_info,
                    'end': end_info,
                },
                'success': result.returncode == 0,
                'output_file': f"exp_{exp_name.replace(' ', '_')}.log",
            }
            
            # Save output log
            with open(exp_result['output_file'], 'w') as f:
                f.write(output)
            
            self.results.append(exp_result)
            self.save_results()
            
            # Print summary
            print(f"\n✓ Experiment completed in {elapsed/60:.1f} minutes")
            print(f"  Test Win Rate: {metrics.get('test_win_rate', 'N/A')}")
            print(f"  Test Return: {metrics.get('test_return', 'N/A')}")
            print(f"  Memory Peak: {end_info['memory_mb']:.0f} MB")
            
            return exp_result
            
        except subprocess.TimeoutExpired:
            print(f"✗ Experiment timed out (>1 hour)")
            self.results.append({
                'timestamp': datetime.now().isoformat(),
                'experiment': exp_name,
                'success': False,
                'error': 'timeout',
            })
            self.save_results()
            return None
        except Exception as e:
            print(f"✗ Experiment failed: {e}")
            self.results.append({
                'timestamp': datetime.now().isoformat(),
                'experiment': exp_name,
                'success': False,
                'error': str(e),
            })
            self.save_results()
            return None
    
    def _extract_metrics(self, output: str) -> Dict[str, Any]:
        """Extract key metrics from the output."""
        metrics = {}
        
        # Look for key lines in output
        for line in output.split('\n'):
            if 'Win Rate' in line and 'test' in line.lower():
                try:
                    parts = line.split(':')[-1].strip().replace('%', '')
                    metrics['test_win_rate'] = float(parts.split('|')[0].strip())
                except:
                    pass
            elif 'Return' in line and 'test' in line.lower():
                try:
                    parts = line.split(':')[-1].strip().replace('%', '')
                    metrics['test_return'] = float(parts)
                except:
                    pass
            elif 'Trades' in line and 'test' in line.lower():
                try:
                    trades = int(line.split(':')[1].split('|')[0].strip())
                    metrics['test_trades'] = trades
                except:
                    pass
            elif 'PnL' in line and 'test' in line.lower():
                try:
                    pnl = float(line.split('$')[-1].split('|')[0].strip())
                    metrics['test_pnl'] = pnl
                except:
                    pass
        
        return metrics
    
    def print_summary(self):
        """Print summary of all experiments."""
        successful = [r for r in self.results if r.get('success', False)]
        
        print(f"\n{'='*70}")
        print(f"EXPERIMENT SUMMARY")
        print(f"{'='*70}")
        print(f"Total runs: {len(self.results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(self.results) - len(successful)}")
        
        if successful:
            print(f"\nBest Results:")
            # Sort by win rate
            sorted_by_wr = sorted(
                successful,
                key=lambda x: x.get('metrics', {}).get('test_win_rate', 0),
                reverse=True
            )
            for i, r in enumerate(sorted_by_wr[:3], 1):
                print(f"  {i}. {r['experiment']}")
                print(f"     Win Rate: {r['metrics'].get('test_win_rate', 'N/A')}%")
                print(f"     Return: {r['metrics'].get('test_return', 'N/A')}%")


if __name__ == '__main__':
    runner = ExperimentRunner()
    
    print("🧪 Starting Model Optimization Experiments")
    print(f"📊 Results file: {runner.log_file}")
    
    # Run experiments
    # Start with simple changes and build up
    
    # Baseline (should match current setup)
    # runner.run_experiment(
    #     "Baseline: 14d, 3sym, 50k steps",
    #     days=14, max_symbols=3, train_steps=50_000, device="cpu"
    # )
    
    # Experiment 1: Larger model
    runner.run_experiment(
        "Exp1: Larger network [64, 32]",
        days=14, max_symbols=3, train_steps=50_000,
        model_arch="[64, 32]",
        device="cpu"
    )
    
    # Experiment 2: More training
    runner.run_experiment(
        "Exp2: 100k steps (2x training)",
        days=14, max_symbols=3, train_steps=100_000,
        device="cpu"
    )
    
    # Experiment 3: More data
    runner.run_experiment(
        "Exp3: 21 days data",
        days=21, max_symbols=3, train_steps=50_000,
        device="cpu"
    )
    
    # Experiment 4: More symbols
    runner.run_experiment(
        "Exp4: 5 symbols",
        days=14, max_symbols=5, train_steps=50_000,
        device="cpu"
    )
    
    # Experiment 5: More positions
    runner.run_experiment(
        "Exp5: 5 max positions",
        days=14, max_symbols=3, train_steps=50_000,
        max_positions=5,
        device="cpu"
    )
    
    # Print final summary
    runner.print_summary()
    print(f"\n✅ Experiments complete. Results saved to {runner.log_file}")


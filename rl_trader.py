"""Train and evaluate PPO trader with memory-efficient dataset loading."""

from __future__ import annotations

import gc
import json
import math
import os
import subprocess
import time
import psutil
import sys
import threading
import uuid
from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback

from rl_trading_env import CryptoTradingEnv, TradingConfig, load_training_data


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')


def _utc_now_epoch() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def start_dashboard_server(directory: str, port: int = 8766) -> None:
    """Serve streamlit dashboard assets in a background thread."""
    try:
        import subprocess
        thread = threading.Thread(
            target=subprocess.run,
            args=([sys.executable, "-m", "streamlit", "run", "streamlit_dashboard.py", "--server.port", str(port), "--server.headless", "true"],),
            daemon=True
        )
        thread.start()
        print(f"[DASHBOARD] RL dashboard: http://localhost:{port}")
    except Exception as exc:
        print(f"[DASHBOARD] Could not start streamlit server on port {port}: {exc}")


class RLDashboardWriter:
    """Persist live RL training metrics to JSON for browser dashboard polling."""

    def __init__(self, state_path: str, total_timesteps: int, mode: str, config: TradingConfig, num_data_rows: int = 0):
        self.state_path = Path(state_path)
        self.index_path = self.state_path.parent / 'rl_dashboard_index.json'
        self.runs_dir = self.state_path.parent / 'rl_dashboard_runs'
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.run_state_path = self.runs_dir / f'{self.run_id}.json'
        self.run_state_relpath = Path(
            os.path.relpath(self.run_state_path, start=self.index_path.parent)
        ).as_posix()
        self.max_points = 500
        self._update_count = 0
        # Rewriting index on every metric update is expensive; periodic refresh is enough for UI discovery.
        self._index_write_stride = 10
        started_at = _utc_now()
        started_ts_epoch = _utc_now_epoch()
        self.state: Dict[str, Any] = {
            'ts': _utc_now(),
            'run': {
                'run_id': self.run_id,
                'mode': mode,
                'status': 'initializing',
                'started_at': started_at,
                'started_ts_epoch': started_ts_epoch,
                'finished_at': None,
                'finished_ts_epoch': None,
                'elapsed_seconds': 0,
                'state_file': self.run_state_relpath,
                'total_timesteps': int(total_timesteps),
                'current_step': 0,
                'progress_pct': 0.0,
            },
            'technical': {
                'step': 0,
                'epoch': 0,
                'n_updates': 0,
                'n_steps': 0,
                'batch_size': 0,
                'learning_rate': 0.0,
                'num_data_rows': int(num_data_rows),
                'memory': {
                    'device': 'unknown',
                    'ram_mb': None,
                    'vram_allocated_mb': None,
                    'vram_reserved_mb': None,
                    'vram_process_mb': None,
                    'vram_gpu_used_mb': None,
                },
                'loss': {
                    'train': None,
                    'dev': None,
                    'test': None,
                    'policy': None,
                    'value': None,
                    'entropy': None,
                    'approx_kl': None,
                    'clip_fraction': None,
                },
            },
            'rewards': {
                'train': None,
                'dev': None,
                'test': None,
            },
            'finance': {
                'initial_cash': float(config.initial_cash),
                'portfolio_value': float(config.initial_cash),
                'total_return_pct': 0.0,
                'realized_pnl': 0.0,
                'unrealized_pnl': 0.0,
                'num_positions': 0,
                'trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate_pct': 0.0,
                'stop_loss_exits': 0,
                'fees_paid': 0.0,
                'drawdown_pct': 0.0,
            },
            'splits': {
                'dev': None,
                'test': None,
            },
            'series': {
                'train_loss': [],
                'policy_loss': [],
                'value_loss': [],
                'train_reward': [],
                'dev_reward': [],
                'test_reward': [],
                'portfolio_value': [],
                'win_rate': [],
                'realized_pnl': [],
                'trades': [],
                'dev_portfolio_value': [],
                'test_portfolio_value': [],
                'dev_realized_pnl': [],
                'test_realized_pnl': [],
                'dev_trades': [],
                'test_trades': [],
                'ram_mb': [],
                'vram_allocated_mb': [],
                'vram_reserved_mb': [],
                'vram_process_mb': [],
                'vram_gpu_used_mb': [],
            },
        }
        self._write(write_index=True)

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _sanitize_for_json(value: Any) -> Any:
        """Recursively replace NaN/Infinity with None so browser JSON parsing never fails."""
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            return {k: RLDashboardWriter._sanitize_for_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [RLDashboardWriter._sanitize_for_json(v) for v in value]
        return value

    @classmethod
    def _json_dumps_safe(cls, payload: Dict[str, Any]) -> str:
        return json.dumps(cls._sanitize_for_json(payload), indent=2, allow_nan=False)

    def _build_final_summary(self) -> Dict[str, Any]:
        run = self.state.get('run', {})
        finance = self.state.get('finance', {})
        splits = self.state.get('splits', {})
        series = self.state.get('series', {})

        dev = splits.get('dev') if isinstance(splits.get('dev'), dict) else None
        test = splits.get('test') if isinstance(splits.get('test'), dict) else None
        split_rows = [x for x in [dev, test] if isinstance(x, dict) and int(x.get('steps', 0) or 0) > 0]

        def _latest_series_value(key: str) -> Optional[float]:
            vals = series.get(key, []) if isinstance(series.get(key), list) else []
            for row in reversed(vals):
                v = self._to_float(row.get('value')) if isinstance(row, dict) else None
                if v is not None and math.isfinite(v):
                    return v
            return None

        if split_rows:
            preferred = test or dev or split_rows[-1]
            trades = sum(int(x.get('trades', 0) or 0) for x in split_rows)
            winning = sum(int(x.get('winning_trades', 0) or 0) for x in split_rows)
            losing = sum(int(x.get('losing_trades', 0) or 0) for x in split_rows)
            if winning == 0 and losing == 0 and trades > 0:
                wr_pct = float(preferred.get('win_rate_pct', 0.0) or 0.0)
                winning = int(round((wr_pct / 100.0) * trades))
                losing = max(0, trades - winning)
            win_rate_pct = (winning / trades * 100.0) if trades > 0 else float(preferred.get('win_rate_pct', 0.0) or 0.0)
            return {
                'kind': 'split_aggregate',
                'mode': run.get('mode'),
                'portfolio_value': float(preferred.get('final_portfolio_value', finance.get('portfolio_value', 0.0)) or 0.0),
                'total_return_pct': float(preferred.get('total_return_pct', finance.get('total_return_pct', 0.0)) or 0.0),
                'realized_pnl': float(sum(float(x.get('realized_pnl', 0.0) or 0.0) for x in split_rows)),
                'unrealized_pnl': float(preferred.get('unrealized_pnl', finance.get('unrealized_pnl', 0.0)) or 0.0),
                'trades': int(trades),
                'winning_trades': int(winning),
                'losing_trades': int(losing),
                'win_rate_pct': float(win_rate_pct),
                'source': 'dev+test aggregate' if len(split_rows) > 1 else 'test/dev final',
            }

        # Fallback to latest in-run telemetry when no split metrics exist.
        return {
            'kind': 'finance_snapshot',
            'mode': run.get('mode'),
            'portfolio_value': float(finance.get('portfolio_value', _latest_series_value('portfolio_value') or 0.0) or 0.0),
            'total_return_pct': float(finance.get('total_return_pct', 0.0) or 0.0),
            'realized_pnl': float(finance.get('realized_pnl', _latest_series_value('realized_pnl') or 0.0) or 0.0),
            'unrealized_pnl': float(finance.get('unrealized_pnl', 0.0) or 0.0),
            'trades': int(finance.get('trades', _latest_series_value('trades') or 0) or 0),
            'winning_trades': int(finance.get('winning_trades', 0) or 0),
            'losing_trades': int(finance.get('losing_trades', 0) or 0),
            'win_rate_pct': float(finance.get('win_rate_pct', 0.0) or 0.0),
            'source': 'train/live snapshot',
        }

    def _append(self, key: str, step: int, value: Optional[float]) -> None:
        if value is None:
            return
        series = self.state['series'][key]
        series.append({'step': int(step), 'value': float(value)})
        if len(series) > self.max_points:
            self.state['series'][key] = series[-self.max_points:]

    def _current_learning_rate(self, model: Any) -> float:
        lr_value = getattr(model, 'learning_rate', 0.0)
        if callable(lr_value):
            # For schedule-based LR, approximate with progress=1.0 (start of training curve).
            return self._to_float(lr_value(1.0)) or 0.0
        return self._to_float(lr_value) or 0.0

    @staticmethod
    def _memory_snapshot(device_label: str) -> Dict[str, Any]:
        mb = 1024.0 * 1024.0
        snapshot: Dict[str, Any] = {
            'device': str(device_label or 'unknown'),
            'ram_mb': None,
            'vram_allocated_mb': None,
            'vram_reserved_mb': None,
            'vram_process_mb': None,
            'vram_gpu_used_mb': None,
        }
        try:
            snapshot['ram_mb'] = float(psutil.Process().memory_info().rss) / mb
        except Exception:
            pass

        try:
            import torch

            if str(device_label).startswith('cuda') and torch.cuda.is_available():
                cuda_device = torch.device(str(device_label))
                snapshot['vram_allocated_mb'] = float(torch.cuda.memory_allocated(cuda_device)) / mb
                snapshot['vram_reserved_mb'] = float(torch.cuda.memory_reserved(cuda_device)) / mb
        except Exception:
            pass

        # nvidia-smi reflects allocator + context + driver-visible process usage.
        try:
            if str(device_label).startswith('cuda'):
                pid = os.getpid()
                proc_query = subprocess.run(
                    [
                        'nvidia-smi',
                        '--query-compute-apps=pid,used_memory',
                        '--format=csv,noheader,nounits',
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                used_mb = 0.0
                for line in proc_query.stdout.splitlines():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) != 2:
                        continue
                    try:
                        row_pid = int(parts[0])
                        row_mb = float(parts[1])
                    except ValueError:
                        continue
                    if row_pid == pid:
                        used_mb += row_mb
                if used_mb > 0:
                    snapshot['vram_process_mb'] = used_mb

                gpu_query = subprocess.run(
                    [
                        'nvidia-smi',
                        '--query-gpu=memory.used',
                        '--format=csv,noheader,nounits',
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                gpu_used_values = []
                for line in gpu_query.stdout.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        gpu_used_values.append(float(line))
                    except ValueError:
                        continue
                if gpu_used_values:
                    snapshot['vram_gpu_used_mb'] = max(gpu_used_values)
        except Exception:
            pass

        return snapshot

    def _write(self, write_index: bool = False) -> None:
        self.state['ts'] = _utc_now()
        run = self.state.get('run', {})
        started_ts = int(run.get('started_ts_epoch') or 0)
        finished_ts_raw = run.get('finished_ts_epoch')
        if finished_ts_raw is not None:
            finished_ts = int(finished_ts_raw)
            run['elapsed_seconds'] = max(0, finished_ts - started_ts)
        else:
            run['elapsed_seconds'] = max(0, _utc_now_epoch() - started_ts)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_run = self.run_state_path.with_suffix(self.run_state_path.suffix + '.tmp')
        tmp_run.write_text(self._json_dumps_safe(self.state), encoding='utf-8')
        tmp_run.replace(self.run_state_path)

        # Keep a stable "latest" file for backward compatibility.
        tmp_latest = self.state_path.with_suffix(self.state_path.suffix + '.tmp')
        tmp_latest.write_text(self._json_dumps_safe(self.state), encoding='utf-8')
        tmp_latest.replace(self.state_path)

        if write_index:
            self._write_index()

    def _write_index(self) -> None:
        index: Dict[str, Any]
        if self.index_path.exists():
            try:
                index = json.loads(self.index_path.read_text(encoding='utf-8'))
            except (json.JSONDecodeError, OSError):
                index = {}
        else:
            index = {}

        runs = index.get('runs', []) if isinstance(index.get('runs'), list) else []
        runs = [x for x in runs if x.get('run_id') != self.run_id]
        runs.append({
            'run_id': self.run_id,
            'mode': self.state['run'].get('mode'),
            'status': self.state['run'].get('status'),
            'started_at': self.state['run'].get('started_at'),
            'finished_at': self.state['run'].get('finished_at'),
            'elapsed_seconds': self.state['run'].get('elapsed_seconds', 0),
            'state_file': self.run_state_relpath,
        })

        # Keep only the most recent entries and derive useful pointers for UI selection.
        runs = runs[-40:]
        latest_by_mode: Dict[str, Dict[str, Any]] = {}
        latest_model_run: Optional[Dict[str, Any]] = None
        model_modes = {'train', 'backtest', 'backtest_3way', 'eval'}
        for entry in reversed(runs):
            mode = str(entry.get('mode') or '')
            if mode and mode not in latest_by_mode:
                latest_by_mode[mode] = {
                    'run_id': entry.get('run_id'),
                    'state_file': entry.get('state_file'),
                    'status': entry.get('status'),
                    'started_at': entry.get('started_at'),
                    'finished_at': entry.get('finished_at'),
                }
            if latest_model_run is None and mode in model_modes:
                latest_model_run = {
                    'run_id': entry.get('run_id'),
                    'state_file': entry.get('state_file'),
                    'mode': mode,
                    'status': entry.get('status'),
                    'started_at': entry.get('started_at'),
                    'finished_at': entry.get('finished_at'),
                }

        index_payload = {
            'ts': self.state['ts'],
            'latest_run_id': self.run_id,
            'latest_state_file': self.run_state_relpath,
            'latest_by_mode': latest_by_mode,
            'latest_model_run': latest_model_run,
            'runs': runs,
        }
        tmp_index = self.index_path.with_suffix(self.index_path.suffix + '.tmp')
        tmp_index.write_text(self._json_dumps_safe(index_payload), encoding='utf-8')
        tmp_index.replace(self.index_path)

    def set_status(self, status: str) -> None:
        self.state['run']['status'] = status
        if status in {'completed', 'failed', 'stopped'}:
            self.state['run']['finished_at'] = _utc_now()
            self.state['run']['finished_ts_epoch'] = _utc_now_epoch()
            self.state['run']['final_summary'] = self._build_final_summary()
        self._write(write_index=True)

    def update_training(
        self,
        step: int,
        total_timesteps: int,
        model: Any,
        env: CryptoTradingEnv,
        logger_values: Dict[str, Any],
        dev_reward: Optional[float] = None,
    ) -> None:
        train_loss = self._to_float(logger_values.get('train/loss'))
        policy_loss = self._to_float(logger_values.get('train/policy_gradient_loss'))
        value_loss = self._to_float(logger_values.get('train/value_loss'))
        entropy_loss = self._to_float(logger_values.get('train/entropy_loss'))
        approx_kl = self._to_float(logger_values.get('train/approx_kl'))
        clip_fraction = self._to_float(logger_values.get('train/clip_fraction'))
        train_reward = self._to_float(logger_values.get('rollout/ep_rew_mean'))

        n_updates = int(self._to_float(logger_values.get('train/n_updates')) or 0)
        epoch = n_updates // max(1, int(getattr(model, 'n_epochs', 1)))
        finance = env.get_metrics_snapshot()
        model_device = str(getattr(model, 'device', 'unknown'))
        memory = self._memory_snapshot(model_device)

        # SB3 logs ep_rew_mean only after completed episodes; use return proxy until then.
        if train_reward is None:
            train_reward = self._to_float(finance.get('total_return'))

        self.state['run']['total_timesteps'] = int(total_timesteps)
        self.state['run']['current_step'] = int(step)
        self.state['run']['progress_pct'] = min(100.0, (float(step) / max(1, total_timesteps)) * 100.0)

        self.state['technical'].update({
            'step': int(step),
            'epoch': int(epoch),
            'n_updates': int(n_updates),
            'n_steps': int(getattr(model, 'n_steps', 0)),
            'batch_size': int(getattr(model, 'batch_size', 0)),
            'learning_rate': self._current_learning_rate(model),
        })
        self.state['technical']['memory'].update(memory)
        self.state['technical']['loss'].update({
            'train': train_loss,
            'policy': policy_loss,
            'value': value_loss,
            'entropy': entropy_loss,
            'approx_kl': approx_kl,
            'clip_fraction': clip_fraction,
        })

        self.state['rewards']['train'] = train_reward
        if dev_reward is not None:
            self.state['rewards']['dev'] = float(dev_reward)

        self.state['finance'].update({
            'portfolio_value': float(finance['portfolio_value']),
            'total_return_pct': float(finance['total_return']) * 100.0,
            'realized_pnl': float(finance['realized_pnl']),
            'unrealized_pnl': float(finance['unrealized_pnl']),
            'num_positions': int(finance['num_positions']),
            'trades': int(finance['trades']),
            'winning_trades': int(finance['winning_trades']),
            'losing_trades': int(finance['losing_trades']),
            'win_rate_pct': float(finance['win_rate']) * 100.0,
            'stop_loss_exits': int(finance['stop_loss_exits']),
            'fees_paid': float(finance['fees_paid']),
            'drawdown_pct': float(finance['drawdown_pct']) * 100.0,
        })

        self._append('train_loss', step, train_loss)
        self._append('policy_loss', step, policy_loss)
        self._append('value_loss', step, value_loss)
        self._append('train_reward', step, train_reward)
        self._append('dev_reward', step, dev_reward)
        self._append('portfolio_value', step, float(finance['portfolio_value']))
        self._append('win_rate', step, float(finance['win_rate']) * 100.0)
        self._append('realized_pnl', step, float(finance['realized_pnl']))
        self._append('trades', step, float(finance['trades']))
        self._append('ram_mb', step, self._to_float(memory.get('ram_mb')))
        self._append('vram_allocated_mb', step, self._to_float(memory.get('vram_allocated_mb')))
        self._append('vram_reserved_mb', step, self._to_float(memory.get('vram_reserved_mb')))
        self._append('vram_process_mb', step, self._to_float(memory.get('vram_process_mb')))
        self._append('vram_gpu_used_mb', step, self._to_float(memory.get('vram_gpu_used_mb')))
        self._update_count += 1
        self._write(write_index=(self._update_count % self._index_write_stride == 0))

    def update_split_metrics(self, split: str, metrics: Dict[str, float]) -> None:
        if split not in ('dev', 'test'):
            return
        avg_reward = float(metrics.get('total_reward', 0.0)) / max(1, int(metrics.get('steps', 1)))
        total_return_pct = float(metrics.get('total_return', 0.0)) * 100.0
        payload = {
            'steps': int(metrics.get('steps', 0)),
            'total_reward': float(metrics.get('total_reward', 0.0)),
            'avg_reward_per_step': avg_reward,
            'proxy_loss': -avg_reward,
            'final_portfolio_value': float(metrics.get('final_portfolio_value', 0.0)),
            'total_return_pct': total_return_pct,
            'trades': int(metrics.get('trades', 0)),
            'winning_trades': int(metrics.get('winning_trades', 0)),
            'losing_trades': int(metrics.get('losing_trades', 0)),
            'win_rate_pct': float(metrics.get('win_rate', 0.0)) * 100.0,
            'realized_pnl': float(metrics.get('realized_pnl', 0.0)),
            'unrealized_pnl': float(metrics.get('unrealized_pnl', 0.0)),
        }
        self.state['splits'][split] = payload
        self.state['rewards'][split] = payload['avg_reward_per_step']
        self.state['technical']['loss'][split] = payload['proxy_loss']
        # Keep top finance KPIs in sync with latest evaluated split.
        self.state['finance'].update({
            'portfolio_value': float(payload['final_portfolio_value']),
            'total_return_pct': float(payload['total_return_pct']),
            'trades': int(payload['trades']),
            'winning_trades': int(payload['winning_trades']),
            'losing_trades': int(payload['losing_trades']),
            'win_rate_pct': float(payload['win_rate_pct']),
            'realized_pnl': float(payload['realized_pnl']),
            'unrealized_pnl': float(payload['unrealized_pnl']),
        })
        self.state['run']['final_summary'] = self._build_final_summary()
        step = int(self.state['run']['current_step'])
        self._append(f'{split}_reward', step, payload['avg_reward_per_step'])
        # Keep split/eval points separate from training series to avoid end-of-run chart jumps.
        self._append(f'{split}_portfolio_value', step, payload['final_portfolio_value'])
        self._append(f'{split}_realized_pnl', step, payload['realized_pnl'])
        self._append(f'{split}_trades', step, float(payload['trades']))
        self._write(write_index=True)


class RLDashboardCallback(BaseCallback):
    """Push periodic PPO + environment telemetry into dashboard state file."""

    def __init__(
        self,
        writer: RLDashboardWriter,
        train_env: CryptoTradingEnv,
        total_timesteps: int,
        eval_callback: Optional[EvalCallback],
        update_freq: int = 500,
    ):
        super().__init__(verbose=0)
        self.writer = writer
        self.train_env = train_env
        self.total_timesteps = total_timesteps
        self.eval_callback = eval_callback
        self.update_freq = max(50, int(update_freq))
        self.gc_freq = max(5_000, int(update_freq) * 5)
        self._last_gc_step = 0
        self._warned_memory = False

    def _check_memory(self) -> None:
        """Monitor memory usage and warn if critical."""
        try:
            process = psutil.Process()
            mem_info = process.memory_info()
            mem_percent = process.memory_percent()
            
            # Warn if using > 70% of available memory
            if mem_percent > 70 and not self._warned_memory:
                print(f"\n[WARNING] High memory usage: {mem_percent:.1f}% ({mem_info.rss / 1e9:.2f} GB)")
                print("[WARNING] Consider reducing --max-symbols, --train-steps, or --days")
                self._warned_memory = True
        except Exception:
            pass  # psutil not available, skip

    def _on_training_start(self) -> None:
        self.writer.set_status('running')

    def _on_step(self) -> bool:
        if self.model is None:
            return True
        
        # Periodic garbage collection to free unused objects
        if self.n_calls - self._last_gc_step >= self.gc_freq:
            self._check_memory()
            gc.collect()
            self._last_gc_step = self.n_calls
        
        if self.n_calls % self.update_freq != 0:
            return True

        logger_values = dict(getattr(self.model.logger, 'name_to_value', {}))
        dev_reward = None
        if self.eval_callback is not None and self.eval_callback.last_mean_reward is not None:
            dev_reward = float(self.eval_callback.last_mean_reward)
        self.writer.update_training(
            step=int(self.model.num_timesteps),
            total_timesteps=self.total_timesteps,
            model=self.model,
            env=self.train_env,
            logger_values=logger_values,
            dev_reward=dev_reward,
        )
        return True

    def _on_training_end(self) -> None:
        gc.collect()  # Final cleanup
        self.writer.set_status('completed')


class RLTrader:
    """Wrapper for training and evaluation of RL trading agent"""
    
    def __init__(
        self,
        config: Optional[TradingConfig] = None,
        model_dir: str = 'models/',
        device: str = 'cpu',
    ):
        self.config = config or TradingConfig()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.device = device
        self.model = None
        self.train_env = None
        self.eval_env = None
        self.dashboard_writer: Optional[RLDashboardWriter] = None
    
    def create_envs(self, train_data: pd.DataFrame, eval_data: Optional[pd.DataFrame] = None):
        """Create training and evaluation environments"""
        self.train_env = CryptoTradingEnv(train_data, self.config)
        
        if eval_data is not None:
            self.eval_env = CryptoTradingEnv(eval_data, self.config)
        
        print(f"Train env: {len(train_data)} rows, {train_data['symbol'].nunique()} symbols")
        if eval_data is not None:
            print(f"Eval env: {len(eval_data)} rows, {eval_data['symbol'].nunique()} symbols")
    
    def train(
        self,
        timesteps: int = 100_000,
        eval_freq: int = 10_000,
        dashboard_state_path: Optional[str] = None,
        dashboard_mode: str = 'train',
        learning_rate: float = 3e-4,
        batch_size: int = 8,
        n_steps: int = 256,
        n_epochs: int = 3,
        policy_arch: Optional[List[int]] = None,
        progress_bar: bool = False,
    ):
        """
        Train agent with PPO
        
        Args:
            timesteps: Total training timesteps
            eval_freq: Evaluate every N steps
            learning_rate: PPO learning rate
            batch_size: PPO batch size
            n_steps: PPO n_steps (trajectory length)
            n_epochs: PPO n_epochs (gradient updates per step)
            policy_arch: Policy network architecture as list of ints
        """
        if self.train_env is None:
            raise ValueError("Call create_envs() first")

        if int(timesteps) < int(getattr(self.train_env, 'n_steps', 0)):
            print(
                "[TRAIN] Warning: --train-steps is smaller than one episode length "
                f"({timesteps} < {self.train_env.n_steps}). Reward curves may be sparse."
            )
        
        if policy_arch is None:
            policy_arch = [32, 16]
        
        # Create callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=eval_freq,
            save_path=str(self.model_dir),
            name_prefix='rl_trader'
        )
        
        callbacks: List[BaseCallback] = [checkpoint_callback]
        eval_callback: Optional[EvalCallback] = None
        
        if self.eval_env is not None:
            eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path=str(self.model_dir),
                eval_freq=eval_freq,
                n_eval_episodes=1,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
        
        # Create or load model
        if self.model is None:
            self.model = PPO(
                'MlpPolicy',
                self.train_env,
                device=self.device,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                verbose=1,
                tensorboard_log=None,
                policy_kwargs={
                    'net_arch': policy_arch,
                },
            )
            print(f"Created new PPO model (device={self.device})")
            print(f"  Hyperparameters:")
            print(f"    learning_rate={learning_rate}, batch_size={batch_size}")
            print(f"    n_steps={n_steps}, n_epochs={n_epochs}")
            print(f"    policy_arch={policy_arch}")


        if dashboard_state_path:
            num_rows = int(getattr(self.train_env, 'n_steps', 0))
            self.dashboard_writer = RLDashboardWriter(
                state_path=dashboard_state_path,
                total_timesteps=timesteps,
                mode=dashboard_mode,
                config=self.config,
                num_data_rows=num_rows,
            )
            dashboard_callback = RLDashboardCallback(
                writer=self.dashboard_writer,
                train_env=self.train_env,
                total_timesteps=timesteps,
                eval_callback=eval_callback,
                update_freq=max(200, eval_freq // 5),
            )
            callbacks.append(dashboard_callback)
        
        # Train
        print(f"\nTraining for {timesteps} timesteps...")
        train_start = time.perf_counter()
        self.model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            tb_log_name='rl_trader',
            progress_bar=bool(progress_bar)
        )
        print(f"[TIMING] Training completed in {time.perf_counter() - train_start:.1f}s")
        
        # Save final model
        self.model.save(self.model_dir / 'final_model')
        print(f"✓ Model saved to {self.model_dir}/final_model")
    
    def load_model(self, path: Optional[str] = None):
        """Load pre-trained model"""
        model_path = Path(path) if path is not None else (self.model_dir / 'final_model')

        self.model = PPO.load(model_path)
        print(f"✓ Loaded model from {model_path}")
    
    def evaluate(
        self,
        eval_data: pd.DataFrame,
        deterministic: bool = True,
        split_name: Optional[str] = None,
        env_reuse: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Evaluate agent on data
        
        Args:
            eval_data: DataFrame with evaluation data
            deterministic: Whether to use deterministic policy
            split_name: Name of split for logging
            env_reuse: Reuse existing env instead of creating new (saves memory)
        
        Returns:
            Dict with metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call train() or load_model()")
        
        eval_data = self._prepare_eval_data(eval_data)
        
        # Reuse env if provided, else create new one
        if env_reuse is not None:
            env = env_reuse
        else:
            env = CryptoTradingEnv(eval_data, self.config)

        model_obs_dim = int(self.model.observation_space.shape[0])
        env_obs_dim = int(env.observation_space.shape[0])
        if model_obs_dim != env_obs_dim:
            raise ValueError(
                f"Observation shape mismatch: model expects {model_obs_dim}, env provides {env_obs_dim}. "
                "Use identical symbol sets for training and evaluation."
            )
        obs, _ = env.reset()
        
        total_reward = 0
        done = False
        step_count = 0
        
        print("Evaluating...")
        while not done:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if step_count % 5000 == 0:
                print(f"  Step {step_count}: Portfolio = ${info['portfolio_value']:.2f}")
        
        # Extract metrics
        final_portfolio = env._get_portfolio_value(env.last_prices)
        finance = env.get_metrics_snapshot()
        
        metrics = {
            'total_reward': total_reward,
            'final_portfolio_value': final_portfolio,
            'total_return': (final_portfolio - self.config.initial_cash) / self.config.initial_cash,
            'steps': step_count,
            'final_positions': len(env.positions),
            'max_portfolio_value': env.peak_portfolio_value,
            'trades': int(finance['trades']),
            'winning_trades': int(finance['winning_trades']),
            'losing_trades': int(finance['losing_trades']),
            'win_rate': float(finance['win_rate']),
            'realized_pnl': float(finance['realized_pnl']),
            'unrealized_pnl': float(finance['unrealized_pnl']),
            'fees_paid': float(finance['fees_paid']),
            'drawdown_pct': float(finance['drawdown_pct']),
        }

        if split_name and 'closed_trades' in finance:
            _log_trades_summary(finance['closed_trades'], split_name)
        
        print(f"\nEvaluation Results:")
        print(f"  Total Return: {metrics['total_return']*100:.2f}%")
        print(f"  Final Portfolio Value: ${metrics['final_portfolio_value']:.2f}")
        print(f"  Max Portfolio Value: ${metrics['max_portfolio_value']:.2f}")
        print(f"  Total Reward: {metrics['total_reward']:.2f}")
        print(f"  Steps: {metrics['steps']}")
        print(f"  Trades: {metrics['trades']} | Win Rate: {metrics['win_rate']*100:.1f}%")
        print(f"  Realized PnL: ${metrics['realized_pnl']:.2f} | Unrealized: ${metrics['unrealized_pnl']:.2f}")

        if split_name and self.dashboard_writer is not None:
            self.dashboard_writer.update_split_metrics(split_name, metrics)
        
        # Clean up if we created a new env
        if env_reuse is None:
            del env
            gc.collect()
        
        return metrics

    def _prepare_eval_data(self, eval_data: pd.DataFrame) -> pd.DataFrame:
        """Align eval data symbol set/order with training env to keep obs shape stable."""
        if self.train_env is None:
            return eval_data

        required_symbols = list(self.train_env.symbols)
        available_symbols = set(eval_data['symbol'].unique())
        missing = [s for s in required_symbols if s not in available_symbols]
        if missing:
            raise ValueError(
                "Evaluation data is missing symbols used in training: "
                f"{missing[:10]}"
            )

        aligned = eval_data[eval_data['symbol'].isin(required_symbols)].copy()
        symbol_rank = {symbol: idx for idx, symbol in enumerate(required_symbols)}
        aligned['_symbol_rank'] = aligned['symbol'].map(symbol_rank)
        aligned = (
            aligned.sort_values(['_symbol_rank', 'open_time'])
            .drop(columns=['_symbol_rank'])
            .reset_index(drop=True)
        )
        return aligned
    
    def backtest_period(
        self,
        data: pd.DataFrame,
        split_ratio: float = 0.8,
        dashboard_state_path: Optional[str] = None,
        timesteps: int = 50_000,
        learning_rate: float = 3e-4,
        batch_size: int = 8,
        n_steps: int = 256,
        n_epochs: int = 3,
        policy_arch: Optional[List[int]] = None,
        eval_freq: Optional[int] = None,
    ):
        """
        Backtest: train on first part, evaluate on second part
        
        Args:
            data: Full dataset
            split_ratio: Fraction to use for training (0.8 = 80% train, 20% test)
            timesteps: Training timesteps
            learning_rate: PPO learning rate
            batch_size: PPO batch size
            n_steps: PPO n_steps
            n_epochs: PPO n_epochs
            policy_arch: Policy network architecture
        """
        train_parts = []
        test_parts = []
        for symbol, group in data.groupby('symbol', sort=False):
            group = group.sort_values('open_time')
            split_idx = int(len(group) * split_ratio)
            split_idx = max(1, min(split_idx, len(group) - 1))
            train_parts.append(group.iloc[:split_idx])
            test_parts.append(group.iloc[split_idx:])

        train_data = pd.concat(train_parts, ignore_index=True)
        test_data = pd.concat(test_parts, ignore_index=True)
        
        print(f"\n{'='*60}")
        print(f"{'BACKTEST MODE':^60}")
        print(f"{'='*60}")
        print(f"Train: {len(train_data)} rows | Test: {len(test_data)} rows")
        print(f"Train symbols: {train_data['symbol'].nunique()} | Test symbols: {test_data['symbol'].nunique()}")
        print(f"{'='*60}\n")
        
        self.create_envs(train_data, test_data)
        self.train(
            timesteps=int(timesteps),
            eval_freq=int(eval_freq) if eval_freq is not None else max(1_000, int(timesteps) // 10),
            dashboard_state_path=dashboard_state_path,
            dashboard_mode='backtest',
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_steps=n_steps,
            n_epochs=n_epochs,
            policy_arch=policy_arch,
        )
        
        print(f"\n{'='*60}")
        print("EVALUATION ON TEST SET")
        print(f"{'='*60}\n")
        
        self.evaluate(test_data, deterministic=True, split_name='test')

    def backtest_3way(
        self,
        data: pd.DataFrame,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        dashboard_state_path: Optional[str] = None,
        timesteps: int = 50_000,
        learning_rate: float = 3e-4,
        batch_size: int = 8,
        n_steps: int = 256,
        n_epochs: int = 3,
        policy_arch: Optional[List[int]] = None,
        eval_freq: Optional[int] = None,
    ):
        """
        3-way backtest: train (60%) / val (20%) / test (20%)
        
        Args:
            data: Full dataset
            train_ratio: Fraction for training (0.6 = 60%)
            val_ratio: Fraction for validation (0.2 = 20%, remainder goes to test)
            dashboard_state_path: Path to write dashboard state
            timesteps: Training timesteps
            learning_rate: PPO learning rate
            batch_size: PPO batch size
            n_steps: PPO n_steps
            n_epochs: PPO n_epochs
            policy_arch: Policy network architecture
        """
        train_parts, val_parts, test_parts = [], [], []
        
        for symbol, group in data.groupby('symbol', sort=False):
            group = group.sort_values('open_time').reset_index(drop=True)
            n = len(group)
            train_idx = int(n * train_ratio)
            val_idx = train_idx + int(n * val_ratio)
            train_idx = max(1, train_idx)
            val_idx = max(train_idx + 1, val_idx)
            
            train_parts.append(group.iloc[:train_idx])
            val_parts.append(group.iloc[train_idx:val_idx])
            test_parts.append(group.iloc[val_idx:])
        
        train_data = pd.concat(train_parts, ignore_index=True)
        val_data = pd.concat(val_parts, ignore_index=True)
        test_data = pd.concat(test_parts, ignore_index=True)
        
        print(f"\n{'='*70}")
        print(f"{'3-WAY BACKTEST (Train / Val / Test)':^70}")
        print(f"{'='*70}")
        print(f"Train: {len(train_data):6d} rows ({train_data['symbol'].nunique():2d} symbols) "
              f"{train_data['open_time'].min()} → {train_data['open_time'].max()}")
        print(f"Val:   {len(val_data):6d} rows ({val_data['symbol'].nunique():2d} symbols) "
              f"{val_data['open_time'].min()} → {val_data['open_time'].max()}")
        print(f"Test:  {len(test_data):6d} rows ({test_data['symbol'].nunique():2d} symbols) "
              f"{test_data['open_time'].min()} → {test_data['open_time'].max()}")
        print(f"{'='*70}\n")
        
        # Create envs with val as eval set during training
        self.create_envs(train_data, val_data)
        
        # Train with val monitoring
        print("PHASE 1: Training on train set with val monitoring...")
        self.train(
            timesteps=timesteps,
            eval_freq=int(eval_freq) if eval_freq is not None else max(10_000, timesteps // 2),
            dashboard_state_path=dashboard_state_path,
            dashboard_mode='backtest_3way',
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_steps=n_steps,
            n_epochs=n_epochs,
            policy_arch=policy_arch,
            progress_bar=False,
        )
        
        # Clean up training data to free memory
        del train_data
        gc.collect()
        
        # Evaluate on val
        print(f"\n{'='*70}")
        print("PHASE 2: Validation on val set")
        print(f"{'='*70}\n")
        val_metrics = self.evaluate(val_data, deterministic=True, split_name='dev', env_reuse=self.eval_env)
        
        # Clean up
        del val_data
        gc.collect()
        
        # Evaluate on test
        print(f"\n{'='*70}")
        print("PHASE 3: Testing on test set")
        print(f"{'='*70}\n")
        test_metrics = self.evaluate(test_data, deterministic=True, split_name='test')
        
        # Summary
        print(f"\n{'='*70}")
        print("BACKTEST SUMMARY")
        print(f"{'='*70}")
        print(f"Val Trades:  {int(val_metrics.get('trades', 0)):3d} | "
              f"WR: {val_metrics.get('win_rate', 0)*100:5.1f}% | "
              f"Return: {val_metrics.get('total_return', 0)*100:6.2f}%")
        print(f"Test Trades: {int(test_metrics.get('trades', 0)):3d} | "
              f"WR: {test_metrics.get('win_rate', 0)*100:5.1f}% | "
              f"Return: {test_metrics.get('total_return', 0)*100:6.2f}%")
        print(f"{'='*70}\n")
        
        return val_metrics, test_metrics

def _parse_symbols(symbols_raw: Optional[str]) -> Optional[Sequence[str]]:
    if not symbols_raw:
        return None
    return [x.strip().upper() for x in symbols_raw.split(',') if x.strip()]


def _balanced_eval_slice(data: pd.DataFrame, max_rows_per_symbol: int = 300) -> pd.DataFrame:
    """Build evaluation sample with all symbols represented to avoid obs-size mismatch."""
    parts = []
    for _, group in data.groupby('symbol', sort=False):
        parts.append(group.sort_values('open_time').tail(max_rows_per_symbol))
    return pd.concat(parts, ignore_index=True)


def _log_trades_summary(closed_trades: List[Dict], split_name: str) -> None:
    """Log detailed breakdown of closed trades during backtest."""
    if not closed_trades:
        print(f"  [{split_name.upper()}] No trades closed")
        return
    
    wins = [t for t in closed_trades if t['realized_pnl'] >= 0]
    losses = [t for t in closed_trades if t['realized_pnl'] < 0]
    total_pnl = sum(t['realized_pnl'] for t in closed_trades)
    avg_pnl = total_pnl / len(closed_trades) if closed_trades else 0
    
    print(f"\n  [{split_name.upper()}] Trade Summary ({len(closed_trades)} trades):")
    print(f"    Wins: {len(wins)} | Losses: {len(losses)} | Win Rate: {len(wins)/len(closed_trades)*100:.1f}%")
    print(f"    Total PnL: ${total_pnl:.2f} | Avg PnL/Trade: ${avg_pnl:.4f}")
    print(f"    Top 5 Trades:")
    sorted_trades = sorted(closed_trades, key=lambda t: t['realized_pnl'], reverse=True)
    for i, trade in enumerate(sorted_trades[:5], 1):
        print(f"      {i}. {trade['symbol']:8s} | "
              f"Entry: ${trade['entry_price']:10.6f} → "
              f"Exit: ${trade['exit_price']:10.6f} | "
              f"Return: {trade['return_pct']*100:6.2f}% | "
              f"PnL: ${trade['realized_pnl']:8.4f} | "
              f"Reason: {trade['reason']}")


def main() -> None:
    """Main training pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='RL Cryptocurrency Trading Agent')
    parser.add_argument('--days', type=int, default=7, help='Recent days to load (default: 7)')
    parser.add_argument('--symbols', type=str, default='', help='Comma separated symbols, e.g. BTCUSDT,ETHUSDT')
    parser.add_argument('--max-symbols', type=int, default=3, help='Max symbols when --symbols not set (default: 3 for memory efficiency)')
    parser.add_argument(
        '--train-steps',
        type=int,
        default=50_000,
        help='Training timesteps for train/backtest/backtest_3way modes',
    )
    parser.add_argument('--mode', choices=['train', 'eval', 'backtest', 'backtest_3way'], default='train', help='Execution mode')
    parser.add_argument('--model', type=str, help='Path to saved model (for eval mode)')
    parser.add_argument('--no-cache', action='store_true', help='Disable cached parquet slices')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto', help='Torch device for PPO')
    parser.add_argument('--dashboard', action='store_true', help='Enable live RL dashboard JSON updates')
    parser.add_argument('--dashboard-port', type=int, default=8766, help='Port for local RL dashboard HTTP server')
    parser.add_argument('--dashboard-state', type=str, default='rl_dashboard_state.json', help='JSON file for RL dashboard state')
    
    # Hyperparameter tuning arguments
    parser.add_argument('--model-arch', type=str, default='[256, 128, 64]', help='Model architecture as string, e.g. "[64, 32]" or "[128, 64, 32]"')
    parser.add_argument('--max-positions', type=int, default=3, help='Maximum number of open positions (default: 3)')
    parser.add_argument('--initial-cash', type=float, default=100.0, help='Starting portfolio value in USD')
    parser.add_argument('--position-duration', type=int, default=1440, help='Max position hold time in minutes (default: 1440 = 1 day)')
    parser.add_argument('--max-budget-per-trade', type=float, default=20.0, help='Maximum USD allocated per trade')
    parser.add_argument('--transaction-cost', type=float, default=0.001, help='Per-trade transaction cost ratio (e.g., 0.001 = 0.1%)')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='PPO learning rate (default: 3e-4)')
    parser.add_argument('--batch-size', type=int, default=16, help='PPO batch size (default: 16)')
    parser.add_argument('--n-steps', type=int, default=512, help='PPO n_steps (default: 512)')
    parser.add_argument('--n-epochs', type=int, default=5, help='PPO n_epochs (default: 5)')
    parser.add_argument('--eval-freq', type=int, default=0, help='Validation frequency in training steps (0 = mode default)')
    parser.add_argument('--trade-action-bonus', type=float, default=15.0, help='Reward bonus for buy/sell actions')
    parser.add_argument('--inactivity-penalty', type=float, default=-5.0, help='Per-step penalty for hold/inactivity')
    parser.add_argument('--invalid-sell-mode', choices=['force_buy', 'hold', 'penalize'], default='force_buy', help='Behavior when sell is chosen without open positions')
    parser.add_argument('--invalid-sell-penalty', type=float, default=3.0, help='Penalty magnitude for invalid sell attempts')
    parser.add_argument('--trade-execution-penalty', type=float, default=0.0, help='Additional penalty for each executed buy/sell to reduce churn')

    args = parser.parse_args()

    selected_symbols = _parse_symbols(args.symbols)
    print(f"Loading {args.days} days of data...")
    data = load_training_data(
        'binance_spot_1m_last4y_single.parquet',
        num_days=args.days,
        symbols=selected_symbols,
        max_symbols=args.max_symbols,
        use_cache=not args.no_cache,
    )
    print(f"✓ Loaded {len(data)} rows from {data['symbol'].nunique()} symbols\n")
 
    # Parse model architecture string
    try:
        model_arch = eval(args.model_arch)
        if not isinstance(model_arch, list):
            raise ValueError("model_arch must be a list")
    except:
        print(f"Error parsing model architecture: {args.model_arch}")
        print("Using default [32, 16]")
        model_arch = [32, 16]
 
    config = TradingConfig(
        initial_cash=args.initial_cash,
        max_positions=args.max_positions,
        position_duration_limit=args.position_duration,
        max_budget_per_trade=args.max_budget_per_trade,
        transaction_cost=args.transaction_cost,
        keep_history=False,
        trade_action_bonus=args.trade_action_bonus,
        inactivity_penalty=args.inactivity_penalty,
        invalid_sell_mode=args.invalid_sell_mode,
        invalid_sell_penalty=args.invalid_sell_penalty,
        trade_execution_penalty=args.trade_execution_penalty,
    )
    resolved_device = args.device
    if args.device == 'auto':
        try:
            import torch

            resolved_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception:
            resolved_device = 'cpu'
    print(f"Using device: {resolved_device}")

    trader = RLTrader(config=config, device=resolved_device)
 
    if args.dashboard:
        dashboard_dir = str(Path(__file__).resolve().parent)
        state_parent = Path(args.dashboard_state).resolve().parent
        if state_parent != Path(dashboard_dir):
            print(
                "[DASHBOARD] Warning: dashboard state path is outside served directory. "
                f"Use --dashboard-state under {dashboard_dir} to view live updates."
            )
        start_dashboard_server(dashboard_dir, port=args.dashboard_port)
 
    if args.mode == 'backtest':
        trader.backtest_period(
            data,
            split_ratio=0.8,
            dashboard_state_path=args.dashboard_state if args.dashboard else None,
            timesteps=args.train_steps,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            n_epochs=args.n_epochs,
            policy_arch=model_arch,
            eval_freq=args.eval_freq if args.eval_freq > 0 else None,
        )
        return
 
    if args.mode == 'backtest_3way':
        trader.backtest_3way(
            data,
            train_ratio=0.6,
            val_ratio=0.2,
            dashboard_state_path=args.dashboard_state if args.dashboard else None,
            timesteps=args.train_steps,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            n_epochs=args.n_epochs,
            policy_arch=model_arch,
            eval_freq=args.eval_freq if args.eval_freq > 0 else None,
        )
        return
 
    if args.mode == 'train':
        trader.create_envs(data)
        trader.train(
            timesteps=args.train_steps,
            eval_freq=args.eval_freq if args.eval_freq > 0 else max(5_000, args.train_steps // 5),
            dashboard_state_path=args.dashboard_state if args.dashboard else None,
            dashboard_mode='train',
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            n_epochs=args.n_epochs,
            policy_arch=model_arch,
            progress_bar=False,
        )
        print("\n" + "=" * 60)
        print("Quick Evaluation on Training Slice")
        print("=" * 60)
        eval_slice = _balanced_eval_slice(data, max_rows_per_symbol=300)
        trader.evaluate(eval_slice, deterministic=True, split_name='dev')
        return
 
    if not args.model:
        raise ValueError('--model required for eval mode')
    if args.dashboard:
        trader.dashboard_writer = RLDashboardWriter(
            state_path=args.dashboard_state,
            total_timesteps=0,
            mode='eval',
            config=config,
            num_data_rows=len(data),
        )
        trader.dashboard_writer.set_status('running')
    trader.load_model(args.model)
    trader.evaluate(data, deterministic=True, split_name='test')
    if trader.dashboard_writer is not None:
        trader.dashboard_writer.set_status('completed')


if __name__ == '__main__':
    main()

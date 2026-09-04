import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from crypto_rl.callbacks import (
    DashboardCallback,
    EntropyDecayCallback,
    UnifiedEvalCallback,
)
from crypto_rl.config import RLConfig
from crypto_rl.data import get_walk_forward_splits, read_n_rows
from crypto_rl.env.action_processing import get_action_mask
from crypto_rl.env.data_utils import compute_static_obs_from_long_df
from crypto_rl.env.metrics import calculate_calmar_ratio
from crypto_rl.env.minimal_env import MinimalCryptoEnv
from scripts.eval_log_action_counter import eval_log_action_counter
from scripts.eval_report import eval_report

CLIP_OBS = 10.0  # Clamps normalized inputs to [-10.0, 10.0]
TEST_FRACTION: float = 0.2
per_asset_stats: dict[str, dict[str, float]] = {}
dummy_vec_env_args: dict[str, Any] = {
    "norm_reward": False,
    "norm_obs": True,
    "clip_obs": CLIP_OBS,
    "training": False,
}


def print_if_not_trial(trial: optuna.trial.Trial | None = None, msg: str = ""):
    if trial is None:
        print(msg)


def _to_datetime(ts):
    """
    Convert a raw timestamp value to a pandas Timestamp.

    The data source can provide the timestamp in three different ways:
      * pd.Timestamp (already a datetime64[ns] object)
      * int/np.integer in seconds, milliseconds, or nanoseconds
    The function normalises all of them to a pandas Timestamp in UTC.
    """
    # 1️⃣ Already a Timestamp → nothing to do
    if isinstance(ts, pd.Timestamp):
        return ts
    # 2️⃣ Raw integer → infer its unit
    if isinstance(ts, (int, np.integer)):
        # microseconds are > 1e12 (≈ 1970‑01‑01 00:00:01 ms)
        if (
            ts >= 1_000_000_000_000_000
        ):  # > 1 quadrillion → microseconds, divide by 1 million to get seconds
            return pd.to_datetime(ts / 1000000, unit="s")
        # milliseconds are > 1e9 (≈ 1970‑01‑01 00:00:01 s)
        if ts >= 1_000_000_000:  # > 1 billion → milliseconds
            return pd.to_datetime(ts, unit="ms")
        # otherwise treat as seconds
        return pd.to_datetime(ts, unit="s")

    # 3️⃣ Anything else – let pandas decide (covers strings etc.)
    return pd.to_datetime(ts)


def run_experiment(config: RLConfig, trial: optuna.trial.Trial | None = None) -> float:
    """Run training and evaluation with walk-forward CV.
    Returns the mean test Calmar across folds (or multi-seed portfolio value) for Optuna.
    """
    # Seed for reproducibility of dataset split
    np.random.seed(config.data_seed)

    print_if_not_trial(trial, "1. Loading raw data...")
    raw_df = read_n_rows(str(config.parquet_path), config.n_rows)

    if config.cv_folds > 1:
        splits = get_walk_forward_splits(raw_df, n_folds=config.cv_folds)
    else:
        n_test = round(len(raw_df) * TEST_FRACTION)
        n_train = len(raw_df) - n_test
        splits = [(raw_df.iloc[:n_train], raw_df.iloc[n_train:])]

    n_splits = len(splits)
    print_if_not_trial(
        trial, f"Dataset split into {n_splits}-fold walk-forward cross-validation."
    )

    run_id = datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S-minimal")
    base_run_dir = Path(config.base_run_dir)
    run_dir = base_run_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    state_file = run_dir / "state.json"
    index_file = Path("rl_dashboard_index.json")
    fold_results = []
    last_model = None
    last_test_prices = None
    last_test_static = None
    last_test_names = None
    last_prices_arr = None
    last_asset_names = None
    last_eval_reward_totals = {}
    last_eval_portfolio_values = []
    last_eval_realized_pnl = []
    last_eval_steps = 0
    last_model = None
    env_config = dataclasses.replace(
        config, disable_logging=trial is not None, parquet_path=None, n_rows=0
    )
    shared_env_config = dataclasses.replace(config, disable_logging=True)

    for fold_idx, (train_prices_df, test_prices_df) in enumerate(splits):
        print_if_not_trial(trial, f"\n=== Fold {fold_idx + 1}/{n_splits} ===")
        start_ts_raw = train_prices_df["open_time"].min()
        end_ts_raw = train_prices_df["open_time"].max()
        training_start_str = (
            _to_datetime(start_ts_raw)
            .tz_localize("UTC")
            .strftime("%Y-%m-%d %H:%M:%S %Z")
        )
        training_end_str = (
            _to_datetime(end_ts_raw).tz_localize("UTC").strftime("%Y-%m-%d %H:%M:%S %Z")
        )
        prices_arr, static_obs, norm_vol_arr, asset_names = compute_static_obs_from_long_df(
            train_prices_df, config.window_size
        )
        last_prices_arr = prices_arr
        last_asset_names = asset_names

        print_if_not_trial(trial, "2. Setting up environment...")

        def make_env():
            e = MinimalCryptoEnv(
                config=env_config,
                prices_arr=prices_arr,
                static_obs=static_obs,
                norm_vol_arr=norm_vol_arr,
                asset_names=asset_names,
                run_id=run_id,
            )
            # Apply ActionMasker directly to the base gym environment
            return ActionMasker(e, get_action_mask)

        env_fns = [make_env for _ in range(config.n_envs)]
        # Vectorize and Normalize AFTER masking
        train_env = VecNormalize(
            DummyVecEnv(env_fns),
            norm_reward=True,
            norm_obs=True,
            gamma=config.gamma,
            clip_obs=CLIP_OBS,
            clip_reward=5.0,
        )
        dashboard_callback = None
        checkpoint_callback = None

        if config.dashboard and fold_idx == 0:
            try:
                index = {"runs": []}
                if index_file.exists():
                    with open(index_file, "r", encoding="utf-8") as f:
                        index = json.load(f)
                run_entry = {
                    "run_id": run_id,
                    "state_file": str(state_file),
                    "mode": "minimal",
                    "status": "initializing",
                    "started_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
                }
                runs = index.get("runs", [])
                runs.insert(0, run_entry)
                index["runs"] = runs
                index["latest_model_run"] = run_entry
                with open(index_file, "w", encoding="utf-8") as f:
                    json.dump(index, f, indent=2)
            except Exception:
                pass

        if config.dashboard:
            dashboard_callback = DashboardCallback(
                state_path=state_file,
                config=config,
                run_id=run_id,
                total_timesteps=config.timesteps,
                num_data_rows=config.n_rows,
                training_start_str=training_start_str,
                training_end_str=training_end_str,
            )

        print_if_not_trial(trial, "Computing test observations...")
        shared_test_prices, shared_test_static, shared_test_norm_vol, shared_test_names = (
            compute_static_obs_from_long_df(test_prices_df, config.window_size)
        )

        checkpoint_dir = run_dir / f"checkpoints_fold_{fold_idx + 1}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        shared_env_args = {
            "prices_arr": shared_test_prices,
            "static_obs": shared_test_static,
            "norm_vol_arr": shared_test_norm_vol,
            "asset_names": shared_test_names,
            "run_id": run_id,
            "is_eval": True,
            "config": shared_env_config,
        }
        eval_callback = None
        eval_env = None
        if config.checkpoint or trial is not None:
            raw_eval_env = MinimalCryptoEnv(**shared_env_args)
            masked_eval_env = ActionMasker(raw_eval_env, get_action_mask)
            eval_env = VecNormalize(
                DummyVecEnv([lambda: Monitor(masked_eval_env)]), **dummy_vec_env_args
            )
            eval_env.obs_rms = train_env.obs_rms
            eval_callback = UnifiedEvalCallback(
                config=config,
                checkpoint_dir=checkpoint_dir,
                eval_env=eval_env,
                trial=trial,
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        verbose = 1 if trial is None else 0
        seed = (
            (config.data_seed + fold_idx * 100)
            if config.data_seed is not None
            else None
        )
        policy_kwargs = {
            "net_arch": dict(pi=[128, 128], qf=[128, 128]),
            "activation_fn": torch.nn.ReLU,
            "normalize_images": False,
        }
        sb3_args_common: dict[str, Any] = {
            "device": device,
            "verbose": verbose,
            "seed": seed,
            "n_steps": config.n_steps,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
        }
        if config.algorithm == "SAC":
            print_if_not_trial(
                trial, f"3. Training SAC model for Fold {fold_idx + 1}..."
            )
            model = SAC(
                env=train_env,
                policy="MlpPolicy",
                ent_coef="auto",
                gamma=config.gamma,
                policy_kwargs=policy_kwargs,
                **sb3_args_common,
            )
        else:
            print_if_not_trial(
                trial, f"3. Training PPO model for Fold {fold_idx + 1}..."
            )
            model = MaskablePPO(
                env=train_env,
                policy="MlpPolicy",
                ent_coef=config.ent_coef_initial,
                clip_range=config.clip_range,
                policy_kwargs=policy_kwargs,
                **sb3_args_common,
            )
        total_training_steps: int = int(config.timesteps * (1.0 - TEST_FRACTION))
        entropy_callback = EntropyDecayCallback(
            ent_coef_initial=config.ent_coef_initial, # High initial exploration
            ent_coef_final=config.ent_coef_final, # Fine-tuned deterministic policy at convergence
            total_timesteps=total_training_steps,
            verbose=1,
        )

        callbacks = [entropy_callback]
        if dashboard_callback is not None:
            callbacks.append(dashboard_callback)
        if eval_callback is not None:
            callbacks.append(eval_callback)
        if checkpoint_callback is not None:
            callbacks.append(checkpoint_callback)

        if callbacks:
            model.learn(total_timesteps=config.timesteps, callback=callbacks)
        else:
            model.learn(total_timesteps=config.timesteps)

        # Pruning cleanup
        if eval_callback is not None:
            eval_env.close()
            if eval_callback.is_pruned:
                train_env.close()
                raise optuna.exceptions.TrialPruned()

        print_if_not_trial(
            trial, f"4. Testing trained model for Fold {fold_idx + 1}..."
        )
        shared_env_args_with_logging = shared_env_args.copy()
        shared_env_args_with_logging["config"] = dataclasses.replace(
            shared_env_config, disable_logging=False
        )
        test_env_raw = ActionMasker(
            MinimalCryptoEnv(**shared_env_args_with_logging), get_action_mask
        )
        test_env = VecNormalize(
            DummyVecEnv([lambda: test_env_raw]), **dummy_vec_env_args
        )
        # Inherit the trained reality
        test_env.obs_rms = train_env.obs_rms
        # VecEnv reset returns ONLY obs (1 value)
        obs = test_env.reset()
        done = False
        eval_steps = 0
        base_test_env: MinimalCryptoEnv = test_env.venv.envs[0].unwrapped
        eval_initial_portfolio_value = base_test_env.portfolio_value
        eval_portfolio_values = [
            {"step": 0, "value": float(eval_initial_portfolio_value)}
        ]
        eval_realized_pnl = [{"step": 0, "value": 0.0}]
        eval_closed_trades = 0
        eval_winning_trades = 0
        eval_reward_totals = {}
        info: dict[str, Any] = {}
        while not done:
            action_masks = np.expand_dims(test_env.venv.envs[0].action_masks(), axis=0)
            action, _ = model.predict(
                obs, action_masks=action_masks, deterministic=True
            )
            # VecEnv step returns 4 values: obs, rewards, dones, infos
            obs, rewards, dones, infos = test_env.step(action)
            reward = float(rewards[0])
            done = bool(dones[0])
            info = infos[0]
            if "reward_components" in info:
                for k, v in info["reward_components"].items():
                    eval_reward_totals[k] = eval_reward_totals.get(k, 0) + v
            if info.get("is_valid_sell", False):
                eval_closed_trades += 1
                if info.get("realised_pnl", 0.0) > 0:
                    eval_winning_trades += 1
            eval_steps += 1
            # Get the true PV, dodging the VecEnv auto-reset on the final step
            current_pv = info.get(
                "final_portfolio_value", base_test_env.portfolio_value
            )
            eval_portfolio_values.append(
                {"step": eval_steps, "value": float(current_pv)}
            )
            eval_realized_pnl.append(
                {
                    "step": eval_steps,
                    "value": float(current_pv - eval_initial_portfolio_value),
                }
            )
        per_asset_stats = info["per_asset_stats"]
        eval_final_portfolio_value = info.get(
            "final_portfolio_value", base_test_env.portfolio_value
        )
        eval_final_trades_count = info.get(
            "final_trades_count", base_test_env.trades_count
        )
        eval_final_fees_paid = info.get(
            "final_fees_paid", base_test_env.fees_paid_total
        )

        fold_calmar = calculate_calmar_ratio(eval_portfolio_values)
        eval_win_rate_pct = (
            (eval_winning_trades / eval_closed_trades * 100.0)
            if eval_closed_trades > 0
            else 0.0
        )
        fold_results.append(
            {
                "fold": fold_idx + 1,
                "final_portfolio_value": eval_final_portfolio_value,
                "pnl": float(eval_final_portfolio_value - eval_initial_portfolio_value),
                "calmar": float(fold_calmar),
                "trades": eval_final_trades_count,
                "closed_trades": int(eval_closed_trades),
                "win_rate_pct": float(eval_win_rate_pct),
                "fees_paid": float(eval_final_fees_paid),
            }
        )

        print_if_not_trial(trial, f"Fold {fold_idx + 1} Results:")
        print_if_not_trial(
            trial,
            f"  Final PV: ${eval_final_portfolio_value:.2f} | PnL: ${eval_final_portfolio_value - eval_initial_portfolio_value:.2f} | Calmar: {fold_calmar:.2f}",
        )
        print_if_not_trial(
            trial,
            f"  Trades: {eval_final_trades_count} | Sells: {eval_closed_trades} | Win Rate: {eval_win_rate_pct:.1f}% | Fees: ${eval_final_fees_paid:.4f}",
        )

        last_model = model
        last_eval_reward_totals = eval_reward_totals
        last_eval_portfolio_values = eval_portfolio_values
        last_eval_realized_pnl = eval_realized_pnl
        last_eval_steps = eval_steps

        print_if_not_trial(trial, "\nPer-Asset Performance Breakdown:")
        print_if_not_trial(
            trial,
            f"{'Symbol':<10} | {'Realized PnL':<13} | {'Total PnL':<11} | {'Trades':<8} | {'Win Rate':<10} | {'Fees':<8}",
        )
        print_if_not_trial(trial, "-" * 72)
        for sym, stats in per_asset_stats.items():
            print_if_not_trial(
                trial,
                f"{sym:<10} | ${stats['realized_pnl']:<12.2f} | ${stats['total_pnl']:<10.2f} | "
                f"{stats['trades']:<8} | {stats['win_rate_pct']:<9.1f}% | ${stats['fees_paid']:<7.4f}",
            )

        last_test_prices = shared_test_prices
        last_test_static = shared_test_static
        last_test_norm_vol = shared_test_norm_vol
        last_test_names = shared_test_names

        test_env.close()
        train_env.close()

    # Aggregate Walk-Forward CV Results
    cv_mean_pv = float(np.mean([r["final_portfolio_value"] for r in fold_results]))
    cv_mean_pnl = float(np.mean([r["pnl"] for r in fold_results]))
    cv_mean_calmar = float(np.mean([r["calmar"] for r in fold_results]))
    cv_mean_win_rate = float(np.mean([r["win_rate_pct"] for r in fold_results]))
    cv_total_trades = sum([r["trades"] for r in fold_results])
    cv_total_fees = sum([r["fees_paid"] for r in fold_results])

    print_if_not_trial(trial, "\n" + "=" * 40)
    print_if_not_trial(trial, f"WALK-FORWARD CV SUMMARY ({n_splits} Folds):")
    print_if_not_trial(trial, f"Mean Test Portfolio Value: ${cv_mean_pv:.2f}")
    print_if_not_trial(trial, f"Mean Test PnL:             ${cv_mean_pnl:.2f}")
    print_if_not_trial(trial, f"Mean Test Calmar:          {cv_mean_calmar:.2f}")
    print_if_not_trial(trial, f"Mean Test Win Rate:        {cv_mean_win_rate:.1f}%")
    print_if_not_trial(trial, f"Total CV Trades:           {cv_total_trades}")
    print_if_not_trial(trial, f"Total Test Fees Paid:           ${cv_total_fees:.4f}")
    print_if_not_trial(trial, "=" * 40 + "\n")

    # Select BTCUSDT if present, otherwise locate the first symbol with start_price > 0
    if "BTCUSDT" in last_asset_names:
        btc_idx = last_asset_names.index("BTCUSDT")
        start_price = last_prices_arr[0, btc_idx]
        end_price = last_prices_arr[-1, btc_idx]
    else:
        start_price = last_prices_arr[0, 0]
        end_price = last_prices_arr[-1, 0]

    if start_price > 1e-8:
        buy_hold_return = (end_price - start_price) / start_price
    else:
        buy_hold_return = 0.0

    buy_hold_final = config.budget_initial * (1 + buy_hold_return)

    # Multi-seed evaluation on the final model
    # Caveat: this should actually rather be used with multiple models (1 per seed), non-deterministic predictions (i.e., sampling from the whole probability distribution), and a non-deterministic environment (e.g., random slippage, partial order filling, randomized latency).
    multi_seed_pv = []
    if not config.skip_multi_seed_eval:
        print("5. Performing multi‑seed evaluation...")
        for mseed in range(5):
            np.random.seed(mseed + 100)
            raw_ms_env = MinimalCryptoEnv(
                prices_arr=last_test_prices,
                static_obs=last_test_static,
                norm_vol_arr=last_test_norm_vol,
                asset_names=last_test_names,
                run_id=run_id,
                is_eval=True,
                config=shared_env_config,
            )
            ms_env_masked = ActionMasker(raw_ms_env, get_action_mask)
            ms_env = VecNormalize(
                DummyVecEnv([lambda: ms_env_masked]), **dummy_vec_env_args
            )
            ms_env.obs_rms = train_env.obs_rms
            ms_obs = ms_env.reset()
            ms_done = False
            final_ms_pv = None
            while not ms_done:
                # Add an explicit batch dimension for the single environment
                action_masks = np.expand_dims(
                    ms_env.venv.envs[0].action_masks(), axis=0
                )
                ms_action, _ = last_model.predict(
                    ms_obs, action_masks=action_masks, deterministic=True
                )
                ms_obs, _, ms_dones, ms_infos = ms_env.step(ms_action)  # 4-value unpack
                ms_done = ms_dones[0]
                # Capture the true PV from info on the terminal step
                if ms_done:
                    final_ms_pv = ms_infos[0].get(
                        "final_portfolio_value", raw_ms_env.portfolio_value
                    )
            multi_seed_pv.append(final_ms_pv)
            ms_env.close()

    if multi_seed_pv:
        arr = np.array(multi_seed_pv)
        print_if_not_trial(
            trial,
            f"  n={len(arr)}  mean=${arr.mean():.2f}  std=${arr.std():.2f}  "
            f"min=${arr.min():.2f}  max=${arr.max():.2f}",
        )

    # Dashboard update
    if config.dashboard and state_file and state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            state["run"]["status"] = "evaluated"
            state["run"]["finished_at"] = datetime.now(UTC).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
            state["finance"]["evaluation_results"] = {
                "final_portfolio_value": cv_mean_pv,
                "pnl": cv_mean_pnl,
                "evaluation_steps": int(last_eval_steps),
                "eval_trades": int(cv_total_trades),
                "eval_win_rate_pct": float(cv_mean_win_rate),
                "buy_hold_baseline": float(buy_hold_final),
                "total_fees_paid": float(cv_total_fees),
                "cv_folds": fold_results,
                "per_asset_breakdown": per_asset_stats,
            }
            # Add CV mean Calmar ratio to finance section for dashboard display
            state["finance"]["calmar"] = float(cv_mean_calmar)
            state["explainability"] = {
                "cumulative_rewards": {
                    k: float(v) for k, v in last_eval_reward_totals.items()
                },
                "hyperparameters": config.to_dict(),
            }
            if "series" not in state:
                state["series"] = {}

            # Downsample eval series to max 1000 points to keep state.json lightweight and responsive
            def _downsample(series_list, max_pts=1000):
                if not series_list or len(series_list) <= max_pts:
                    return series_list
                step_sz = len(series_list) / max_pts
                res = [series_list[int(i * step_sz)] for i in range(max_pts)]
                if res[-1] != series_list[-1]:
                    res[-1] = series_list[-1]
                return res

            state["series"]["test_portfolio_value"] = _downsample(
                last_eval_portfolio_values
            )
            state["series"]["test_realized_pnl"] = _downsample(last_eval_realized_pnl)
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            print_if_not_trial(
                trial,
                f"Dashboard state updated with evaluation results in {state_file}",
            )
        except Exception as e:
            print_if_not_trial(trial, f"Error updating dashboard state: {e}")

    if trial is None:
        eval_log_action_counter()
        eval_report()

    return float(cv_mean_calmar)  # Return mean Calmar ratio for Optuna optimization

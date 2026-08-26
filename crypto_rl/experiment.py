import gc
import inspect
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

from crypto_rl import MinimalCryptoEnv, read_last_n
from crypto_rl.callbacks import CheckpointCallback, DashboardCallback, TrialEvalCallback
from crypto_rl.data import get_walk_forward_splits, read_n_rows
from crypto_rl.env.action_processing import get_action_mask
from crypto_rl.env.data_utils import compute_static_obs_from_long_df
from crypto_rl.env.metrics import calculate_calmar_ratio
from scripts.eval_log_action_counter import eval_log_action_counter
from scripts.eval_report import eval_report

# Global budget variable required by other parts of the package
global BUDGET_INITIAL
CLIP_OBS = 10.0  # Clamps normalized inputs to [-10.0, 10.0]


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


def run_experiment(args, trial: optuna.trial.Trial | None = None) -> float:
    """Run training and evaluation with walk-forward CV.
    Returns the mean test Calmar across folds (or multi-seed portfolio value) for Optuna.
    """
    # Seed for reproducibility of dataset split
    np.random.seed(args.data_seed)

    global BUDGET_INITIAL
    BUDGET_INITIAL = args.budget_initial
    print_if_not_trial(trial, "1. Loading raw data...")
    raw_df = read_n_rows(args.parquet_path, args.n_rows)

    cv_folds = getattr(args, "cv_folds", 3)
    if cv_folds > 1:
        splits = get_walk_forward_splits(raw_df, n_folds=cv_folds)
    else:
        TEST_FRACTION: float = 0.2
        n_test = round(len(raw_df) * TEST_FRACTION)
        n_train = len(raw_df) - n_test
        splits = [(raw_df.iloc[:n_train], raw_df.iloc[n_train:])]

    n_splits = len(splits)
    print_if_not_trial(
        trial, f"Dataset split into {n_splits}-fold walk-forward cross-validation."
    )

    run_id = datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S-minimal")
    base_run_dir = Path(args.run_dir) if args.run_dir else Path("logs")
    run_dir = base_run_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    index_file = Path("rl_dashboard_index.json")

    env_args_allowed: list[str] = inspect.getfullargspec(MinimalCryptoEnv.__init__)[0]
    all_args = dict(args._get_kwargs())
    # Exclude data loading parameters so they don't override the precomputed split
    excluded_env_args = {
        "parquet_path",
        "n_rows",
        "prices_arr",
        "static_obs",
        "asset_names",
        "self",
    }
    env_args: dict = {
        k: v
        for k, v in all_args.items()
        if k in env_args_allowed and k not in excluded_env_args
    }

    fold_results = []
    last_model = None
    last_test_env = None
    last_prices_arr = None
    last_asset_names = None
    last_eval_reward_totals = {}
    last_eval_portfolio_values = []
    last_eval_realized_pnl = []
    last_eval_trades_count = 0
    last_eval_closed_trades = 0
    last_eval_winning_trades = 0
    last_eval_initial_portfolio_value = BUDGET_INITIAL
    last_eval_steps = 0
    last_training_start_str = None
    last_training_end_str = None

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
        last_training_start_str = training_start_str
        last_training_end_str = training_end_str

        prices_arr, static_obs, asset_names = compute_static_obs_from_long_df(
            train_prices_df, args.window_size
        )
        last_prices_arr = prices_arr
        last_asset_names = asset_names

        print_if_not_trial(trial, "2. Setting up environment...")

        def make_env():
            e = MinimalCryptoEnv(
                prices_arr=prices_arr,
                static_obs=static_obs,
                asset_names=asset_names,
                run_id=run_id,
                disable_logging=trial is not None,
                **env_args,
            )
            # Apply ActionMasker directly to the base gym environment
            return ActionMasker(e, get_action_mask)

        env_fns = [make_env for _ in range(args.n_envs)]
        # Vectorize and Normalize AFTER masking
        train_env = VecNormalize(
            DummyVecEnv(env_fns),
            norm_reward=True,
            norm_obs=False,
            gamma=args.gamma,
            clip_obs=CLIP_OBS,
            clip_reward=5.0,
        )
        callback = None
        checkpoint_callback = None
        _TARGET_CHECKPOINTS = 300
        safe_check_freq = max(500, args.timesteps // _TARGET_CHECKPOINTS)

        if args.dashboard and fold_idx == 0:
            state_file = run_dir / "state.json"
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

        if args.dashboard:
            state_file = run_dir / "state.json"
            callback = DashboardCallback(
                state_file,
                window_size=args.window_size,
                reward_type=args.reward_type,
                check_freq=safe_check_freq,
                run_id=run_id,
                total_timesteps=args.timesteps,
                num_data_rows=args.n_rows,
                training_start_str=training_start_str,
                training_end_str=training_end_str,
            )

        print_if_not_trial(trial, "Computing test observations...")
        shared_test_prices, shared_test_static, shared_test_names = (
            compute_static_obs_from_long_df(test_prices_df, args.window_size)
        )

        checkpoint_dir = run_dir / f"checkpoints_fold_{fold_idx + 1}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        shard_env_args = {
            "prices_arr": shared_test_prices,
            "static_obs": shared_test_static,
            "asset_names": shared_test_names,
            "run_id": run_id,
            "is_eval": True,
            "disable_logging": True,
            **env_args,
        }
        if args.checkpoint:
            checkpoint_test_env = MinimalCryptoEnv(**shard_env_args)
            checkpoint_callback = CheckpointCallback(
                checkpoint_dir=checkpoint_dir,
                test_env=ActionMasker(checkpoint_test_env, get_action_mask),
                check_freq=safe_check_freq,
                max_checkpoints=args.max_checkpoints,
            )

        eval_callback = None
        eval_env = None
        if trial is not None:
            raw_env = MinimalCryptoEnv(**shard_env_args)
            masked_eval_env = ActionMasker(raw_env, get_action_mask)
            eval_env = VecNormalize(
                DummyVecEnv([lambda: Monitor(masked_eval_env)]),
                norm_reward=False,
                norm_obs=True,
                clip_obs=CLIP_OBS,
                training=False,
            )
            # Sync the statistics from the training environment
            eval_env.obs_rms = train_env.obs_rms
            eval_freq = max(1000, args.timesteps // 5)
            eval_callback = TrialEvalCallback(
                eval_env=eval_env,
                trial=trial,
                n_eval_episodes=1,
                eval_freq=eval_freq,
                deterministic=True,
                verbose=0,
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        verbose = 1 if trial is None else 0
        seed = (args.data_seed + fold_idx * 100) if args.data_seed is not None else None
        ent_coef = args.ent_coef
        batch_size = args.batch_size
        gamma = args.gamma
        learning_rate = args.learning_rate
        policy_kwargs = {
            "net_arch": dict(pi=[128, 128], qf=[128, 128]),
            "activation_fn": torch.nn.ReLU,
            "normalize_images": False,
        }
        n_steps = args.n_steps
        sb3_args_common: dict[str, Any] = {
            "device": device,
            "verbose": verbose,
            "seed": seed,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        }
        if args.algorithm == "SAC":
            print_if_not_trial(
                trial, f"3. Training SAC model for Fold {fold_idx + 1}..."
            )
            model = SAC(
                env=train_env,
                policy="MlpPolicy",
                ent_coef="auto",
                gamma=gamma,
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
                ent_coef=ent_coef,
                clip_range=args.clip_range,
                policy_kwargs=policy_kwargs,
                **sb3_args_common,
            )

        callbacks = []
        if callback is not None:
            callbacks.append(callback)
        if eval_callback is not None:
            callbacks.append(eval_callback)
        if checkpoint_callback is not None:
            callbacks.append(checkpoint_callback)

        if callbacks:
            model.learn(total_timesteps=args.timesteps, callback=callbacks)
        else:
            model.learn(total_timesteps=args.timesteps)

        # Pruning cleanup
        if eval_callback is not None:
            eval_env.close()
            if eval_callback.is_pruned:
                train_env.close()
                raise optuna.exceptions.TrialPruned()

        print_if_not_trial(
            trial, f"4. Testing trained model for Fold {fold_idx + 1}..."
        )
        shard_env_args_with_logging: dict[str, Any] = shard_env_args | {
            "disable_logging": False
        }
        test_env_raw = ActionMasker(
            MinimalCryptoEnv(**shard_env_args_with_logging), get_action_mask
        )
        test_env = VecNormalize(
            DummyVecEnv([lambda: test_env_raw]),
            norm_reward=False,
            norm_obs=True,
            training=False,
            clip_obs=CLIP_OBS,
        )
        # Inherit the trained reality
        test_env.obs_rms = train_env.obs_rms
        obs, _ = test_env.reset()
        done = False
        eval_steps = 0
        base_test_env: MinimalCryptoEnv = test_env.unwrapped
        eval_initial_portfolio_value = base_test_env.portfolio_value
        eval_portfolio_values = [
            {"step": 0, "value": float(eval_initial_portfolio_value)}
        ]
        eval_realized_pnl = [{"step": 0, "value": 0.0}]
        eval_closed_trades = 0
        eval_winning_trades = 0
        eval_reward_totals = {}
        while not done:
            action_masks = np.expand_dims(test_env.action_masks(), axis=0)
            action, _ = model.predict(
                obs, action_masks=action_masks, deterministic=True
            )
            obs, reward, done, _, info = test_env.step(action)
            if "reward_components" in info:
                for k, v in info["reward_components"].items():
                    eval_reward_totals[k] = eval_reward_totals.get(k, 0) + v
            if info.get("is_valid_sell", False):
                eval_closed_trades += 1
                if info.get("realised_pnl", 0.0) > 0:
                    eval_winning_trades += 1
            eval_steps += 1
            eval_portfolio_values.append(
                {"step": eval_steps, "value": float(base_test_env.portfolio_value)}
            )
            eval_realized_pnl.append(
                {
                    "step": eval_steps,
                    "value": float(
                        base_test_env.portfolio_value - eval_initial_portfolio_value
                    ),
                }
            )

        eval_trades_count = base_test_env.trades_count

        fold_calmar = calculate_calmar_ratio(eval_portfolio_values)

        eval_win_rate_pct = (
            (eval_winning_trades / eval_closed_trades * 100.0)
            if eval_closed_trades > 0
            else 0.0
        )

        fold_results.append(
            {
                "fold": fold_idx + 1,
                "final_portfolio_value": float(base_test_env.portfolio_value),
                "pnl": float(
                    base_test_env.portfolio_value - eval_initial_portfolio_value
                ),
                "calmar": float(fold_calmar),
                "trades": int(eval_trades_count),
                "closed_trades": int(eval_closed_trades),
                "win_rate_pct": float(eval_win_rate_pct),
                "fees_paid": float(base_test_env.fees_paid_total),
            }
        )

        print_if_not_trial(trial, f"Fold {fold_idx + 1} Results:")
        print_if_not_trial(
            trial,
            f"  Final PV: ${base_test_env.portfolio_value:.2f} | PnL: ${base_test_env.portfolio_value - eval_initial_portfolio_value:.2f} | Calmar: {fold_calmar:.2f}",
        )
        print_if_not_trial(
            trial,
            f"  Trades: {eval_trades_count} | Sells: {eval_closed_trades} | Win Rate: {eval_win_rate_pct:.1f}% | Fees: ${base_test_env.fees_paid_total:.4f}",
        )

        last_model = model
        last_test_env = test_env
        last_eval_reward_totals = eval_reward_totals
        last_eval_portfolio_values = eval_portfolio_values
        last_eval_realized_pnl = eval_realized_pnl
        last_eval_trades_count = eval_trades_count
        last_eval_closed_trades = eval_closed_trades
        last_eval_winning_trades = eval_winning_trades
        last_eval_initial_portfolio_value = eval_initial_portfolio_value
        last_eval_steps = eval_steps

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

    buy_hold_final = BUDGET_INITIAL * (1 + buy_hold_return)

    # Multi-seed evaluation on the final model
    multi_seed_pv = []
    if not getattr(args, "skip_multi_seed_eval", False):
        print("5. Performing multi‑seed evaluation...")
        for mseed in range(5):
            np.random.seed(mseed + 100)
            try:
                ms_prices = read_last_n(args.parquet_path, n=args.n_rows)
                # Split by unique timestamps
                unique_times = sorted(ms_prices.open_time.unique())
                split_idx = int(len(unique_times) * (1 - 0.2))
                split_time = unique_times[split_idx]
                ms_test = ms_prices[ms_prices.open_time >= split_time].copy()
                del ms_prices
                gc.collect()
                # Filter to train symbols
                ms_test = ms_test[ms_test["symbol"].isin(last_asset_names)]
                ms_symbols = ms_test["symbol"].unique().tolist()
                missing_symbols = set(last_asset_names) - set(ms_symbols)
                if missing_symbols:
                    dummy_rows = []
                    first_time = ms_test["open_time"].min()
                    for sym in missing_symbols:
                        dummy_rows.append(
                            {
                                "open_time": first_time,
                                "symbol": sym,
                                "open": 0.0,
                                "high": 0.0,
                                "low": 0.0,
                                "close": 0.0,
                                "volume": 0.0,
                            }
                        )
                    ms_test = pd.concat(
                        [ms_test, pd.DataFrame(dummy_rows)], ignore_index=True
                    )
                ms_prices_arr, ms_static_obs, ms_asset_names = (
                    compute_static_obs_from_long_df(ms_test, args.window_size)
                )
                raw_ms_env = MinimalCryptoEnv(
                    prices_arr=ms_prices_arr,
                    static_obs=ms_static_obs,
                    asset_names=ms_asset_names,
                    run_id=run_id,
                    is_eval=True,
                    disable_logging=True,
                    **env_args,
                )
                ms_env = ActionMasker(raw_ms_env, get_action_mask)
                ms_obs, _ = ms_env.reset()
                ms_done = False
                while not ms_done:
                    # Add an explicit batch dimension for the single environment
                    action_masks = np.expand_dims(ms_env.action_masks(), axis=0)
                    ms_action, _ = last_model.predict(
                        ms_obs, action_masks=action_masks, deterministic=True
                    )
                    ms_obs, _, ms_done, _, _ = ms_env.step(ms_action)
                multi_seed_pv.append(raw_ms_env.portfolio_value)
                ms_env.close()
            except Exception as exc:
                print_if_not_trial(trial, f"  seed {mseed} failed: {exc}")

    if multi_seed_pv:
        arr = np.array(multi_seed_pv)
        print_if_not_trial(
            trial,
            f"  n={len(arr)}  mean=${arr.mean():.2f}  std=${arr.std():.2f}  "
            f"min=${arr.min():.2f}  max=${arr.max():.2f}",
        )

    # Dashboard update
    state_file = run_dir / "state.json"
    if args.dashboard and state_file and state_file.exists():
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
            }
            # Add CV mean Calmar ratio to finance section for dashboard display
            state["finance"]["calmar"] = float(cv_mean_calmar)
            state["explainability"] = {
                "cumulative_rewards": {
                    k: float(v) for k, v in last_eval_reward_totals.items()
                },
                "hyperparameters": all_args,
            }
            if "series" not in state:
                state["series"] = {}
            state["series"]["test_portfolio_value"] = last_eval_portfolio_values
            state["series"]["test_realized_pnl"] = last_eval_realized_pnl
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

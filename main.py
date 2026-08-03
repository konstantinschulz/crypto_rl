import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import optuna
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor

from crypto_rl import MinimalCryptoEnv, read_last_n
from crypto_rl.callbacks import DashboardCallback, TrialEvalCallback
from crypto_rl.cli import build_parser
from crypto_rl.data import read_train_test
from scripts.eval_log_action_counter import eval_log_action_counter
from scripts.eval_report import eval_report

# Global budget variable required by other parts of the package
global BUDGET_INITIAL


def print_if_not_trial(trial: optuna.trial.Trial | None = None, msg: str = ""):
    if trial is None:
        print(msg)


def run_experiment(args, trial: optuna.trial.Trial | None = None) -> float:
    """Run a single training/evaluation loop.
    Returns the mean multi‑seed test portfolio value, which Optuna will maximize.
    """
    # Seed for reproducibility of dataset split
    np.random.seed(args.data_seed)

    global BUDGET_INITIAL
    BUDGET_INITIAL = args.budget_initial
    print_if_not_trial(trial, "1. Loading raw data...")
    TEST_FRACTION: float = 0.2
    n_test = round(args.rows * TEST_FRACTION)
    n_train = args.rows - n_test
    train_prices_df, test_prices_df = read_train_test(
        args.parquet_path, n_train, n_test
    )

    run_id = datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S-minimal")

    print_if_not_trial(trial, "2. Setting up environment...")
    train_env = MinimalCryptoEnv(
        train_prices_df,
        window_size=args.window_size,
        run_id=run_id,
        fee_rate=args.fee_rate,
        reward_type=args.reward_type,
        hold_cost_rate=args.hold_cost_rate,
        empty_buy_penalty=args.empty_buy_penalty,
        empty_sell_penalty=args.empty_sell_penalty,
        illegal_sell_penalty=args.illegal_sell_penalty,
        illegal_buy_penalty=args.illegal_buy_penalty,
        trade_freq_incentive=args.trade_freq_incentive,
        profit_bonus=args.profit_bonus,
        parquet_path=args.parquet_path,
        n_rows=args.rows,
        action_space_type=args.action_space_type,
        max_single_step_allocation=args.max_single_step_allocation,
    )

    # Dashboard callback (optional)
    callback = None
    run_dir = None
    state_file = None
    index_file = Path("rl_dashboard_index.json")
    if args.dashboard:
        base_run_dir = Path(args.run_dir) if args.run_dir else Path("logs")
        run_dir = base_run_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        state_file = run_dir / "state.json"
        # Update index file
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
        callback = DashboardCallback(
            state_file,
            window_size=args.window_size,
            reward_type=args.reward_type,
            check_freq=max(1, args.timesteps // 100 if args.timesteps >= 100 else 1),
            run_id=run_id,
            total_timesteps=args.timesteps,
            num_data_rows=args.rows,
        )

    # Optuna evaluation callback (optional)
    eval_callback = None
    eval_env = None
    if trial is not None:
        raw_env = MinimalCryptoEnv(
            test_prices_df,
            window_size=args.window_size,
            run_id=run_id,
            fee_rate=args.fee_rate,
            reward_type=args.reward_type,
            trade_freq_incentive=args.trade_freq_incentive,
            is_eval=True,
            action_space_type=args.action_space_type,
        )
        eval_env = Monitor(raw_env)
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
    seed = args.data_seed
    ent_coef = args.ent_coef
    batch_size = args.batch_size
    gamma = args.gamma
    learning_rate = args.learning_rate
    policy_kwargs = {
        "net_arch": dict(pi=[128, 128], qf=[128, 128]),
        # "net_arch": dict(pi=[256, 256, 128], qf=[256, 256, 128]),
        "activation_fn": torch.nn.ReLU,
        "normalize_images": False,  # not images
    }
    n_steps = args.n_steps

    if args.algorithm == "SAC":
        print_if_not_trial(trial, "3. Training SAC model...")
        model = SAC(
            "MlpPolicy",
            train_env,
            device=device,
            verbose=verbose,
            seed=seed,
            ent_coef=ent_coef if ent_coef != 0.01 else "auto",
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
        )
    else:
        print_if_not_trial(trial, "3. Training PPO model...")
        model = PPO(
            "MlpPolicy",
            train_env,
            device=device,
            verbose=verbose,
            seed=seed,
            ent_coef=ent_coef,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            learning_rate=learning_rate,
            clip_range=args.clip_range,
            policy_kwargs=policy_kwargs | dict(ortho_init=True),
        )

    callbacks = []
    if callback is not None:
        callbacks.append(callback)
    if eval_callback is not None:
        callbacks.append(eval_callback)

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

    print_if_not_trial(trial, "4. Testing the trained model...")
    test_env = MinimalCryptoEnv(
        test_prices_df,
        window_size=args.window_size,
        run_id=run_id,
        fee_rate=args.fee_rate,
        reward_type=args.reward_type,
        trade_freq_incentive=args.trade_freq_incentive,
        is_eval=True,
        action_space_type=args.action_space_type,
    )
    obs, _ = test_env.reset()
    done = False
    eval_steps = 0
    eval_initial_portfolio_value = test_env.portfolio_value
    eval_portfolio_values = [{"step": 0, "value": float(test_env.portfolio_value)}]
    eval_realized_pnl = [{"step": 0, "value": 0.0}]
    eval_closed_trades = 0
    eval_winning_trades = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = test_env.step(action)
        if info.get("is_valid_sell", False):
            eval_closed_trades += 1
            if info.get("realised_pnl", 0.0) > 0:
                eval_winning_trades += 1
        eval_steps += 1
        eval_portfolio_values.append(
            {"step": eval_steps, "value": float(test_env.portfolio_value)}
        )
        eval_realized_pnl.append(
            {
                "step": eval_steps,
                "value": float(test_env.portfolio_value - eval_initial_portfolio_value),
            }
        )

    eval_trades_count = test_env.trades_count
    test_env.close()
    train_env.close()

    print_if_not_trial(trial, "-" * 30)
    print_if_not_trial(trial, "RESULTS:")
    print_if_not_trial(
        trial, f"Final Test Portfolio Value: ${test_env.portfolio_value:.2f}"
    )
    print_if_not_trial(trial, f"Total Trades (Eval):        {eval_trades_count}")
    print_if_not_trial(trial, f"Total Closed Trades (Sells): {eval_closed_trades}")
    print_if_not_trial(
        trial, f"Total Fees Paid (Eval):     ${test_env.fees_paid_total:.4f}"
    )

    # Compute win rate and buy‑hold baseline for dashboard
    eval_win_rate_pct = (
        (eval_winning_trades / eval_closed_trades * 100.0)
        if eval_closed_trades > 0
        else 0.0
    )
    pivoted_df = train_env.prices_df
    first_asset = pivoted_df.columns[0]
    buy_hold_return = (
        pivoted_df.iloc[-1][first_asset] - pivoted_df.iloc[0][first_asset]
    ) / pivoted_df.iloc[0][first_asset]
    buy_hold_final = BUDGET_INITIAL * (1 + buy_hold_return)

    # Multi‑seed evaluation (Tier 5)
    multi_seed_pv = []
    for mseed in range(5):
        np.random.seed(mseed + 100)
        try:
            ms_prices = read_last_n(args.parquet_path, n=args.rows)
            split = int(len(ms_prices.open_time.unique()) * TEST_FRACTION)
            ms_test = ms_prices.iloc[split:]
            ms_env = MinimalCryptoEnv(
                ms_test,
                window_size=args.window_size,
                run_id=run_id,
                fee_rate=args.fee_rate,
                reward_type=args.reward_type,
                is_eval=True,
                action_space_type=args.action_space_type,
            )
            ms_obs, _ = ms_env.reset()
            ms_done = False
            while not ms_done:
                ms_action, _ = model.predict(ms_obs, deterministic=True)
                ms_obs, _, ms_done, _, _ = ms_env.step(ms_action)
            multi_seed_pv.append(ms_env.portfolio_value)
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
    if args.dashboard and state_file and state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            state["run"]["status"] = "evaluated"
            state["run"]["finished_at"] = datetime.now(UTC).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
            state["finance"]["evaluation_results"] = {
                "final_portfolio_value": float(test_env.portfolio_value),
                "pnl": float(test_env.portfolio_value - eval_initial_portfolio_value),
                "evaluation_steps": int(eval_steps),
                "eval_trades": int(eval_trades_count),
                "eval_win_rate_pct": float(eval_win_rate_pct),
                "buy_hold_baseline": float(buy_hold_final),
                "total_fees_paid": float(test_env.fees_paid_total),
            }
            if "series" not in state:
                state["series"] = {}
            state["series"]["test_portfolio_value"] = eval_portfolio_values
            state["series"]["test_realized_pnl"] = eval_realized_pnl
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
    # Return mean of multi‑seed portfolio values (or single test if none)
    if multi_seed_pv:
        return float(np.mean(multi_seed_pv))
    return float(test_env.portfolio_value)


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

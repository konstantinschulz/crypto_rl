from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO, SAC
from crypto_rl import MinimalCryptoEnv, read_last_n
from crypto_rl.callbacks import DashboardCallback
from crypto_rl.cli import build_parser
import numpy as np
import torch
from crypto_rl.data import read_train_test


def main():
    parser = build_parser()
    args = parser.parse_args()
    # Set global budget initial if overridden
    global BUDGET_INITIAL
    BUDGET_INITIAL = args.budget_initial
    # Apply data seed for reproducible dataset selection
    np.random.seed(args.data_seed)
    print("1. Loading raw data...")
    # Split 80/20 into train and test
    n_test = round(args.rows * 0.8)
    n_train = args.rows - n_test
    train_prices_df, test_prices_df = read_train_test(
        args.parquet_path, n_train, n_test
    )
    # If dashboard integration is requested, prepare run directory and callback
    run_id = datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S-minimal")

    print("2. Setting up environment...")
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
    )

    run_dir = None
    state_file = None
    index_file = Path("rl_dashboard_index.json")
    callback = None
    if args.dashboard:
        # Create run directory
        base_run_dir = Path(args.run_dir) if args.run_dir else Path("logs")
        run_dir = base_run_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        state_file = run_dir / "state.json"

        # Update index file to include this run
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    verbose: int = 1
    seed: int = args.data_seed
    ent_coef: float = args.ent_coef
    batch_size: int = args.batch_size
    gamma: float = args.gamma
    learning_rate: float = args.learning_rate
    policy_kwargs: dict[str, Any] = {
        "net_arch": dict(pi=[128, 128], qf=[128, 128]),
        "activation_fn": torch.nn.ReLU,
    }
    n_steps: int = args.n_steps
    if args.algorithm == "SAC":
        print("3. Training simplest SAC model...")
        model = SAC(
            "MlpPolicy",
            train_env,
            device=device,
            verbose=verbose,
            seed=seed,
            # For SAC, ent_coef default can be "auto" or a float
            ent_coef=ent_coef if ent_coef != 0.01 else "auto",
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
        )
    else:
        print("3. Training simplest PPO model...")
        model = PPO(
            "MlpPolicy",
            train_env,
            device=device,
            verbose=verbose,
            seed=seed,
            ent_coef=ent_coef,  # encourage action diversity
            n_steps=n_steps,  # larger rollout buffer for better gradient estimation
            batch_size=batch_size,  # smaller batch size for more frequent updates
            gamma=gamma,
            learning_rate=learning_rate,
            clip_range=args.clip_range,
            policy_kwargs=(policy_kwargs | dict(ortho_init=True)),
        )
    if callback is not None:
        model.learn(total_timesteps=args.timesteps, callback=callback)
    else:
        model.learn(total_timesteps=args.timesteps)

    print("4. Testing the trained model...")
    test_env = MinimalCryptoEnv(
        test_prices_df,
        window_size=args.window_size,
        run_id=run_id,
        fee_rate=args.fee_rate,
        reward_type=args.reward_type,
        trade_freq_incentive=args.trade_freq_incentive,
        is_eval=True,
        action_space_type=args.action_space_type,
    )  # Use same run_id for evaluation logs
    obs, _ = test_env.reset()
    done = False
    eval_trades_count = 0
    eval_winning_trades = 0
    eval_closed_trades = 0
    eval_steps = 0
    eval_initial_portfolio_value = test_env.portfolio_value

    eval_portfolio_values = [{"step": 0, "value": float(test_env.portfolio_value)}]
    eval_realized_pnl = [{"step": 0, "value": 0.0}]

    while not done:
        # Predict the action using our trained model
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

    print("-" * 30)
    print("RESULTS:")
    print(f"Final Test Portfolio Value: ${test_env.portfolio_value:.2f}")
    print(f"Total Trades (Eval):        {eval_trades_count}")
    print(f"Total Closed Trades (Sells): {eval_closed_trades}")
    print(f"Total Fees Paid (Eval):     ${test_env.fees_paid_total:.4f}")
    eval_win_rate_pct = (
        (eval_winning_trades / eval_closed_trades * 100.0)
        if eval_closed_trades > 0
        else 0.0
    )
    print(f"Win Rate (Eval):            {eval_win_rate_pct:.1f}%")

    # Baseline comparison (if we just bought and held the first asset)
    # Use the pivoted prices_df which is now indexed by time and contains floats.
    pivoted_df = train_env.prices_df
    first_asset = pivoted_df.columns[0]
    buy_hold_return = (
        pivoted_df.iloc[-1][first_asset] - pivoted_df.iloc[0][first_asset]
    ) / pivoted_df.iloc[0][first_asset]
    buy_hold_final = BUDGET_INITIAL * (1 + buy_hold_return)
    print(f"Buy/Hold Baseline ({first_asset}): ${buy_hold_final:.2f}")

    # -----------------------------------------------------------------
    # Multi-seed evaluation (Tier 5)
    # Run the same trained policy on 5 independently-sampled price windows
    # from the same held-out time range.  This gives mean ± std terminal
    # portfolio value, which is a far more trustworthy signal than a single
    # run when the test window is itself randomly drawn.
    # -----------------------------------------------------------------
    print("\n5-seed multi-window evaluation:")
    multi_seed_pv = []
    # Seed 0 is already captured by the main eval run above.
    for mseed in range(5):
        np.random.seed(mseed + 100)  # offset to avoid overlapping with the main seed
        try:
            ms_prices = read_last_n(args.parquet_path, n=args.rows)
            ms_split = int(len(ms_prices) * 0.8)
            ms_test = ms_prices.iloc[ms_split:]
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
            print(f"  seed {mseed} failed: {exc}")
    if multi_seed_pv:
        arr = np.array(multi_seed_pv)
        print(
            f"  n={len(arr)}  mean=${arr.mean():.2f}  std=${arr.std():.2f}  "
            f"min=${arr.min():.2f}  max=${arr.max():.2f}"
        )

    # --- Dashboard Update for Evaluation Results ---
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
            print(f"Dashboard state updated with evaluation results in {state_file}")
        except Exception as e:
            print(f"Error updating dashboard state with evaluation results: {e}")


if __name__ == "__main__":
    main()

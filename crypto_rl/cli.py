"""
crypto_rl.cli
=============
Command-line argument definitions for the crypto RL training script.

Call :func:`build_parser` to get a pre-configured :class:`argparse.ArgumentParser`.
"""

import argparse

from crypto_rl.config import RLConfig


def build_parser() -> RLConfig:
    """Return a fully configured argument parser for the training CLI."""
    parser = argparse.ArgumentParser(
        description="Minimal RL training with optional dashboard reporting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --n-rows 10000 --timesteps 20000
  python main.py --dashboard --n-rows 15000 --timesteps 50000 --run-dir logs
        """,
    )
    default_config: RLConfig = RLConfig()
    # ------------------------------------------------------------------ arguments (alphabetical)
    parser.add_argument(
        "--action-dead-zone",
        type=float,
        default=default_config.action_dead_zone,
        help="Dead zone threshold for continuous actions (default 0.15)",
    )
    parser.add_argument(
        "--action-space-type",
        type=str,
        default=default_config.action_space_type,
        choices=["continuous", "multidiscrete"],
        help="Action space representation: continuous (Box) or multidiscrete (MultiDiscrete) (default: continuous)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=default_config.algorithm,
        choices=["PPO", "SAC"],
        help="RL algorithm to use for training: PPO or SAC (default: SAC)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=default_config.batch_size,
        help="Minibatch size for PPO model",
    )
    parser.add_argument(
        "--budget-initial",
        type=float,
        default=default_config.budget_initial,
        help="Initial cash budget for the agent (default 100.0)",
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        default=default_config.checkpoint,
        help="Enable checkpointing of best Calmar model",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=default_config.clip_range,
        help="Clipping parameter for the PPO model, it can be a function of the current progress remaining (from 1 to 0).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=default_config.cv_folds,
        help="Number of walk-forward CV folds (default: 1). Set to 1 for standard single train/test split.",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        default=default_config.dashboard,
        help="Enable dashboard UI",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=default_config.data_seed,
        help="Seed for data subset selection (default: None = random / no seed)",
    )
    parser.add_argument(
        "--empty-buy-penalty",
        type=float,
        default=default_config.empty_buy_penalty,
        help="Penalty multiplier for empty BUY actions (default 0.001)",
    )
    parser.add_argument(
        "--empty-sell-penalty",
        type=float,
        default=default_config.empty_sell_penalty,
        help="Penalty multiplier for empty SELL actions (default 0.001)",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=default_config.ent_coef,
        help="Entropy coefficient for the loss calculation of the PPO model",
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=default_config.fee_rate,
        help="Trading fee rate (flat percentage of trade volume, e.g. 0.001 for 0.1%%)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=default_config.gamma,
        help="Discount factor for PPO model",
    )
    parser.add_argument(
        "--hold-cost-rate",
        type=float,
        default=default_config.hold_cost_rate,
        help=(
            "Hold cost rate per step as a fraction of asset value. "
            "Default 1e-6 ≈ 0.05%% per day at 1-min bars. "
            "The old default (0.0001) was ~6%%/hr — catastrophically high, "
            "causing the agent to never hold any position."
        ),
    )
    parser.add_argument(
        "--hold-incentive",
        type=float,
        default=default_config.hold_incentive,
        help="Micro-incentive reward per asset remaining in the action dead zone (default 0.0005)",
    )
    parser.add_argument(
        "--illegal-buy-penalty",
        type=float,
        default=default_config.illegal_buy_penalty,
        help="Penalty multiplier for illegal BUY actions (default 0.005)",
    )
    parser.add_argument(
        "--illegal-sell-penalty",
        type=float,
        default=default_config.illegal_sell_penalty,
        help="Penalty multiplier for illegal SELL actions (default 0.005)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=default_config.learning_rate,
        help="Learning rate for the PPO model",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=default_config.max_checkpoints,
        help="Maximum number of checkpoints to retain (default 5)",
    )
    parser.add_argument(
        "--max-single-step-allocation",
        type=float,
        default=default_config.max_single_step_allocation,
        help="Maximum amount of cash to be allocated within a single step (default 0.5)",
    )
    parser.add_argument(
        "--min-turnover-threshold",
        type=float,
        default=default_config.min_turnover_threshold,
        help="Minimum portfolio turnover required before executing continuous rebalance (default: 0.02)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=default_config.n_envs,
        help="Number of parallel environments for training (default: 4).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=default_config.n_steps,
        help="The number of steps to run for each PPO model environment per update (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)",
    )
    parser.add_argument(
        "--parquet-path",
        type=str,
        default=default_config.parquet_path,
        help="Path to Parquet dataset file (default: subset.parquet)",
    )
    parser.add_argument(
        "--profit-bonus",
        type=float,
        default=default_config.profit_bonus,
        help="Bonus reward for profitable trades (default 0.15). Encourages the agent to maximize the absolute amount of profit (realized PnL) for each trade.",
    )
    parser.add_argument(
        "--drawdown-penalty-coef",
        type=float,
        default=default_config.drawdown_penalty_coef,
        help="Coefficient for drawdown penalty term (default 0.5)",
    )
    parser.add_argument(
        "--reward-type",
        type=str,
        default=default_config.reward_type,
        choices=["pnl", "excess_return"],
        help="Reward function type to use for training (default: excess_return)",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=default_config.n_rows,
        help=(
            "Number of last rows to load from parquet file (default: 10000 ~7 days). "
            "Lower values use less RAM; higher values gives more training data. "
            "Uses pyarrow row-group-aware reading if available for memory efficiency."
        ),
    )
    parser.add_argument(
        "--base-run-dir",
        type=str,
        default=default_config.base_run_dir,
        help=(
            "Directory where run state will be written (if --dashboard is used). "
            "Default: logs/"
        ),
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=default_config.timesteps,
        help="Total training timesteps (default 20000)",
    )
    parser.add_argument(
        "--skip-multi-seed-eval",
        action="store_true",
        default=default_config.skip_multi_seed_eval,
        help="Skip the slow multi-seed evaluation loop after training (useful for fast smoke tests)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=default_config.window_size,
        help="Observation window size in minutes (default: 10)",
    )
    args = parser.parse_args()
    # Rebuild the config object using the parsed arguments
    final_config = RLConfig(**vars(args))
    return final_config

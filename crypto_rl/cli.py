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
        help=f"Dead zone threshold for continuous actions (default {default_config.action_dead_zone})",
    )
    parser.add_argument(
        "--action-space-type",
        type=str,
        default=default_config.action_space_type,
        choices=["continuous", "multidiscrete"],
        help=f"Action space representation: continuous (Box) or multidiscrete (MultiDiscrete) (default: {default_config.action_space_type})",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=default_config.algorithm,
        choices=["PPO", "SAC"],
        help=f"RL algorithm to use for training: PPO or SAC (default: {default_config.algorithm})",
    )
    parser.add_argument(
        "--base-run-dir",
        type=str,
        default=default_config.base_run_dir,
        help=(
            "Directory where run state will be written (if --dashboard is used). "
            f"Default: {default_config.base_run_dir}"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=default_config.batch_size,
        help=f"Minibatch size for PPO model (default: {default_config.batch_size})",
    )
    parser.add_argument(
        "--budget-initial",
        type=float,
        default=default_config.budget_initial,
        help=f"Initial cash budget for the agent (default {default_config.budget_initial})",
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
        help=f"Number of walk-forward CV folds (default: {default_config.cv_folds}). Set to 1 for standard single train/test split.",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        default=default_config.dashboard,
        help="Enable dashboard UI",
    )
    parser.add_argument(
        "--dashboard-freq",
        type=int,
        default=default_config.dashboard_freq,
        help=f"Frequency (in steps) to write dashboard JSON state (default: {default_config.dashboard_freq})",
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
        help=f"Penalty multiplier for empty BUY actions (default {default_config.empty_buy_penalty})",
    )
    parser.add_argument(
        "--empty-sell-penalty",
        type=float,
        default=default_config.empty_sell_penalty,
        help=f"Penalty multiplier for empty SELL actions (default {default_config.empty_sell_penalty})",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=default_config.ent_coef,
        help="Entropy coefficient for the loss calculation of the PPO model",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=default_config.eval_freq,
        help=f"Frequency (in steps) to run full episode evaluation for Calmar ratio (default: {default_config.eval_freq})",
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
        help=f"Micro-incentive reward per asset remaining in the action dead zone (default {default_config.hold_incentive})",
    )
    parser.add_argument(
        "--illegal-buy-penalty",
        type=float,
        default=default_config.illegal_buy_penalty,
        help=f"Penalty multiplier for illegal BUY actions (default {default_config.illegal_buy_penalty})",
    )
    parser.add_argument(
        "--illegal-sell-penalty",
        type=float,
        default=default_config.illegal_sell_penalty,
        help=f"Penalty multiplier for illegal SELL actions (default {default_config.illegal_sell_penalty})",
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
        help=f"Maximum number of checkpoints to retain (default {default_config.max_checkpoints})",
    )
    parser.add_argument(
        "--max-single-step-allocation",
        type=float,
        default=default_config.max_single_step_allocation,
        help=f"Maximum amount of cash to be allocated within a single step (default {default_config.max_single_step_allocation})",
    )
    parser.add_argument(
        "--min-turnover-threshold",
        type=float,
        default=default_config.min_turnover_threshold,
        help=f"Minimum portfolio turnover required before executing continuous rebalance (default: {default_config.min_turnover_threshold})",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=default_config.n_envs,
        help=f"Number of parallel environments for training (default: {default_config.n_envs}).",
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
        help=f"Path to Parquet dataset file (default: {default_config.parquet_path})",
    )
    parser.add_argument(
        "--profit-bonus",
        type=float,
        default=default_config.profit_bonus,
        help=f"Bonus reward for profitable trades (default {default_config.profit_bonus}). Encourages the agent to maximize the absolute amount of profit (realized PnL) for each trade.",
    )
    parser.add_argument(
        "--drawdown-penalty-coef",
        type=float,
        default=default_config.drawdown_penalty_coef,
        help=f"Coefficient for drawdown penalty term (default {default_config.drawdown_penalty_coef})",
    )
    parser.add_argument(
        "--reward-type",
        type=str,
        default=default_config.reward_type,
        choices=["pnl", "excess_return"],
        help=f"Reward function type to use for training (default: {default_config.reward_type})",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=default_config.n_rows,
        help=f"Number of last rows to load from parquet file (default: {default_config.n_rows}). "
        "Lower values use less RAM; higher values gives more training data. "
        "Uses pyarrow row-group-aware reading if available for memory efficiency.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=default_config.timesteps,
        help=f"Total training timesteps (default {default_config.timesteps})",
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
        help=f"Observation window size in minutes (default: {default_config.window_size})",
    )
    args = parser.parse_args()
    # Rebuild the config object using the parsed arguments
    final_config = RLConfig(**vars(args))
    return final_config

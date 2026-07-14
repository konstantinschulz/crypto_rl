"""
crypto_rl.cli
=============
Command-line argument definitions for the crypto RL training script.

Call :func:`build_parser` to get a pre-configured :class:`argparse.ArgumentParser`.
"""

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Return a fully configured argument parser for the training CLI."""
    parser = argparse.ArgumentParser(
        description="Minimal RL training with optional dashboard reporting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --rows 10000 --timesteps 20000
  python main.py --dashboard --rows 15000 --timesteps 50000 --run-dir logs
        """,
    )

    # ------------------------------------------------------------------ data
    parser.add_argument(
        "--parquet-path",
        type=str,
        default="subset.parquet",
        help="Path to Parquet dataset file (default: subset.parquet)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=10000,
        help=(
            "Number of last rows to load from parquet file (default: 10000 ~7 days). "
            "Lower values use less RAM; higher values gives more training data. "
            "Uses pyarrow row-group-aware reading if available for memory efficiency."
        ),
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=42,
        help="Seed for data subset selection (default: 42)",
    )

    # ------------------------------------------------------------------ training
    parser.add_argument(
        "--action-space-type",
        type=str,
        default="continuous",
        choices=["continuous", "multidiscrete"],
        help="Action space representation: continuous (Box) or multidiscrete (MultiDiscrete) (default: continuous)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="SAC",
        choices=["PPO", "SAC"],
        help="RL algorithm to use for training: PPO or SAC (default: SAC)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=20000,
        help="Total timesteps to train the model (default: 20000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Minibatch size for PPO model",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0003,
        help="Learning rate for the PPO model",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for PPO model",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="Entropy coefficient for the loss calculation of the PPO model",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="Clipping parameter for the PPO model, it can be a function of the current progress remaining (from 1 to 0).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="The number of steps to run for each PPO model environment per update (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)",
    )

    # ------------------------------------------------------------------ environment
    parser.add_argument(
        "--budget-initial",
        type=float,
        default=100.0,
        help="Initial cash budget for the agent (default 100.0)",
    )
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.001,
        help="Trading fee rate (flat percentage of trade volume, e.g. 0.001 for 0.1%)",
    )
    parser.add_argument(
        "--hold-cost-rate",
        type=float,
        default=0.000001,
        help=(
            "Hold cost rate per step as a fraction of asset value. "
            "Default 1e-6 ≈ 0.05%% per day at 1-min bars. "
            "The old default (0.0001) was ~6%%/hr — catastrophically high, "
            "causing the agent to never hold any position."
        ),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Observation window size in minutes (default: 10)",
    )
    parser.add_argument(
        "--reward-type",
        type=str,
        default="excess_return",
        choices=["pnl", "excess_return"],
        help="Reward function type to use for training (default: excess_return)",
    )
    parser.add_argument(
        "--empty-buy-penalty",
        type=float,
        default=0.001,
        help="Penalty multiplier for empty BUY actions (default 0.001)",
    )
    parser.add_argument(
        "--empty-sell-penalty",
        type=float,
        default=0.001,
        help="Penalty multiplier for empty SELL actions (default 0.001)",
    )
    parser.add_argument(
        "--illegal-buy-penalty",
        type=float,
        default=0.005,
        help="Penalty multiplier for illegal BUY actions (default 0.005)",
    )
    parser.add_argument(
        "--illegal-sell-penalty",
        type=float,
        default=0.005,
        help="Penalty multiplier for illegal SELL actions (default 0.005)",
    )
    parser.add_argument(
        "--profit-bonus",
        type=float,
        default=0.15,
        help="Bonus reward for profitable trades (default 0.15). Encourages the agent to maximize the absolute amount of profit (realized PnL) for each trade.",
    )
    parser.add_argument(
        "--trade-freq-incentive",
        type=float,
        default=0.01,
        help="Bonus per profitable trade for frequency incentive (default 0.01). This encourages the agent to trade more frequently.",
    )

    # ------------------------------------------------------------------ dashboard
    parser.add_argument(
        "--dashboard",
        #action="store_true",
        type=bool,
        default=True,
        help=(
            "Enable dashboard integration: creates run entry in rl_dashboard_index.json "
            "and writes periodic state.json for live monitoring in streamlit_dashboard.py"
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help=(
            "Directory where run state will be written (if --dashboard is used). "
            "Default: logs/"
        ),
    )

    return parser

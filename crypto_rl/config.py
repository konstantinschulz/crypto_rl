import os
from dataclasses import asdict, dataclass


@dataclass
class RLConfig:
    algorithm: str = "PPO"
    base_run_dir: str = os.path.abspath("logs")
    batch_size: int = 128
    budget_initial: float = 100.0
    checkpoint: bool = True
    chunks_dir: str = os.path.join(base_run_dir, "chunks")
    clear_cache: bool = False
    clip_range: float = 0.11
    cv_folds: int = 1
    dashboard: bool = True
    dashboard_freq: int = 500  # Fast: Write JSON state
    data_cache_dir: str = os.path.join(os.path.abspath("data"), "memmap_cache")
    data_seed: int | None = 42
    disable_logging: bool = False
    # Optional reward shaping, off by default (ROADMAP P1-1)
    drawdown_penalty_coef: float = 0.0
    ent_coef_final: float = 0.0005
    ent_coef_initial: float = 0.015
    # how long you want average trades to be held; 1 week of 1-minute data = 10,080 steps
    episode_length: int = 10080
    eval_freq: int | str = "auto"
    # Evaluate a fixed, deterministic 2-week slice; 14 days * 1440 minutes = 20,160 steps
    eval_length: int = 20160
    fee_rate: float = 0.001
    gamma: float = 0.99
    learning_rate: float = 1e-4
    logs_dir: str = os.path.abspath("logs")
    # Largest target weight per asset; the top bin of the action space
    max_asset_allocation: float = 0.25
    max_checkpoints: int = 5
    n_envs: int = 9
    n_rows: int = 18000000  # 40000 18000000
    n_steps: int = 512
    net_arch_dim: int = 256
    optuna_dir: str = os.path.join(logs_dir, "optuna")
    parquet_path: str | None = "binance_spot_1m_last4y_single_htf.parquet"
    # No-trade band: an asset is only traded when its target weight differs
    # from its current weight by at least this much (ROADMAP P1-3)
    rebalance_band: float = 0.02
    # "excess_log_return": log(PV_t / PV_{t-1}) minus the equal-weight basket's
    # log return; fees are included in PV.
    # "log_return": the portfolio log return alone.
    # Default chosen by the 2026-10-20 ablation (scripts/reward_ablation.py):
    # the excess variant roughly halved turnover and fees.
    reward_type: str = "excess_log_return"
    skip_multi_seed_eval: bool = True
    test_fraction: float = 0.2
    timesteps: int = 25000  # 3000
    train_chunks: int = 1
    # Optional reward shaping, off by default (ROADMAP P1-1)
    turnover_penalty: float = 0.0
    turnover_penalty_steps_threshold: int = 15
    # Fraction of the data used for validation (intermediate evals, checkpoint
    # ranking, Optuna objective). Carved from the end of what would otherwise be
    # the training slice, directly before each test slice.
    val_fraction: float = 0.1
    # Width of the target-weight bins: {0, w, 2w, ..., max_asset_allocation}
    weight_step: float = 0.05
    window_size: int = 120

    def to_dict(self):
        return asdict(self)

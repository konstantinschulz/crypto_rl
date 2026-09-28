import os
from dataclasses import asdict, dataclass

RULE_PENALTY = 0.0005  # This is one central value for empty_buy_penalty, empty_sell_penalty, illegal_buy_penalty, illegal_sell_penalty


@dataclass
class RLConfig:
    action_dead_zone: float = 0.50
    action_space_type: str = "multidiscrete"
    algorithm: str = "PPO"
    base_run_dir: str = "logs"
    batch_size: int = 128
    budget_initial: float = 100.0
    capital_preservation_bonus: float = 0.0001
    checkpoint: bool = True
    clear_cache: bool = False
    clip_range: float = 0.11
    cv_folds: int = 1
    dashboard: bool = True
    dashboard_freq: int = 500  # Fast: Write JSON state
    data_cache_dir: str = os.path.join(os.path.abspath("data"), "memmap_cache")
    data_seed: int | None = 42
    disable_logging: bool = False
    drawdown_penalty_coef: float = 0.1
    empty_buy_penalty: float = RULE_PENALTY
    empty_sell_penalty: float = RULE_PENALTY
    ent_coef_final: float = 0.0005
    ent_coef_initial: float = 0.015
    # how long you want average trades to be held; 1 week of 1-minute data = 10,080 steps
    episode_length: int = 10080
    eval_freq: int | str = "auto"
    # Evaluate a fixed, deterministic 2-week slice; 14 days * 1440 minutes = 20,160 steps
    eval_length: int = 20160
    fee_rate: float = 0.001
    gamma: float = 0.99
    hold_cost_rate: float = 0.0
    hold_penalty_threshold: float = -0.03
    illegal_buy_penalty: float = RULE_PENALTY
    illegal_sell_penalty: float = RULE_PENALTY
    learning_rate: float = 1e-4
    logs_dir: str = os.path.abspath("logs")
    loss_cut_bonus: float = 0.001
    max_asset_allocation: float = 0.25
    max_checkpoints: int = 5
    max_single_step_allocation: float = 0.15
    n_envs: int = 9
    n_rows: int = 18000000  # 40000 18000000
    n_steps: int = 512
    net_arch_dim: int = 128
    optuna_dir: str = os.path.join(logs_dir, "optuna")
    parquet_path: str | None = "binance_spot_1m_last4y_single_htf.parquet"
    profit_bonus: float = 0.0
    reward_type: str = "excess_return"
    skip_multi_seed_eval: bool = True
    target_volatility: float = 0.02  # 2% target volatility baseline
    test_fraction: float = 0.2
    timesteps: int = 25000  # 3000
    turnover_penalty: float = 0.05
    turnover_penalty_steps_threshold: int = 15
    window_size: int = 120

    def to_dict(self):
        return asdict(self)

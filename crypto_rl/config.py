from dataclasses import asdict, dataclass
from pathlib import Path

RULE_PENALTY = 0.0005  # This is one central value for empty_buy_penalty, empty_sell_penalty, illegal_buy_penalty, illegal_sell_penalty


@dataclass
class RLConfig:
    timesteps: int = 1200000
    action_dead_zone: float = 0.50
    action_space_type: str = "multidiscrete"
    algorithm: str = "PPO"
    base_run_dir: str = "logs"
    batch_size: int = 128
    budget_initial: float = 100.0
    checkpoint: bool = True
    clip_range: float = 0.11
    cv_folds: int = 1
    dashboard: bool = True
    dashboard_freq: int = 500  # Fast: Write JSON state
    data_seed: int | None = None
    disable_logging: bool = False
    drawdown_penalty_coef: float = 0.1
    empty_buy_penalty: float = RULE_PENALTY
    empty_sell_penalty: float = RULE_PENALTY
    ent_coef: float = 0.01
    eval_freq: int = max(
        500, timesteps // 12
    )  # Heavy: Run full episode for Calmar
    fee_rate: float = 0.001
    gamma: float = 0.99
    hold_cost_rate: float = 0.0
    hold_incentive: float = 0.0
    illegal_buy_penalty: float = RULE_PENALTY
    illegal_sell_penalty: float = RULE_PENALTY
    learning_rate: float = 1e-4
    max_checkpoints: int = 5
    max_single_step_allocation: float = 0.15
    min_turnover_threshold: float = 0.02
    n_envs: int = 9
    n_rows: int = 400000
    n_steps: int = 512
    parquet_path: str | None = "binance_spot_1m_last4y_single.parquet"
    profit_bonus: float = 0.0
    reward_type: str = "excess_return"
    skip_multi_seed_eval: bool = True
    window_size: int = 120

    def to_dict(self):
        return asdict(self)

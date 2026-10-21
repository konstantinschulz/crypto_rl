"""Short reward ablation (ROADMAP P1-1).

Trains PPO with each reward variant in ARMS and scores the policy on
the full validation slice, reporting behavior (exposure, turnover, fills,
rebalancing frequency) as well as return. Only the train and validation slices are built
and used; the test fold is never touched. Results go to
logs/reward_ablation/<arm>_s<seed>.json.

Run from the repo root so the default cache paths resolve there:
    python scripts/reward_ablation.py build      # memmap caches + split (once)
    python scripts/reward_ablation.py run --arm elr --seed 0 --timesteps 600000
    python scripts/reward_ablation.py summarize

All arms x 3 seeds in parallel (~1 GB RAM and ~1 h each at 600k steps):
    for a in lr lr_dd elr elr_dd; do for s in 0 1 2; do echo "$a $s"; done; done |
      xargs -P 12 -L 1 bash -c 'python scripts/reward_ablation.py run --arm $0 \
        --seed $1 --timesteps 600000 --evals 4 --threads 1 \
        > logs/reward_ablation/$0_s$1.log 2>&1'

The 2026-10-20 results are summarized in HYPERPARAMETER_RATIONALE.md.
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from crypto_rl.config import RLConfig
from crypto_rl.data import DEFAULT_SYMBOLS, FoldSplit
from crypto_rl.env.data_utils import (
    build_fold_caches,
    compute_static_obs_from_long_df,
    fold_cache_prefixes,
    parquet_open_time_range,
)
from crypto_rl.env.logging_utils import LoggerBase
from crypto_rl.env.metrics import calculate_calmar_ratio
from crypto_rl.env.minimal_env import MinimalCryptoEnv
from crypto_rl.experiment.data_loading import load_raw_data, prepare_splits
from crypto_rl.experiment.evaluation_utils import compute_window_baselines
from crypto_rl.experiment.fold_runner import build_policy_kwargs

OUT = REPO / "logs" / "reward_ablation"


class PrintLogger(LoggerBase):
    def debug(self, message):
        pass  # the env's per-episode PnL-mismatch notes are not needed here

    def info(self, message):
        print(message, flush=True)

    def log(self, message, trial=None):
        print(message, flush=True)

    def print_if_not_trial(self, log_level, trial=None, msg=""):
        print(msg, flush=True)


LOGGER = PrintLogger()

ARMS = {
    "lr": {"reward_type": "log_return", "drawdown_penalty_coef": 0.0},
    "lr_dd": {"reward_type": "log_return", "drawdown_penalty_coef": 0.5},
    "elr": {"reward_type": "excess_log_return", "drawdown_penalty_coef": 0.0},
    "elr_dd": {"reward_type": "excess_log_return", "drawdown_penalty_coef": 0.5},
}


def base_config(**overrides) -> RLConfig:
    # Env settings from scripts/run_large.sh; PPO settings from ROADMAP P2-4
    # (lr 3e-4, clip 0.2, constant ent_coef 0.01), since lr 2e-5 learns
    # next to nothing in a few hundred thousand steps.
    cfg = RLConfig(
        parquet_path=str(REPO / "binance_spot_1m_last4y_single_htf.parquet"),
        n_rows=3_600_000,  # 9 assets -> ~400k bars (~278 days)
        cv_folds=1,
        test_fraction=0.1,
        val_fraction=0.1,
        data_seed=42,
        max_asset_allocation=0.20,
        fee_rate=0.001,
        window_size=240,
        # The 2026-10-20 ablation ran at 1-minute decisions, before P2-1
        decision_interval=1,
        gamma=0.992,
        n_steps=1024,
        batch_size=128,
        n_envs=9,
        net_arch_dim=256,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef_initial=0.01,
        ent_coef_final=0.01,
        disable_logging=True,
        dashboard=False,
        checkpoint=False,
    )
    return dataclasses.replace(cfg, **overrides)


def load_slices(cfg: RLConfig, build: bool = False):
    data_range = parquet_open_time_range(cfg.parquet_path)
    split_file = OUT / "split.json"
    if build:
        raw = load_raw_data(cfg, LOGGER, None)
        split = FoldSplit(*prepare_splits(raw, cfg, LOGGER, None)[0])
        prefixes = fold_cache_prefixes(cfg, len(DEFAULT_SYMBOLS), 0, data_range, split)
        build_fold_caches(raw, cfg, split, prefixes, ("train", "val"))
        del raw
        OUT.mkdir(parents=True, exist_ok=True)
        split_file.write_text(json.dumps(list(split)))
    else:
        split = FoldSplit(*json.loads(split_file.read_text()))
        prefixes = fold_cache_prefixes(cfg, len(DEFAULT_SYMBOLS), 0, data_range, split)
    train = compute_static_obs_from_long_df(None, cfg, cache_prefix=prefixes.train)
    val = compute_static_obs_from_long_df(None, cfg, cache_prefix=prefixes.val)
    return train, val, split


def cmd_build(_args):
    cfg = base_config()
    train, val, split = load_slices(cfg, build=True)
    info = {
        "train_bars": len(train[0]),
        "val_bars": len(val[0]),
        "assets": train[3],
        "train_end": split.train_end_str,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data.json").write_text(json.dumps(info, indent=2))
    print(info)


def _env(cfg, arrays, is_eval):
    # parquet_path=None: reset() must use the given slice, not stream random
    # windows from the whole parquet (that would also reach val/test data).
    # The real runner does the same (runner.py, env_config).
    cfg = dataclasses.replace(cfg, parquet_path=None, n_rows=0)
    prices, static, norm_vol, names = arrays
    e = MinimalCryptoEnv(
        prices_arr=prices,
        static_obs=static,
        norm_vol_arr=norm_vol,
        asset_names=names,
        config=cfg,
        is_eval=is_eval,
        logger=LOGGER,
    )
    return e


def evaluate(model, cfg, val, obs_rms) -> dict:
    """One deterministic episode over the whole validation slice."""
    vec = VecNormalize(
        DummyVecEnv([lambda: _env(cfg, val, is_eval=True)]),
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,
    )
    vec.obs_rms = obs_rms
    obs = vec.reset()
    env: MinimalCryptoEnv = vec.venv.envs[0].unwrapped
    start, end = env.current_step, env.max_steps
    baselines = compute_window_baselines(
        env.prices_arr,
        start_step=start,
        max_steps=end,
        asset_names=env.asset_names,
        fee_rate=env.fee_rate,
        budget_initial=env.portfolio_value,
    )
    pv0 = env.portfolio_value
    pvs, exposure, notional = [pv0], [], 0.0
    rebalance_steps = 0
    held_assets: set[int] = set()
    first_fill = None
    reward_totals: dict[str, float] = {}
    steps, done, info = 0, False, {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        h_before = env.holdings.copy()
        px = env.prices_arr[env.current_step - 1]
        obs, _, dones, infos = vec.step(action)
        done, info = bool(dones[0]), infos[0]
        for k, v in info["reward_components"].items():
            reward_totals[k] = reward_totals.get(k, 0.0) + v
        steps += 1
        if done:
            pvs.append(info["final_portfolio_value"])
            break
        traded = np.abs(env.holdings - h_before) * px
        if traded.sum() > 0:
            rebalance_steps += 1
            if first_fill is None:
                first_fill = steps
        notional += float(traded.sum())
        held_assets |= set(np.flatnonzero(env.holdings > 0).tolist())
        pv = env.portfolio_value
        pvs.append(pv)
        exposure.append(1.0 - env.cash / pv)
    pvs_a = np.array(pvs)
    peak = np.maximum.accumulate(pvs_a)
    final = pvs_a[-1]
    vec.close()
    return {
        "steps": steps,
        "return": final / pv0 - 1.0,
        "excess_vs_equal_weight": (final - baselines["equal_weight"]) / pv0,
        "excess_vs_btc": (final - baselines["btc"]) / pv0,
        "equal_weight_return": baselines["equal_weight"] / pv0 - 1.0,
        "max_drawdown": float(np.max(1.0 - pvs_a / peak)),
        "calmar": calculate_calmar_ratio(
            [{"step": i, "value": float(v)} for i, v in enumerate(pvs_a)]
        ),
        "mean_exposure": float(np.mean(exposure)),
        "exposure_p10_p50_p90": [
            float(q) for q in np.percentile(exposure, [10, 50, 90])
        ],
        "final_exposure": float(exposure[-1]),
        "turnover_x": notional / float(np.mean(pvs_a)),  # traded notional / mean PV
        "fills": int(info["final_trades_count"]),
        "fees_pct": float(info["final_fees_paid"]) / pv0,
        "rebalance_step_share": rebalance_steps / steps,
        "distinct_assets_held": len(held_assets),
        "first_fill_step": first_fill,
        "reward_totals": reward_totals,
    }


def cmd_run(args):
    torch.set_num_threads(args.threads)
    cfg = base_config(**ARMS[args.arm])
    train, val, _ = load_slices(cfg)
    train_env = VecNormalize(
        DummyVecEnv(
            [lambda: _env(cfg, train, is_eval=False) for _ in range(cfg.n_envs)]
        ),
        norm_reward=True,
        norm_obs=True,
        gamma=cfg.gamma,
        clip_obs=10.0,
        clip_reward=5.0,
    )
    train_env.seed(args.seed)
    model = PPO(
        env=train_env,
        policy="MlpPolicy",
        ent_coef=cfg.ent_coef_initial,
        clip_range=cfg.clip_range,
        policy_kwargs=build_policy_kwargs(cfg),
        device=args.device,
        verbose=0,
        seed=args.seed,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
    )
    t0 = time.time()
    evals = []
    chunk = args.timesteps // args.evals
    for i in range(args.evals):
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        r = evaluate(model, cfg, val, train_env.obs_rms)
        r["timesteps"] = model.num_timesteps
        evals.append(r)
        print(
            f"[{args.arm} s{args.seed}] {model.num_timesteps} steps "
            f"({time.time() - t0:.0f}s): ret {r['return']:+.4f} "
            f"xs_ew {r['excess_vs_equal_weight']:+.4f} expo {r['mean_exposure']:.2f} "
            f"turn {r['turnover_x']:.1f} fills {r['fills']}",
            flush=True,
        )
    OUT.mkdir(parents=True, exist_ok=True)
    out = {
        "arm": args.arm,
        "seed": args.seed,
        "arm_config": ARMS[args.arm],
        "timesteps": model.num_timesteps,
        "train_seconds": time.time() - t0,
        "evals": evals,
    }
    (OUT / f"{args.arm}_s{args.seed}.json").write_text(json.dumps(out, indent=2))


def cmd_summarize(_args):
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("*_s*.json"))]
    keys = [
        "return",
        "excess_vs_equal_weight",
        "max_drawdown",
        "mean_exposure",
        "final_exposure",
        "turnover_x",
        "fills",
        "fees_pct",
        "distinct_assets_held",
    ]
    print("per run (final eval):")
    for r in rows:
        e = r["evals"][-1]
        print(
            f"  {r['arm']:7s} s{r['seed']}  "
            + "  ".join(f"{k}={e[k]:.4g}" for k in keys)
            + f"  rebalance_share={e.get('rebalance_step_share', float('nan')):.4g}"
        )
    print("\nper arm, mean ± std over seeds (final eval):")
    for arm in ARMS:
        es = [r["evals"][-1] for r in rows if r["arm"] == arm]
        if not es:
            continue
        cells = []
        for k in keys:
            v = np.array([e[k] for e in es], dtype=float)
            cells.append(f"{k}={v.mean():.4g}±{v.std():.2g}")
        print(f"  {arm:7s} (n={len(es)})  " + "  ".join(cells))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=list(ARMS), required=True)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--timesteps", type=int, default=300_000)
    r.add_argument("--evals", type=int, default=3)
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--device", default="cpu")
    sub.add_parser("summarize")
    args = p.parse_args()
    {"build": cmd_build, "run": cmd_run, "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    main()

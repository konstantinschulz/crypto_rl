"""Supervised signal probe: do the agent's features predict forward returns?

PPO shows no significant edge even at zero fees (2026-10-26). This script asks
the more basic question without RL noise. For each walk-forward fold it fits
models on the training slice and scores them on the validation slice (never
the test slice, ROADMAP P0-4):

- Cross-sectional: each asset's forward log return minus the cross-asset mean
  (asset selection relative to the equal-weight basket).
- Market timing: the cross-asset mean forward log return (basket vs cash).

Features at row i are what the agent observes for a decision filled at
prices[i]: the 13 per-asset static indicators, the 5 macro features, and
vol-normalized past returns (1 min to 24 h) that summarize the price windows,
plus their cross-sectional demeaned versions. The "+long" set adds 3- and
7-day momentum, which the observation cannot see.

Reported per horizon: rank IC (t-stat over non-overlapping windows) and two
simple strategies, gross and net of fees, with their turnover: a long-only
top-3 tilt rebalanced every horizon vs the basket, and basket-or-cash timing.
Cross-sectional ICs are also split into a static part (a fixed asset ranking
from the end of the training slice) and a time-varying part, because a fixed
tilt toward assets that happened to do well can pass for a signal.

Usage:
    .conda/bin/python scripts/signal_probe.py
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from crypto_rl.env.data_utils import FEATURE_VERSION
from crypto_rl.env.feature_utils import MACRO_DIM, STATIC_PER_ASSET_DIM

STATIC_NAMES = [
    "vol_norm",
    "intrabar_vol",
    "vwap_dev",
    "mom_norm",
    "rsi",
    "macd",
    "rel_mom_norm",
    "rvol_24h",
    "vwap_dev_24h",
    "btc_corr_60m",
    "htf_slope_15m",
    "htf_slope_1h",
    "htf_regime_24h",
]
MACRO_NAMES = [
    "btc_mom_norm",
    "btc_vol_norm",
    "btc_regime_0",
    "btc_regime_1",
    "btc_regime_2",
]
# Past-return lags in bars; the observation's windows reach back 1440 bars
OBS_LAGS = [1, 5, 15, 30, 60, 120, 240, 720, 1440]
LONG_LAGS = [4320, 10080]
WARMUP = max(LONG_LAGS)  # rows dropped at the start of each slice
HORIZONS = [60, 240, 1440]
STRIDE = 15  # sample at the 15-minute decision cadence
# 30 days of decision rows, for the static / time-varying split
STATIC_WINDOW = 30 * 96
BARS_PER_YEAR = 1440 * 365


def load_slice(cache_dir: str, prefix_glob: str, split: str) -> dict:
    matches = glob.glob(os.path.join(cache_dir, f"{prefix_glob}_{split}_*_prices.npy"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {split} cache for {prefix_glob}, got {matches}"
        )
    base = matches[0][: -len("_prices.npy")]
    with open(base + "_names.json") as f:
        names = json.load(f)
    return {
        "prices": np.asarray(
            np.load(base + "_prices.npy", mmap_mode="r"), dtype=np.float64
        ),
        "static": np.load(base + "_static.npy", mmap_mode="r"),
        "norm_vol": np.asarray(
            np.load(base + "_norm_vol.npy", mmap_mode="r"), dtype=np.float64
        ),
        "names": names,
    }


def build(data: dict, horizons: list[int], long_features: bool):
    """Feature tensor X (times, assets, k), targets {h: (times, assets)}, names."""
    P, S, V = data["prices"], data["static"], data["norm_vol"]
    T, N = P.shape
    max_h = max(horizons)
    rows = np.arange(WARMUP, T - max_h, STRIDE)
    logp = np.log(P)
    safe_vol = np.maximum(V[rows], 1e-8)

    feats, names = [], []
    for f, name in enumerate(STATIC_NAMES):
        col = MACRO_DIM + f * N
        feats.append(np.asarray(S[rows, col : col + N], dtype=np.float64))
        names.append(name)
    lags = OBS_LAGS + (LONG_LAGS if long_features else [])
    for lag in lags:
        r = logp[rows] - logp[rows - lag]
        feats.append(r / (safe_vol * np.sqrt(lag)))
        names.append(f"ret_{lag}")
    # Cross-sectional versions: the target is relative to the basket
    for i in range(len(feats)):
        feats.append(feats[i] - feats[i].mean(axis=1, keepdims=True))
        names.append(f"xs_{names[i]}")
    macro = np.asarray(S[rows, :MACRO_DIM], dtype=np.float64)
    for m, name in enumerate(MACRO_NAMES):
        feats.append(np.repeat(macro[:, m : m + 1], N, axis=1))
        names.append(name)
    X = np.stack(feats, axis=2)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    targets = {h: logp[rows + h] - logp[rows] for h in horizons}
    return X, targets, names


def standardize(train: np.ndarray, *others: np.ndarray):
    mu = train.reshape(-1, train.shape[-1]).mean(axis=0)
    sd = train.reshape(-1, train.shape[-1]).std(axis=0) + 1e-8
    return [np.clip((a - mu) / sd, -5, 5) for a in (train, *others)]


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    Xb = np.hstack([X, np.ones((len(X), 1))])
    reg = alpha * len(X) * np.eye(Xb.shape[1])
    reg[-1, -1] = 0.0
    return np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)


def ridge_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return X @ w[:-1] + w[-1]


def fit_ridge(X: np.ndarray, y: np.ndarray, gap: int) -> np.ndarray:
    """Pick alpha on the last 20% of the training slice, then refit on all.

    ``gap`` rows between the inner split's halves keep overlapping targets
    from leaking into the inner validation.
    """
    cut = int(len(X) * 0.8)
    best_alpha, best_ic = 1e-3, -np.inf
    for alpha in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        w = ridge_fit(X[: cut - gap], y[: cut - gap], alpha)
        ic = np.corrcoef(ridge_predict(X[cut:], w), y[cut:])[0, 1]
        if ic > best_ic:
            best_alpha, best_ic = alpha, ic
    return ridge_fit(X, y, best_alpha)


def fit_mlp(X: np.ndarray, y: np.ndarray, gap: int, seed: int = 0, loss: str = "huber"):
    """Two-layer MLP with early stopping on the last 20% of the training slice.

    ``loss="huber"`` down-weights large moves, so predictions lean toward the
    median outcome; ``"mse"`` targets the mean, which is what a portfolio earns.
    """
    import torch  # imported here: paper trading uses the ridge only (~500 MB)

    loss_fn = {
        "huber": torch.nn.functional.huber_loss,
        "mse": torch.nn.functional.mse_loss,
    }[loss]
    torch.manual_seed(seed)
    cut = int(len(X) * 0.8)
    y_sd = y.std() + 1e-12
    Xt = torch.tensor(X[: cut - gap], dtype=torch.float32)
    yt = torch.tensor(y[: cut - gap] / y_sd, dtype=torch.float32)
    Xv = torch.tensor(X[cut:], dtype=torch.float32)
    yv = y[cut:]
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    best_state, best_ic, patience = None, -np.inf, 0
    for _ in range(30):
        net.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 4096):
            idx = perm[i : i + 4096]
            batch_loss = loss_fn(net(Xt[idx]).squeeze(1), yt[idx])
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            ic = np.corrcoef(net(Xv).squeeze(1).numpy(), yv)[0, 1]
        if ic > best_ic + 1e-4:
            best_ic, patience = ic, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience >= 4:
                break
    net.load_state_dict(best_state)
    net.eval()
    return lambda Z: (
        net(torch.tensor(Z, dtype=torch.float32)).squeeze(1).detach().numpy()
    )


def xs_rank_ic(pred: np.ndarray, target: np.ndarray, step: int):
    """Mean per-time Spearman IC across assets; t-stat over times ``step`` apart."""
    from scipy.stats import rankdata

    rp = np.apply_along_axis(rankdata, 1, pred)
    rt = np.apply_along_axis(rankdata, 1, target)
    rp -= rp.mean(axis=1, keepdims=True)
    rt -= rt.mean(axis=1, keepdims=True)
    ic = (rp * rt).sum(1) / np.sqrt((rp**2).sum(1) * (rt**2).sum(1) + 1e-12)
    sub = ic[::step]
    return ic.mean(), sub.mean() / (sub.std(ddof=1) + 1e-12) * np.sqrt(len(sub))


def ts_ic(pred: np.ndarray, target: np.ndarray, step: int):
    """Spearman correlation over non-overlapping times and its t-stat."""
    from scipy.stats import rankdata

    a, b = rankdata(pred[::step]), rankdata(target[::step])
    r = np.corrcoef(a, b)[0, 1]
    return r, r * np.sqrt((len(a) - 2) / max(1e-12, 1 - r**2))


def split_static(pred_tr: np.ndarray, pred_va: np.ndarray, window: int):
    """Static and time-varying parts of cross-sectional predictions.

    Static: each asset's mean prediction over the last ``window`` training
    rows, held constant (a fixed asset ranking). Time-varying: the prediction
    minus its trailing ``window``-row per-asset mean (no lookahead).
    """
    T, N = pred_va.shape
    static = np.broadcast_to(pred_tr[-window:].mean(axis=0), (T, N)).copy()
    # Break exact rank ties of the constant predictor at random
    static += 1e-9 * np.random.default_rng(0).standard_normal((T, N))
    both = np.vstack([pred_tr[-window:], pred_va])
    cs = np.cumsum(np.vstack([np.zeros((1, N)), both]), axis=0)
    trailing = (cs[window:-1] - cs[: -window - 1]) / window
    return static, pred_va - trailing[-T:]


def top3_tilt(pred: np.ndarray, fwd: np.ndarray, step: int, fee: float):
    """Equal-weight top 3 vs the basket, rebalanced every ``step`` rows.

    Returns annualized (gross excess, net excess) log returns and turnover per
    day (x PV). Drift between rebalances is ignored.
    """
    idx = np.arange(0, len(pred), step)
    n = pred.shape[1]
    w_prev = np.full(n, 1.0 / n)
    gross, cost, turnover = 0.0, 0.0, 0.0
    for t in idx:
        w = np.zeros(n)
        w[np.argsort(pred[t])[-3:]] = 1.0 / 3.0
        tv = np.abs(w - w_prev).sum()
        r = np.expm1(fwd[t])
        gross += np.log1p(w @ r) - np.log1p(r.mean())
        cost += -np.log1p(-fee * tv)
        turnover += tv
        w_prev = w
    years = len(idx) * step * STRIDE / BARS_PER_YEAR
    days = years * 365
    return gross / years, (gross - cost) / years, turnover / days


def timing(pred: np.ndarray, fwd: np.ndarray, step: int, fee: float):
    """Basket when the predicted basket return is positive, else cash."""
    idx = np.arange(0, len(pred), step)
    e_prev = 1.0
    gross, cost, turnover = 0.0, 0.0, 0.0
    for t in idx:
        e = 1.0 if pred[t] > 0 else 0.0
        r = np.expm1(fwd[t])
        gross += np.log1p(e * r) - np.log1p(r)
        cost += -np.log1p(-fee * abs(e - e_prev))
        turnover += abs(e - e_prev)
        e_prev = e
    years = len(idx) * step * STRIDE / BARS_PER_YEAR
    return gross / years, (gross - cost) / years, turnover / (years * 365)


def probe_fold(train: dict, val: dict, long_features: bool, fee: float, seed: int):
    Xtr, ytr, names = build(train, HORIZONS, long_features)
    Xva, yva, _ = build(val, HORIZONS, long_features)
    Xtr, Xva = standardize(Xtr, Xva)
    _, N, K = Xtr.shape
    out = {}
    for h in HORIZONS:
        step = max(1, h // STRIDE)  # non-overlapping windows
        gap = step + 1
        res = {}
        # Cross-sectional: pooled over (time, asset), target demeaned per time
        xs_tr = ytr[h] - ytr[h].mean(axis=1, keepdims=True)
        xs_va = yva[h] - yva[h].mean(axis=1, keepdims=True)
        flat_tr, flat_va = Xtr.reshape(-1, K), Xva.reshape(-1, K)
        # Rows are time-major, so the inner split stays chronological
        w = fit_ridge(flat_tr, xs_tr.reshape(-1), gap * N)
        mlp = fit_mlp(flat_tr, xs_tr.reshape(-1), gap * N, seed)
        for model, predict in (
            ("ridge", lambda Z, w=w: ridge_predict(Z, w)),
            ("mlp", mlp),
        ):
            pred = predict(flat_va).reshape(-1, N)
            ic, t = xs_rank_ic(pred, xs_va, step)
            g, n_, tv = top3_tilt(pred, yva[h], step, fee)
            static, varying = split_static(
                predict(flat_tr).reshape(-1, N), pred, STATIC_WINDOW
            )
            res[f"xs_{model}"] = {
                "ic": ic,
                "t": t,
                "gross": g,
                "net": n_,
                "turnover_day": tv,
                "ic_static": xs_rank_ic(static, xs_va, step)[0],
                "ic_varying": xs_rank_ic(varying, xs_va, step)[0],
            }
        # Univariate cross-sectional ICs of the raw features
        res["xs_univariate"] = {
            names[k]: xs_rank_ic(Xva[:, :, k], xs_va, step)[0] for k in range(K)
        }
        # Market timing: one row per time, features averaged over assets
        mk_tr, mk_va = Xtr.mean(axis=1), Xva.mean(axis=1)
        m_tr, m_va = ytr[h].mean(axis=1), yva[h].mean(axis=1)
        w = fit_ridge(mk_tr, m_tr, gap)
        mlp = fit_mlp(mk_tr, m_tr, gap, seed)
        for model, predict in (
            ("ridge", lambda Z, w=w: ridge_predict(Z, w)),
            ("mlp", mlp),
        ):
            pred = predict(mk_va)
            ic, t = ts_ic(pred, m_va, step)
            g, n_, tv = timing(pred, m_va, step, fee)
            res[f"mkt_{model}"] = {
                "ic": ic,
                "t": t,
                "gross": g,
                "net": n_,
                "turnover_day": tv,
            }
        out[h] = res
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cache-dir", default="data/memmap_cache")
    p.add_argument("--data-seed", type=int, default=42)
    p.add_argument("--n-rows", type=int, default=18_000_000)
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--window-size", type=int, default=240)
    p.add_argument("--fee-rate", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="Optional JSON file for all results")
    args = p.parse_args()

    results = {}
    for fold in range(args.cv_folds):
        prefix = (
            f"seed_{args.data_seed}_rows_{args.n_rows}_assets_*_folds_{args.cv_folds}"
            f"_val_{args.val_fraction}_test_{args.test_fraction}_win_{args.window_size}"
            f"_feat_v{FEATURE_VERSION}_fold_{fold}"
        )
        train = load_slice(args.cache_dir, prefix, "train")
        val = load_slice(args.cache_dir, prefix, "val")
        assert train["names"] == val["names"]
        assert train["static"].shape[1] == MACRO_DIM + STATIC_PER_ASSET_DIM * len(
            train["names"]
        )
        for long_features in (False, True):
            key = f"fold{fold + 1}_{'obs+long' if long_features else 'obs'}"
            print(
                f"[{key}] train {train['prices'].shape[0]} bars, val {val['prices'].shape[0]} bars",
                flush=True,
            )
            results[key] = probe_fold(
                train, val, long_features, args.fee_rate, args.seed
            )

    def row(label, r):
        split = (
            f" | IC static {r['ic_static']:+.3f}, time-varying {r['ic_varying']:+.3f}"
            if "ic_static" in r
            else ""
        )
        return (
            f"  {label:<10} IC {r['ic']:+.3f} (t {r['t']:+5.2f}) | gross {r['gross']:+.2f}/yr"
            f" | net {r['net']:+.2f}/yr | turnover {r['turnover_day']:.2f}x/day{split}"
        )

    for h in HORIZONS:
        print(f"\n===== Horizon {h} bars ({h / 60:g} h), fee {args.fee_rate:.2%} =====")
        for key, res in results.items():
            r = res[h]
            print(f"{key}")
            for model in ("xs_ridge", "xs_mlp", "mkt_ridge", "mkt_mlp"):
                print(row(model, r[model]))
        # Univariate features whose sign agrees across all folds (obs+long set)
        uni = [
            results[k][h]["xs_univariate"] for k in results if k.endswith("obs+long")
        ]
        consistent = {
            name: [u[name] for u in uni]
            for name in uni[0]
            if all(np.sign(u[name]) == np.sign(uni[0][name]) for u in uni)
        }
        top = sorted(consistent.items(), key=lambda kv: -min(abs(v) for v in kv[1]))[:8]
        print("  Univariate XS rank IC, same sign in every fold (obs+long):")
        for name, ics in top:
            print(f"    {name:<22} " + " ".join(f"{v:+.3f}" for v in ics))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, default=float)


if __name__ == "__main__":
    main()

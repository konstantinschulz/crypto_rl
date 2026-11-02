"""Paper trading of the Kraken low-turnover strategy, on live data.

The strategy is the one frozen before the fold-3 holdout (2026-10-28; see
kraken_backtest.py): a ridge model predicts each coin's 7-day return over the
basket from the coin panel features plus the Binance futures features. Each
day its predictions are z-scored across the coins Kraken lists in EUR and
smoothed with an EMA (half-life 2 days). Once a week the portfolio buys the
top 25% of coins, keeps a held coin while it ranks in the top 50%, and caps a
coin at 1.5× the equal weight (kraken_backtest.select_weights).

No orders are sent. Fills are simulated at Kraken's live quotes as market
orders: buys at the ask and sells at the bid, plus the taker fee of the
tier implied by the paper volume, and Kraken's minimum order sizes apply.
A small account (€50-100) holds fewer coins, each at least MIN_POSITION_EUR,
and stays in the entry fee tier.
The equal-weight basket of the same coins (rebalanced at each rebalance,
without costs) is tracked alongside as the benchmark.

Files under data/paper/: model.npz (the frozen model), state.json (cash,
holdings, basket), equity.csv (one row per daily step), trades.csv.

Usage:
    .conda/bin/python scripts/paper_trade.py fit            # once (~7 GB RAM)
    .conda/bin/python scripts/paper_trade.py init --capital-eur 10000
    .conda/bin/python scripts/paper_trade.py step           # daily after 01:27 UTC
    .conda/bin/python scripts/paper_trade.py report
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

import coin_panel
import kraken_costs
import live_data
import signal_probe
from kraken_backtest import daily_zscore, ema, select_weights
from signal_walk_forward import standardization_stats, standardize

repo_root = Path(__file__).resolve().parent.parent
PAPER_DIR = repo_root / "data" / "paper"
MODEL_PATH = live_data.PAPER_MODEL_PATH
STATE_PATH = PAPER_DIR / "state.json"
EQUITY_PATH = PAPER_DIR / "equity.csv"
TRADES_PATH = PAPER_DIR / "trades.csv"

# The frozen variant (kraken_backtest.py --variant 2 7 0.5, defaults for the
# rest)
HORIZON_DAYS = 7
HALF_LIFE_DAYS = 2
REBALANCE_DAYS = 7
TOP_FRACTION = 0.25
EXIT_FRACTION = 0.5
WEIGHT_CAP = 1.5
# Daily decision time of the backtest's later years (see live_data
# GRID_OFFSET_MINUTES)
DECISION_TIME_OF_DAY = pd.Timedelta(hours=1, minutes=27)
# Daily predictions behind the EMA: after 14 days the start's weight is <1%
SMOOTHING_DAYS = 14
# Refuse to trade on a broken data update
MIN_TRADABLE_COINS = 40
# Smallest position at a rebalance (EUR): Kraken's minimum orders are worth
# €3.5-5.5 (2026-10-28), and a position has to clear them. A €100 account holds
# at most 16 coins; backtests with this limit (kraken_backtest.py
# --min-order-eur 5 --min-position-eur 6) keep the edge at €100 and lose most
# of it at €50, where the 8 coins it can hold are too concentrated.
MIN_POSITION_EUR = 6.0


# --- model ---------------------------------------------------------------


def fit() -> None:
    """Fit the ridge on all history (2022-02 → 2026-01), as the walk-forward
    does at each refit, and save it with its standardization."""
    panel = coin_panel.load_panel(include_holdout=True)
    derivatives = coin_panel.derivatives_features(panel)
    features, names = coin_panel.model_features(panel, derivatives)
    del derivatives
    tradable = panel["tradable"]
    horizon_rows = HORIZON_DAYS * coin_panel.ROWS_PER_DAY
    forward_log = coin_panel.forward_log_returns(panel["close"], horizon_rows)
    has_forward = tradable & np.isfinite(forward_log)
    target = coin_panel.cross_sectional_demean(np.expm1(forward_log), has_forward)
    usable = tradable & np.isfinite(target)

    stride = 4  # hourly rows, as signal_walk_forward --train-row-stride
    train_times = np.arange(0, len(target) - horizon_rows, stride)
    train_mask = usable[train_times]
    x = features[train_times][train_mask]
    y = target[train_times][train_mask]
    del features
    low, high = np.percentile(y, [0.5, 99.5])
    y = np.clip(y, low, high)
    mean, std = standardization_stats(x)
    x = standardize(x, mean, std)
    coins_per_time = int(np.median(train_mask.sum(1)))
    gap = (horizon_rows // stride + 1) * coins_per_time
    weights = signal_probe.fit_ridge(x, y, gap)
    in_sample = np.corrcoef(signal_probe.ridge_predict(x, weights), y)[0, 1]

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        MODEL_PATH,
        weights=weights,
        mean=mean,
        std=std,
        feature_names=np.array(names),
        symbols=panel["symbols"],
        trained_through=panel["decision_times"][-1],
    )
    print(
        f"Saved {MODEL_PATH}: {len(y)} training rows through "
        f"{panel['decision_times'][-1]}, {len(names)} features, in-sample "
        f"correlation {in_sample:.3f}"
    )


def daily_signal(panel: dict, model: dict, on_kraken: np.ndarray) -> dict:
    """Today's smoothed cross-sectional signal and the eligible coins."""
    n_rows = len(panel["decision_times"])
    rows = n_rows - 1 - coin_panel.ROWS_PER_DAY * np.arange(SMOOTHING_DAYS)[::-1]
    rows = rows[rows >= 0]
    sliced = {
        k: (
            v[rows]
            if k
            in ("close", "features", "lag_returns", "norm_vol", "tradable", "macro")
            else v
        )
        for k, v in panel.items()
        if k != "derivatives"
    }
    derivatives, derivative_names = panel["derivatives"]
    features, names = coin_panel.model_features(
        sliced, (derivatives[rows], derivative_names)
    )
    assert names == list(model["feature_names"]), "features differ from the model's"
    x = standardize(features.reshape(-1, len(names)), model["mean"], model["std"])
    predictions = signal_probe.ridge_predict(x, model["weights"]).reshape(len(rows), -1)

    eligible = sliced["tradable"] & on_kraken
    stale = np.isin(panel["symbols"], panel["stale_symbols"])
    eligible[-1] &= ~stale
    predictions = np.where(eligible, predictions, np.nan)
    smoothed = ema(daily_zscore(predictions, eligible), HALF_LIFE_DAYS)
    signal = smoothed[-1]
    return {
        "signal": signal,
        "eligible": eligible[-1] & np.isfinite(signal),
        "raw_prediction": predictions[-1],
    }


# --- Kraken quotes and simulated fills ----------------------------------


def kraken_quotes(session: requests.Session, pairs: pd.DataFrame) -> dict:
    """symbol → (bid, ask) from Kraken's ticker, plus the EUR/USD rate."""
    result = kraken_costs._get(
        session, "Ticker", pair=",".join([*pairs["pair"], "ZEURZUSD"])
    )
    quotes = {}
    for symbol, pair in zip(pairs["symbol"], pairs["pair"]):
        if pair in result:
            quotes[symbol] = (float(result[pair]["b"][0]), float(result[pair]["a"][0]))
    eur_usd = float(result["ZEURZUSD"]["c"][0])
    return {"quotes": quotes, "eur_usd": eur_usd}


def load_state() -> dict:
    with open(STATE_PATH) as f:
        return json.load(f)


def save_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    tmp.replace(STATE_PATH)


def append_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def thirty_day_volume_eur(now: pd.Timestamp) -> float:
    if not TRADES_PATH.exists():
        return 0.0
    trades = pd.read_csv(TRADES_PATH, parse_dates=["time"])
    recent = trades["time"] >= now - pd.Timedelta(days=30)
    return float(trades.loc[recent, "value_eur"].sum())


def rebalance(
    state: dict,
    symbols: list[str],
    signal: dict,
    quotes: dict,
    pairs: pd.DataFrame,
    eur_usd: float,
    now: pd.Timestamp,
) -> list[dict]:
    """Trade the paper portfolio to its new target weights; returns the trades."""
    units = state["units"]
    mid = {s: (b + a) / 2 for s, (b, a) in quotes.items()}
    # A held coin without a quote (delisted, halted) is kept and left out
    holdings_value = {s: u * mid[s] for s, u in units.items() if u > 0 and s in mid}
    value = state["cash_eur"] + sum(holdings_value.values())
    drifted = np.array([holdings_value.get(s, 0.0) / value for s in symbols])
    coins = np.where(signal["eligible"] & np.isin(symbols, list(quotes)))[0]
    target = select_weights(
        signal["signal"],
        coins,
        drifted,
        TOP_FRACTION,
        EXIT_FRACTION,
        WEIGHT_CAP,
        max_holdings=int(value // MIN_POSITION_EUR),
    )
    _, taker_fee = kraken_costs.fee_rates(thirty_day_volume_eur(now) * eur_usd)
    info = pairs.set_index("symbol")
    trades = []

    def record(symbol, side, quantity, price, fee):
        trades.append(
            {
                "time": now.isoformat(),
                "symbol": symbol,
                "side": side,
                "units": quantity,
                "price_eur": price,
                "value_eur": quantity * price,
                "fee_eur": fee,
                "half_spread_eur": quantity * abs(price - mid[symbol]),
            }
        )

    # Sells first, so their cash pays for the buys
    deltas = {
        s: target[j] * value - holdings_value.get(s, 0.0) for j, s in enumerate(symbols)
    }
    for symbol, delta in sorted(deltas.items(), key=lambda kv: kv[1]):
        if delta >= 0 or units.get(symbol, 0) <= 0:
            continue
        bid = quotes[symbol][0] if symbol in quotes else None
        if bid is None:
            print(f"  {symbol}: held but no Kraken quote; kept", flush=True)
            continue
        quantity = min(units[symbol], -delta / mid[symbol])
        remainder = units[symbol] - quantity
        if (
            remainder < info.at[symbol, "ordermin"]
            or remainder * bid < info.at[symbol, "costmin_eur"]
        ):
            quantity = units[symbol]  # don't leave unsellable dust behind
        if (
            quantity < info.at[symbol, "ordermin"]
            or quantity * bid < info.at[symbol, "costmin_eur"]
        ):
            continue  # below Kraken's minimum (a whole position clears it)
        quantity = round(quantity, int(info.at[symbol, "lot_decimals"]))
        fee = quantity * bid * taker_fee
        state["cash_eur"] += quantity * bid - fee
        units[symbol] = units[symbol] - quantity
        if units[symbol] <= 0:
            units.pop(symbol)
        record(symbol, "sell", quantity, bid, fee)

    buys = {s: d for s, d in deltas.items() if d > 0}
    budget = state["cash_eur"]
    scale = min(1.0, budget / sum(buys.values())) if buys else 1.0
    for symbol, delta in sorted(buys.items(), key=lambda kv: -kv[1]):
        ask = quotes[symbol][1]
        spend = delta * scale  # EUR including the fee
        quantity = spend / (ask * (1 + taker_fee))
        quantity = np.floor(quantity * 10 ** int(info.at[symbol, "lot_decimals"])) / (
            10 ** int(info.at[symbol, "lot_decimals"])
        )
        if (
            quantity * ask < info.at[symbol, "costmin_eur"]
            or quantity < info.at[symbol, "ordermin"]
        ):
            continue  # below Kraken's minimum; the cash waits
        fee = quantity * ask * taker_fee
        state["cash_eur"] -= quantity * ask + fee
        units[symbol] = units.get(symbol, 0.0) + quantity
        record(symbol, "buy", quantity, ask, fee)

    # The benchmark: equal weight over the same eligible coins, at mid, free
    basket_value = state["basket_cash_eur"] + sum(
        u * mid[s] for s, u in state["basket_units"].items() if s in mid
    )
    eligible_symbols = [symbols[j] for j in coins]
    state["basket_units"] = {
        s: basket_value / len(eligible_symbols) / mid[s] for s in eligible_symbols
    }
    state["basket_cash_eur"] = 0.0
    state["last_rebalance"] = now.isoformat()
    state["taker_fee"] = taker_fee
    return trades


# --- commands ------------------------------------------------------------


def daily_decision_time(now: pd.Timestamp) -> pd.Timestamp:
    """The latest daily decision time (01:27 UTC) that has passed."""
    today = now.floor("D") + DECISION_TIME_OF_DAY
    return today if now >= today else today - pd.Timedelta(days=1)


def init(capital_eur: float) -> None:
    if STATE_PATH.exists():
        raise SystemExit(f"{STATE_PATH} exists; delete it to start over")
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    save_state(
        {
            "created": pd.Timestamp.now(tz="UTC").isoformat(),
            "capital_eur": capital_eur,
            "cash_eur": capital_eur,
            "units": {},
            "basket_cash_eur": capital_eur,
            "basket_units": {},
            "last_rebalance": None,
            "last_step": None,
        }
    )
    print(f"Paper account with €{capital_eur:,.0f} created in {STATE_PATH}")


def step(force_rebalance: bool = False) -> None:
    now = pd.Timestamp.now(tz="UTC")
    decision_time = daily_decision_time(now)
    state = load_state()
    if state["last_step"] == decision_time.isoformat() and not force_rebalance:
        print(f"Already stepped for {decision_time}; nothing to do")
        return
    model = dict(np.load(MODEL_PATH, allow_pickle=True))
    pairs = pd.read_csv(kraken_costs.PAIRS_PATH)

    symbols = [str(s) for s in model["symbols"]]
    panel = live_data.live_panel(live_data.Client(), decision_time, symbols)
    fresh = ~np.isin(symbols, panel["stale_symbols"])
    n_tradable = int((panel["tradable"][-1] & fresh).sum())
    if n_tradable < MIN_TRADABLE_COINS:
        raise SystemExit(
            f"Only {n_tradable} tradable coins with fresh data at {decision_time}; "
            "data update looks broken, not trading"
        )
    on_kraken = np.isin(symbols, pairs["symbol"])
    signal = daily_signal(panel, model, on_kraken)

    session = requests.Session()
    market = kraken_quotes(session, pairs)
    quotes = market["quotes"]
    trades = []
    due = state["last_rebalance"] is None or (
        decision_time - pd.Timestamp(state["last_rebalance"])
        >= pd.Timedelta(days=REBALANCE_DAYS) - pd.Timedelta(hours=1)
    )
    if due or force_rebalance:
        trades = rebalance(
            state, symbols, signal, quotes, pairs, market["eur_usd"], now
        )
        append_csv(TRADES_PATH, trades)

    mid = {s: (b + a) / 2 for s, (b, a) in quotes.items()}
    holdings = {s: u * mid.get(s, np.nan) for s, u in state["units"].items()}
    value = state["cash_eur"] + sum(holdings.values())
    liquidation = state["cash_eur"] + sum(
        u * quotes[s][0] for s, u in state["units"].items() if s in quotes
    )
    basket = state["basket_cash_eur"] + sum(
        u * mid.get(s, np.nan) for s, u in state["basket_units"].items()
    )
    top = sorted(holdings.items(), key=lambda kv: -kv[1])
    append_csv(
        EQUITY_PATH,
        [
            {
                "time": now.isoformat(),
                "decision_time": decision_time.isoformat(),
                "value_eur": value,
                "liquidation_eur": liquidation,
                "cash_eur": state["cash_eur"],
                "basket_eur": basket,
                "holdings": len(holdings),
                "rebalanced": bool(trades),
                "trades": len(trades),
                "fees_eur": sum(t["fee_eur"] for t in trades),
                "tradable_coins": n_tradable,
                "stale": ";".join(panel["stale_symbols"]),
                "weights": ";".join(f"{s}:{v / value:.3f}" for s, v in top),
            }
        ],
    )
    state["last_step"] = decision_time.isoformat()
    save_state(state)

    ranked = np.argsort(-np.nan_to_num(signal["signal"], nan=-np.inf))
    print(
        f"{decision_time:%Y-%m-%d %H:%M} UTC | value €{value:,.2f} (basket "
        f"€{basket:,.2f}) | {len(holdings)} holdings | "
        f"{'rebalanced: ' + str(len(trades)) + ' trades' if trades else 'no rebalance'}"
    )
    print(
        "  top signal: "
        + ", ".join(
            f"{symbols[j][:-4]} {signal['signal'][j]:+.2f}"
            for j in ranked[:8]
            if signal["eligible"][j]
        )
    )
    if panel["stale_symbols"]:
        print(f"  stale data (excluded today): {panel['stale_symbols']}")


def report() -> None:
    state = load_state()
    equity = pd.read_csv(EQUITY_PATH, parse_dates=["time"])
    trades = pd.read_csv(TRADES_PATH) if TRADES_PATH.exists() else pd.DataFrame()
    capital = state["capital_eur"]
    last = equity.iloc[-1]
    days = (equity["time"].iloc[-1] - pd.Timestamp(state["created"])).days
    portfolio = np.log(last["value_eur"] / capital)
    basket = np.log(last["basket_eur"] / capital)
    print(
        f"Since {state['created'][:10]} ({days} days, {len(equity)} steps, "
        f"{int(equity['rebalanced'].sum())} rebalances):"
    )
    print(
        f"  portfolio €{last['value_eur']:,.2f} ({portfolio:+.2%} log), liquidation "
        f"€{last['liquidation_eur']:,.2f}"
    )
    print(f"  basket    €{last['basket_eur']:,.2f} ({basket:+.2%} log)")
    print(f"  excess    {portfolio - basket:+.2%} log")
    if len(trades):
        print(
            f"  traded €{trades['value_eur'].sum():,.0f}: fees €{trades['fee_eur'].sum():,.2f}, "
            f"half spreads €{trades['half_spread_eur'].sum():,.2f}"
        )
    print(f"  holdings: {last['weights']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fit", help="Fit and save the frozen model")
    init_parser = commands.add_parser("init", help="Create the paper account")
    init_parser.add_argument("--capital-eur", type=float, default=10_000)
    step_parser = commands.add_parser("step", help="Daily update (and rebalance)")
    step_parser.add_argument(
        "--force-rebalance",
        action="store_true",
        help="Rebalance now even if the week is not over",
    )
    commands.add_parser("report", help="Performance since the start")
    args = parser.parse_args()
    if args.command == "fit":
        fit()
    elif args.command == "init":
        init(args.capital_eur)
    elif args.command == "step":
        step(args.force_rebalance)
    else:
        report()


if __name__ == "__main__":
    main()

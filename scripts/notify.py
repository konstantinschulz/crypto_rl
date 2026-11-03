"""Telegram messages about paper trading: a daily summary and failure alerts.

The bot's token sits in ~/.config/crypto_rl/telegram.env as API_KEY=...
(created with @BotFather). The chat to write to is found once from the
bot's incoming messages (send the bot any message first) and saved to the
same file as CHAT_ID=..., because Telegram keeps those messages only a day.

Standard library only, so the failure alert still works when the trading
environment is broken.

Usage (the systemd units run the first two):
    python scripts/notify.py daily               # after each step
    python scripts/notify.py failure UNIT        # OnFailure: log tail of UNIT
    python scripts/notify.py test
    python scripts/notify.py daily --print       # show, don't send
"""

import argparse
import csv
import html
import json
import math
import subprocess
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
PAPER_DIR = repo_root / "data" / "paper"
CONFIG_PATH = Path.home() / ".config" / "crypto_rl" / "telegram.env"
API_URL = "https://api.telegram.org/bot{token}/{method}"
MAX_MESSAGE = 4096  # Telegram's limit, in characters


# --- Telegram --------------------------------------------------------------


def read_config() -> dict:
    """KEY=VALUE lines (comments, `export` and quotes allowed)."""
    if not CONFIG_PATH.exists():
        raise SystemExit(f"{CONFIG_PATH} missing; put API_KEY=<bot token> there")
    config = {}
    for line in CONFIG_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        config[key.strip()] = value.strip().strip("'\"")
    if not config.get("API_KEY"):
        raise SystemExit(f"no API_KEY in {CONFIG_PATH}")
    return config


def call(token: str, method: str, **params) -> dict:
    """One Bot API call, retried on network errors and server errors."""
    url = API_URL.format(token=token, method=method)
    data = urllib.parse.urlencode(params).encode()
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, data=data, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code < 500 or attempt == 2:
                # The body says why (bad token, chat not found, ...)
                detail = error.read().decode(errors="replace")
                raise SystemExit(f"Telegram {method}: HTTP {error.code} {detail}")
        except urllib.error.URLError:
            if attempt == 2:
                raise
        time.sleep(10 * (attempt + 1))
    raise AssertionError("unreachable")


def chat_id(config: dict) -> str:
    """The saved chat, or the latest private chat that wrote to the bot."""
    if config.get("CHAT_ID"):
        return config["CHAT_ID"]
    updates = call(config["API_KEY"], "getUpdates")["result"]
    chats = [
        u["message"]["chat"]
        for u in updates
        if "message" in u and u["message"]["chat"]["type"] == "private"
    ]
    if not chats:
        raise SystemExit(
            "No chat found: send your bot any message in Telegram, then retry "
            "(the bot sees messages of the last 24 hours)"
        )
    found = str(chats[-1]["id"])
    with open(CONFIG_PATH, "a") as f:
        f.write(f"\nCHAT_ID={found}\n")
    print(f"Saved CHAT_ID={found} ({chats[-1].get('first_name', '')}) to {CONFIG_PATH}")
    return found


def send(text: str) -> None:
    """Send an HTML-formatted message (cut to Telegram's limit)."""
    config = read_config()
    if len(text) > MAX_MESSAGE:
        text = text[: MAX_MESSAGE - 20].rsplit("\n", 1)[0] + "\n…"
    call(
        config["API_KEY"],
        "sendMessage",
        chat_id=chat_id(config),
        text=text,
        parse_mode="HTML",
        disable_web_page_preview="true",
    )


# --- messages --------------------------------------------------------------


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def pct(value: float) -> str:
    return f"{value:+.2%}".replace("-", "−")


def coin(symbol: str) -> str:
    return symbol.removesuffix("USDT")


def significant(x: float, digits: int = 4, min_decimals: int = 0) -> str:
    """Fixed-point with about ``digits`` significant digits (0.000002783)."""
    if x == 0 or not math.isfinite(x):
        return f"{x:.{min_decimals}f}"
    decimals = max(min_decimals, digits - 1 - math.floor(math.log10(abs(x))))
    return f"{x:,.{decimals}f}"


def quantity(units: float) -> str:
    """Coin units, shortened for meme coins (222.8M BONK)."""
    if units >= 1e6:
        return f"{units / 1e6:.1f}M"
    if units >= 1e4:
        return f"{units / 1e3:.1f}k"
    return significant(units)


def table(header: list[str], rows: list[list[str]]) -> str:
    """A monospaced table: the first column left-aligned, the rest right."""
    widths = [max(len(r[k]) for r in [header, *rows]) for k in range(len(header))]
    lines = [
        "  ".join(
            cell.ljust(widths[k]) if k == 0 else cell.rjust(widths[k])
            for k, cell in enumerate(row)
        )
        for row in [header, *rows]
    ]
    return f"<pre>{html.escape(chr(10).join(lines))}</pre>"


def cost_basis(trades: list[dict]) -> dict[str, float]:
    """EUR paid for the units still held, fees included (average cost: a
    sale removes its share of the cost)."""
    units, cost = {}, {}
    for t in trades:
        symbol, quantity_ = t["symbol"], float(t["units"])
        if t["side"] == "buy":
            units[symbol] = units.get(symbol, 0.0) + quantity_
            cost[symbol] = (
                cost.get(symbol, 0.0) + float(t["value_eur"]) + float(t["fee_eur"])
            )
        elif units.get(symbol, 0.0) > 0:
            kept = max(0.0, 1 - quantity_ / units[symbol])
            cost[symbol] *= kept
            units[symbol] -= quantity_
    return cost


def holdings_table(state: dict, last: dict, all_trades: list[dict]) -> str:
    """Each holding: units × price = EUR, and its gain since bought."""
    prices = state.get("prices_eur")
    if not prices:
        # Before the step saved its prices: back out of the rounded weights
        value = float(last["value_eur"])
        weights = dict(w.split(":") for w in last["weights"].split(";") if w)
        prices = {
            s: float(weights[s]) * value / u
            for s, u in state["units"].items()
            if s in weights and u > 0
        }
    cost = cost_basis(all_trades)
    rows = []
    for symbol, units in state["units"].items():
        price = prices.get(symbol, math.nan)
        rows.append((units * price, symbol, units, price))
    rows.sort(key=lambda r: -r[0] if math.isfinite(r[0]) else math.inf)
    body = [
        [
            coin(symbol),
            quantity(units),
            significant(price, min_decimals=2),
            f"{value:,.2f}",
            pct(value / cost[symbol] - 1) if cost.get(symbol) else "",
        ]
        for value, symbol, units, price in rows
    ]
    body.append(["cash", "", "", f"{state['cash_eur']:,.2f}", ""])
    note = (
        "" if state.get("prices_eur") else "\n(prices estimated from rounded weights)"
    )
    return table(["coin", "units", "price €", "value €", "gain"], body) + note


def trades_table(trades: list[dict]) -> str:
    """The step's trades: units × price = EUR, sells first."""
    body = [
        [
            f"{t['side']} {coin(t['symbol'])}",
            quantity(float(t["units"])),
            significant(float(t["price_eur"]), min_decimals=2),
            f"{float(t['value_eur']):,.2f}",
            f"{float(t['fee_eur']):,.2f}",
        ]
        for t in sorted(trades, key=lambda t: (t["side"] != "sell", t["symbol"]))
    ]
    return table(["trade", "units", "price €", "value €", "fee €"], body)


def daily_message() -> str:
    """Status after the latest step: value vs basket, holdings, trades."""
    state = json.loads((PAPER_DIR / "state.json").read_text())
    equity = read_csv(PAPER_DIR / "equity.csv")
    if not equity:
        return "<b>Paper trading</b>\nNo step has run yet."
    last = equity[-1]
    now = datetime.fromisoformat(last["time"])
    decision = datetime.fromisoformat(last["decision_time"])
    value, basket = float(last["value_eur"]), float(last["basket_eur"])
    capital = float(state["capital_eur"])
    created = datetime.fromisoformat(state["created"])

    lines = [
        f"<b>Paper trading · {decision:%a %Y-%m-%d %H:%M} UTC</b>",
        (
            f"Value <b>€{value:,.2f}</b> (liquidation "
            f"€{float(last['liquidation_eur']):,.2f}, cash €{float(last['cash_eur']):,.2f})"
        ),
        f"Basket €{basket:,.2f}",
        "",
    ]

    def period(label, start_value, start_basket):
        portfolio = value / start_value - 1
        benchmark = basket / start_basket - 1
        return (
            f"{label}: portfolio {pct(portfolio)}, basket {pct(benchmark)}, "
            f"<b>excess {pct(portfolio - benchmark)}</b>"
        )

    days = (now - created).days
    lines.append(period(f"Since start ({days} d)", capital, capital))
    # The step at least 7 days before this one
    week_ago = [
        r
        for r in equity
        if datetime.fromisoformat(r["time"]) <= now - timedelta(days=6, hours=12)
    ]
    if week_ago:
        lines.append(
            period(
                "Last 7 d",
                float(week_ago[-1]["value_eur"]),
                float(week_ago[-1]["basket_eur"]),
            )
        )

    all_trades = read_csv(PAPER_DIR / "trades.csv")
    trades = [t for t in all_trades if t["time"] == last["time"]]
    lines.append("")
    if trades:
        fees = sum(float(t["fee_eur"]) for t in trades)
        spread = sum(float(t["half_spread_eur"]) for t in trades)
        traded = sum(float(t["value_eur"]) for t in trades)
        lines.append(
            f"<b>Rebalanced:</b> {len(trades)} trades, €{traded:,.2f} traded, "
            f"fees €{fees:,.2f}, spread €{spread:,.2f}"
        )
        lines.append(trades_table(trades))
    elif state.get("last_rebalance"):
        # paper_trade.step rebalances at the first daily decision at least
        # 7 days minus 1 hour after the last rebalance
        due = datetime.fromisoformat(state["last_rebalance"]) + timedelta(
            days=7, hours=-1
        )
        next_rebalance = decision + timedelta(
            days=max(1, math.ceil((due - decision) / timedelta(days=1)))
        )
        lines.append(f"No rebalance; next on {next_rebalance:%a %Y-%m-%d}")

    lines.append(
        f"<b>Holdings ({len(state['units'])})</b> at mid prices; gain since "
        "bought, fees included:"
    )
    lines.append(holdings_table(state, last, all_trades))

    if last["stale"]:
        stale = ", ".join(coin(s) for s in last["stale"].split(";"))
        lines.append(f"\n⚠️ Stale data, excluded today: {html.escape(stale)}")
    if decision.date() < datetime.now(decision.tzinfo).date() - timedelta(days=1):
        lines.append(
            f"\n⚠️ The latest step is from {decision:%Y-%m-%d}: is the timer running?"
        )
    return "\n".join(lines)


def failure_message(unit: str) -> str:
    """The unit failed: say so, with the tail of its log."""
    try:
        log = subprocess.run(
            ["journalctl", "--user", "-u", unit, "-n", "30", "--no-pager", "-o", "cat"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        log = f"(no log: {error})"
    log = log.strip()[-3000:] or "(empty log)"
    return (
        f"🚨 <b>{html.escape(unit)} failed</b> "
        f"({datetime.now().astimezone():%Y-%m-%d %H:%M %Z})\n"
        f"<pre>{html.escape(log)}</pre>"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    daily = commands.add_parser("daily", help="Summary after the latest step")
    daily.add_argument("--print", action="store_true", help="Print, don't send")
    failure = commands.add_parser("failure", help="Alert that a unit failed")
    failure.add_argument("unit")
    commands.add_parser("test", help="Send a test message")
    args = parser.parse_args()

    if args.command == "daily":
        message = daily_message()
        if args.print:
            print(message)
            return
    elif args.command == "failure":
        message = failure_message(args.unit)
    else:
        message = "✅ Paper-trading notifications work."
    send(message)


if __name__ == "__main__":
    main()

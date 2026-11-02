#!/usr/bin/env bash
# Set up the daily paper-trading step on a small VM (tested for 1 GB RAM;
# Ubuntu 24.04, whose Python 3.12 matches the research environment).
#
# The step needs the frozen model and Kraken's pair table, which live under
# the git-ignored data/. Pack them on the research machine and put the file
# where the VM can download it:
#
#   tar -czf paper_bundle.tar.gz data/paper/model.npz data/kraken/pairs.csv
#
# Optionally add data/live (~65 MB) for a warm start; without it the first
# step fills the cache from Binance itself (~45 min). Then, on the VM:
#
#   sudo apt install -y python3-venv git
#   git clone https://github.com/konstantinschulz/crypto_rl.git && cd crypto_rl
#   scripts/vm/setup.sh https://.../paper_bundle.tar.gz 100
#
# Arguments: the bundle's URL (or a local path) and the paper capital in EUR.
# Logs: journalctl --user -u paper-trade; status: .venv/bin/python
# scripts/paper_trade.py report
set -euo pipefail

bundle=${1:?usage: scripts/vm/setup.sh BUNDLE_URL_OR_PATH [CAPITAL_EUR]}
capital=${2:-100}
repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo"

python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements-paper.txt

if [[ -f "$bundle" ]]; then
    tar -xzf "$bundle"
else
    curl -fL "$bundle" | tar -xz
fi
test -f data/paper/model.npz && test -f data/kraken/pairs.csv

if [[ ! -f data/paper/state.json ]]; then
    .venv/bin/python scripts/paper_trade.py init --capital-eur "$capital"
fi

# The units point at the research machine's checkout and conda env; rewrite
# them for this checkout and its venv
units=~/.config/systemd/user
mkdir -p "$units"
for unit in paper-trade.service paper-trade.timer; do
    sed -e "s|%h/dev/crypto_rl/.conda/bin/python|$repo/.venv/bin/python|g" \
        -e "s|%h/dev/crypto_rl|$repo|g" \
        "scripts/systemd/$unit" >"$units/$unit"
done
# Run user units without a login session
sudo loginctl enable-linger "$USER"
systemctl --user daemon-reload
systemctl --user enable --now paper-trade.timer
systemctl --user list-timers paper-trade.timer

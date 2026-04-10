# RL Cryptocurrency Trading System

A minimal but production-grade reinforcement learning system for learning optimal trading strategies on 1-minute cryptocurrency data.

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Training with Dashboard
```bash
# Terminal 1: Start the Streamlit dashboard
streamlit run streamlit_dashboard.py --server.port 8766

# Terminal 2: Run training (dashboard auto-detects new runs)
python rl_trader.py --days 7 --train-steps 50000 --mode train
```

Open `http://localhost:8766` to monitor training in real-time.

### Basic Commands
```bash
# Simple training
python rl_trader.py --days 7 --train-steps 50000 --mode train

# Backtest (train on 80%, test on 20%)
python rl_trader.py --days 14 --mode backtest

# 3-Way backtest (train 60%, validate 20%, test 20%)
python rl_trader.py --days 14 --mode backtest_3way --train-steps 50000

# Faster 3-way run (fewer validation passes during training)
python rl_trader.py --days 14 --mode backtest_3way --train-steps 50000 --eval-freq 25000

# Force GPU (if CUDA is available)
python rl_trader.py --days 14 --mode backtest_3way --dashboard --device cuda

# Evaluate existing model
python rl_trader.py --mode eval --model models/final_model.zip --days 3
```

## 📚 Documentation

- **[RL_TRADING_README.md](RL_TRADING_README.md)** — Comprehensive guide
  - Architecture overview
  - Reward function details
  - Training pipeline
  - Usage examples
  - Extending the system
  - Troubleshooting

- **[DASHBOARD_GUIDE.md](DASHBOARD_GUIDE.md)** — Streamlit dashboard deep dive
  - How the dashboard works
  - Interface walkthrough
  - Metrics interpretation
  - Troubleshooting dashboard issues
  - Advanced usage patterns

- **[MEMORY_EFFICIENT_QUICKSTART.md](MEMORY_EFFICIENT_QUICKSTART.md)** — Quick start for memory-limited systems
  - Safe configuration presets
  - Memory optimization tips
  - OOM error solutions

- **[MEMORY_OPTIMIZATION_GUIDE.md](MEMORY_OPTIMIZATION_GUIDE.md)** — Deep dive into optimizations
  - Detailed technical explanations
  - Performance trade-offs
  - Tuning recommendations

- **[CHANGES.md](CHANGES.md)** — Recent updates and improvements
  - Memory optimization changes
  - Performance improvements
  - System updates

## 🎯 Key Features

### Streamlit Dashboard
- **Auto-Detection**: Automatically discovers new training/eval runs
- **Real-Time Monitoring**: Updates every 2 seconds
- **Run Selection**: Switch between runs via dropdown
- **Rich Metrics**: Portfolio value, PnL, rewards, losses, win rate
- **Zero Reload**: No page refresh needed once opened

### Compute Efficiency (new defaults)
- **GPU Auto-Select**: `--device auto` is now the default (`cuda` if available, else `cpu`)
- **Cheaper 3-Way Validation**: `backtest_3way` now validates less frequently during training by default
- **Lower Dashboard I/O Cost**: run index writes are throttled while still auto-detecting new runs
- **Lower Callback Overhead**: less frequent forced garbage collection during PPO training

### Memory Optimized
- **70-80% RAM Reduction** from original implementation
- Runs on 4-8GB systems comfortably
- Configurable data loading and model size

### Flexible Training
- **Multiple Modes**: train, backtest, eval, backtest_3way
- **PPO Agent**: Industry-standard policy gradient RL algorithm
- **Realistic Constraints**: Transaction costs, position limits, stop-loss

## 🏗️ Architecture

### Core Components
1. **rl_trading_env.py** — Gymnasium-compatible trading environment
2. **rl_trader.py** — Training and evaluation wrapper
3. **streamlit_dashboard.py** — Real-time monitoring dashboard

### Data Flow
```
rl_trader.py (training/eval)
     ↓
RLDashboardWriter (writes every 50-100 steps)
     ↓
rl_dashboard_runs/<run_id>.json (per-run state)
rl_dashboard_index.json (run index)
     ↓
streamlit_dashboard.py (reads every 2 seconds)
     ↓
http://localhost:8766 (live web interface)
```

## 📊 Dashboard Metrics

**Key Performance Indicators:**
- Step, Trades, Portfolio Value, Realized PnL, Win Rate

**Charts:**
- Portfolio vs PnL over time
- Train/Dev/Test rewards per step
- Train/Policy/Value losses

**Sidebar:**
- Mode (train, backtest, eval, backtest_3way)
- Status (initializing, running, completed)
- Started/Progress timestamps

## 🔧 Configuration

### Memory-Limited System (4-8 GB)
```bash
python rl_trader.py --days 7 --max-symbols 3 --train-steps 50000
```

### Standard System (8+ GB)
```bash
python rl_trader.py --days 14 --max-symbols 10 --train-steps 100000
```

### Custom Config
```bash
python rl_trader.py \
  --days 7 \
  --max-symbols 5 \
  --train-steps 50000 \
  --mode backtest_3way
```

### Speed-Oriented Config
```bash
# Keep dashboard enabled, but reduce validation overhead during training
python rl_trader.py \
  --days 14 \
  --mode backtest_3way \
  --dashboard \
  --eval-freq 25000

# Prefer GPU explicitly (if available)
python rl_trader.py \
  --days 14 \
  --mode backtest_3way \
  --dashboard \
  --device cuda
```

## 🐛 Troubleshooting

### Dashboard not showing runs
- Run at least one training session to create `rl_dashboard_index.json`
- Verify dashboard can read: `cat rl_dashboard_index.json`
- Hard-refresh browser (Ctrl+Shift+R) to clear Streamlit cache

### Out of Memory errors
- Use fewer days: `--days 3`
- Use fewer symbols: `--max-symbols 1`
- Run backtest_3way which manages memory automatically
- See [MEMORY_EFFICIENT_QUICKSTART.md](MEMORY_EFFICIENT_QUICKSTART.md) for detailed guidance

### Training not improving
- Check dashboard rewards are increasing over time
- Try longer training: `--train-steps 100000`
- Adjust learning rate or other hyperparameters
- View TensorBoard logs: `tensorboard --logdir=./tb_logs/`

## 📦 System Requirements

- **Python**: 3.8+
- **RAM**: 4 GB minimum (8+ GB recommended)
- **Storage**: ~500 MB for data and models
- **Dependencies**: See requirements.txt

## 📖 Next Steps

1. **Start the Dashboard** (recommended):
   ```bash
   streamlit run streamlit_dashboard.py --server.port 8766
   ```

2. **Run Training**:
   ```bash
   python rl_trader.py --days 7 --train-steps 50000 --mode train
   ```

3. **Monitor in Browser**:
   - Open http://localhost:8766
   - Select your run from the dropdown
   - Watch metrics update in real-time

4. **Read Full Documentation**:
   - See [RL_TRADING_README.md](RL_TRADING_README.md) for comprehensive guide

## 🔗 References

- [Gymnasium](https://gymnasium.farama.org/) — RL environment standard
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — RL algorithms
- [Streamlit](https://streamlit.io/) — Web dashboard framework

## 📝 License & Citation

This is a minimal, educational implementation for learning RL + trading together.

---

**Created**: March 2026  
**Status**: Production-ready, memory-optimized  
**Python**: 3.8+



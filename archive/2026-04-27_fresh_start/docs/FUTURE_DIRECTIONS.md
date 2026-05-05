# Future Directions for RL Trading Agent

Based on the experiments conducted so far, we have reached a ceiling with the current "lightweight" setup. While the Recurrent PPO (LSTM) model successfully learned to trade the training/validation data (producing positive Val returns), it fails to generalize to unseen Test data. To achieve true profitability, we need to fundamentally scale up the approach.

Here is a comprehensive overview of approaches we can take to make the model more profitable, categorized by the core pillars of RL:

### 1. Massive Scale-Up of Compute & Training Horizon (The Biggest Win)
Right now, the models are training for **15,000 steps** on the CPU in about 2 minutes. Deep Reinforcement Learning typically requires **millions** of steps to stabilize and generalize.
* **Scale Training Steps**: Increase `--train-steps` from 15k to **500k, 1M, or 5M**.
* **Utilize GPU**: Switch to `--device cuda`. VRAM usage for 1D time-series data is actually very low. We can afford to push this hard.
* **Parallel Environments**: Use `SubprocVecEnv` to run 8 or 16 parallel environment instances simultaneously during training. This drastically improves the diversity of data the PPO agent sees in a single batch, stabilizing policy gradients.

### 2. Data Expansion & Regime Diversity (The "Experience" Factor)
90 days of data implies the model only sees 1 or 2 macro market movements. 
* **Years of History**: Train on 1 to 2 years of data to force the model to learn through violent bull runs, crypto winters, and sideways chop.
* **Broader Coin Universe**: Expand from 5 symbols to 20+. This prevents the model from overfitting to the quirks of $BNB or $BTC and forces it to learn universal price action dynamics.
* **Multi-Timeframe Context**: Instead of just 15m bars, feed the model multiple granularities simultaneously (e.g., current 15m bar, current 1h bar, and current 4h bar). Institutional bots look at top-down trends.
* **Alpha Features**: Add higher-tier crypto-specific features: Orderbook imbalance approximations, funding rates, or a "BTC proxy" feature attached to altcoins so it knows the macro market direction.

### 3. Model Architecture Details (The "Brain")
Our current `[384, 192, 96]` MLP with a 1-layer, 128-cell LSTM is an excellent start, but we can go deeper.
* **Deep Recurrence**: Increase the LSTM capacity to `lstm_hidden_size=256` and `lstm_layers=2` or `3` to handle longer-term dependencies over the lookback window.
* **Continuous Action Space**: Instead of discrete sets (0: Hold, 1: Buy, 2: Sell), switch to continuous actions `[-1.0, 1.0]`. This allows the model to scale its position size.
* **Custom Feature Extractors (CNNs)**: Prepend a 1D Convolutional Neural Network (CNN) before the LSTM. CNNs are exceptional at detecting local patterns.

### 4. Advanced Objective & Reward Design (The "Incentive")
* **Risk-Adjusted Rewards (Sharpe/Sortino)**: Instead of rewarding pure equity delta, calculate the Sharpe or Sortino ratio over a rolling 24-hour window and use *that* as the reward.
* **Dynamic / Curriculum Constraints**: Start the training with no penalties for high trade rates so the agent explores, and slowly "ramp up" the turnover/trade-rate penalties automatically as training progresses.
* **Asymmetric Drawdown Penalties**: Apply an exponential penalty if the portfolio drops below the starting capital. 

### 5. Meta-Optimization
* **Optuna Hyperparameter Sweeps**: Set up Optuna to run 100 variations over a weekend, sweeping `learning_rate`, `batch_size`, `gamma`, `ent_coef`, and `clip_range`.
* **Ensembling**: Train 5 different models on different random seeds and write an inference script that only executes a real trade if 3 out of 5 models agree (Majority Voting).

---

### Recommended Next Move
Execute Category 1 and Category 2.
We will create a new configuration: **1-Year Context, Millions of Steps, GPU Enabled, Parallel Environments**.


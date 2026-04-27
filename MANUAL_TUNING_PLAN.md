# Manual Hyperparameter Tuning Plan

## Objective
Before doing massive scale-ups or expensive, unguided Optuna sweeps, we want to establish a strong baseline using educated guesses. The goal is to quickly iterate on core hyperparameter sets that theoretically stabilize training and improve generalizability on unseen data.

## The Strategy: Targeted Parameter Manipulation

### Step 1: Baseline Stabilization (Experiment 1)
We will run a medium-sized test (e.g., 30 days of data, 30k train steps) using a set of parameters known to improve PPO stability:
* **Higher Batch Size & Trajectory**: Increase `batch_size` to 128 (from typical 16/64) and `n_steps` to 1024. This gives the agent a much broader and less noisy gradient update.
* **Lower Learning Rate**: Decrease `learning_rate` to `1e-4` (from `3e-4`) to prevent aggressive policy shifts that destroy learned representations.
* **Targeted Penalties**: Introduce mild `trade_execution_penalty` to reduce churn, and an `inactivity_penalty` to ensure the agent doesn't just learn to "hold" indefinitely.

### Step 2: Observation & Adjustment
Once Experiment 1 completes, we will observe the `Val Return` and `Test Return`.
* **If it overtrades**: We increase the `trade_execution_penalty`.
* **If it collapses to zero trades**: We reduce the `inactivity_penalty` or increase the `trade_action_bonus`.
* **If Val is high but Test is negative (Overfitting)**: We increase `batch_size` further, or add an `ent_coef` (entropy) if supported, or increase the dataset size.

### Step 3: Progressive Scaling
Only once we hit a locally profitable configuration on the medium 30-day/30k step scale, will we:
1. Scale `days` up to 90 or 365.
2. Scale `train-steps` up to 100k+.
3. Optionally employ Optuna in a very narrow search space around our winning parameters.

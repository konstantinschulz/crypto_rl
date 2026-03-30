# Model Improvement Experiments - March 30, 2026

## Current Baseline (14 days, 3 symbols)
- Model: PPO with small network [32, 16]
- Training: 50,000 timesteps
- Config: initial_cash=100, max_positions=3
- Hardware: CPU
- Result: Not profitable

## Hypothesis & Experiment Plan

### Experiments to Run
1. **Exp 1**: Increase model capacity (larger network)
2. **Exp 2**: Increase training timesteps
3. **Exp 3**: Use more data (21 days instead of 14)
4. **Exp 4**: Increase max positions (more diversification)
5. **Exp 5**: Better reward shaping (adjust penalties/bonuses)
6. **Exp 6**: Longer position hold time
7. **Exp 7**: Combination of above successful changes

### Constraints
- Avoid memory overflow (watch RAM usage)
- Avoid CPU overload (keep responsive)
- Can use GPU if needed
- Document all results

## Results Log

(To be populated as experiments run)



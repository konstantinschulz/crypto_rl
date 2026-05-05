#!/usr/bin/env python3
"""Quick test script to verify RL environment loads and creates valid observations."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Test imports
try:
    from rl_trading_env import CryptoTradingEnv, TradingConfig
    print("✓ Successfully imported CryptoTradingEnv and TradingConfig")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test loading data
try:
    from rl_trading_env import load_training_data
    print("✓ Successfully imported load_training_data")
    
    # Try loading 2 days of data with max 3 symbols
    data = load_training_data(
        'binance_spot_1m_last4y_single.parquet',
        num_days=2,
        symbols=None,
        max_symbols=3,
        use_cache=True,
    )
    print(f"✓ Loaded data: {len(data)} rows, {data['symbol'].nunique()} symbols")
    print(f"  Symbols: {sorted(data['symbol'].unique())}")
except Exception as e:
    print(f"✗ Failed to load data: {e}")
    sys.exit(1)

# Test environment creation
try:
    config = TradingConfig(
        inactivity_penalty=-5.0,
        trade_action_bonus=15.0,
    )
    print(f"✓ Created TradingConfig with inactivity_penalty={config.inactivity_penalty}")
    
    env = CryptoTradingEnv(data, config)
    print(f"✓ Created trading environment")
    print(f"  Price matrix shape: {env.price_matrix.shape}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
except Exception as e:
    print(f"✗ Failed to create environment: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test reset and step
try:
    obs, info = env.reset()
    print(f"✓ Environment reset successful")
    print(f"  Initial observation shape: {obs.shape}")
    print(f"  Initial observation (first 5): {obs[:5]}")
    
    # Take a few steps
    print("\n  Testing environment steps:")
    for step in range(5):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        portfolio = info.get('portfolio_value', 0)
        trades = info.get('trades', 0)
        print(f"    Step {step+1}: action={action}, reward={reward:.2f}, portfolio=${portfolio:.2f}, trades={trades}")
        if done:
            break
    
    print(f"✓ Environment stepping works correctly")
except Exception as e:
    print(f"✗ Failed during environment step: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All tests passed!")


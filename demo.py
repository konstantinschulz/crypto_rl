#!/usr/bin/env python
"""
Minimal Demo: RL Trading Environment

Quick test to verify everything works. Run this first before full training.
"""

import sys

from rl_trading_env import CryptoTradingEnv, TradingConfig, load_training_data


def demo_environment():
    """Quick environment demo with random actions"""
    print("="*70)
    print("RL TRADING ENVIRONMENT DEMO")
    print("="*70)
    
    # Load small dataset
    print("\n1. Loading data...")
    try:
        data = load_training_data(
            'binance_spot_1m_last4y_single.parquet',
            num_days=2,
            max_symbols=8,
            use_cache=True,
        )
        print(f"   ✓ Loaded {len(data):,} rows from {data['symbol'].nunique()} symbols")
        print(f"   Date range: {data['open_time'].min()} to {data['open_time'].max()}")
    except FileNotFoundError:
        print("   ✗ Error: parquet file not found")
        print("   Make sure binance_spot_1m_last4y_single.parquet is in current directory")
        return False
    
    # Create environment
    print("\n2. Creating environment...")
    config = TradingConfig(
        initial_cash=100.0,
        max_positions=5,
        max_budget_per_trade=20.0,
    )
    
    env = CryptoTradingEnv(data, config)
    print(f"   ✓ Environment initialized")
    print(f"   - Action space: {env.action_space}")
    print(f"   - Observation space: {env.observation_space}")
    print(f"   - Symbols: {len(env.symbols)}")
    
    # Reset and get initial state
    print("\n3. Resetting environment...")
    obs, info = env.reset()
    print(f"   ✓ Observation shape: {obs.shape}")
    print(f"   - Initial cash: ${env.cash:.2f}")
    print(f"   - Initial positions: {len(env.positions)}")
    
    # Run 100 random steps
    print("\n4. Running 100 random trading steps...")
    print("   (taking action every 10 steps for display)")
    print()
    print("   Step | Portfolio $ | Positions | Cash $  | Reward")
    print("   -----+--------------+-----------+---------+--------")
    
    total_reward = 0
    log_interval = 10
    max_steps = 100
    
    for step in range(max_steps):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        # Log every N steps
        if step % log_interval == 0:
            portfolio = info.get('portfolio_value', env.cash)
            print(f"   {step:4d} | ${portfolio:11.2f} | {len(env.positions):9d} | "
                  f"${env.cash:7.2f} | {reward:7.3f}")
        
        if done:
            print(f"   ... (episode ended at step {step})")
            break
    
    final_portfolio = info.get('portfolio_value', env.cash)
    final_return = (final_portfolio - config.initial_cash) / config.initial_cash * 100
    
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"Initial Portfolio:    ${config.initial_cash:,.2f}")
    print(f"Final Portfolio:       ${final_portfolio:,.2f}")
    print(f"Return:               {final_return:+.2f}%")
    print(f"Total Reward:         {total_reward:+.2f}")
    print(f"Final Positions:      {len(env.positions)}")
    print(f"Final Cash:           ${env.cash:,.2f}")
    print(f"Steps Executed:       {env.current_step}")
    print("="*70)
    
    return True


def demo_agent():
    """Demo with a trained agent (stub for future integration)"""
    print("\n" + "="*70)
    print("AGENT TRAINING DEMO")
    print("="*70)
    print("\nTo train an agent:")
    print("  python rl_trader.py --days 7 --train-steps 50000 --mode train")
    print("\nTo evaluate an existing model:")
    print("  python rl_trader.py --mode eval --model models/final_model.zip --days 3")
    print("\nTo backtest (train/test split):")
    print("  python rl_trader.py --days 14 --mode backtest")
    print("="*70)


def main():
    """Run all demos"""
    try:
        success = demo_environment()
        if success:
            demo_agent()
            print("\n✓ System is ready! You can now run full training.")
            return 0
        else:
            print("\n✗ Demo failed. Check errors above.")
            return 1
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 1
    
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

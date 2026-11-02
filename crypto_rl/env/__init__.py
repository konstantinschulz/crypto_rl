__all__ = ["MinimalCryptoEnv"]


def __getattr__(name):
    # Lazy, so that importing crypto_rl.env.data_utils (the feature code used
    # by paper trading) doesn't pull in gymnasium and optuna
    if name == "MinimalCryptoEnv":
        from crypto_rl.env.minimal_env import MinimalCryptoEnv

        return MinimalCryptoEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

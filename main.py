from crypto_rl.cli import build_parser
from crypto_rl.config import RLConfig
from crypto_rl.experiment import run_experiment


def main():
    config: RLConfig = build_parser()
    run_experiment(config)


if __name__ == "__main__":
    main()

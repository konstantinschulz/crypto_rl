from crypto_rl.cli import build_parser
from crypto_rl.experiment import run_experiment


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

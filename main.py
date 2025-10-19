from __future__ import annotations

import argparse

from src.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Latent conformal experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/gmm_em.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()
    df = run_experiment(args.config)
    results_path = df.attrs.get("results_path", "experiments/results")
    print(f"Completed {len(df)} runs. Results written to {results_path}")


if __name__ == "__main__":
    main()

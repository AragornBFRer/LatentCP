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
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Number of trials (random seeds) to run; overrides config seeds when provided",
    )
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="Optional path to write results CSV (overrides config io.results_csv)",
    )
    args = parser.parse_args()
    seeds = None
    if args.trials is not None:
        if args.trials <= 0:
            parser.error("--trials must be positive when specified")
        seeds = range(1, args.trials + 1)
    df = run_experiment(
        args.config,
        seeds=seeds,
        results_path_override=args.results,
    )
    results_path = df.attrs.get("results_path", "experiments/results")
    print(f"Completed {len(df)} runs. Results written to {results_path}")


if __name__ == "__main__":
    main()

Run helpers for experiments
===========================

This folder contains helper scripts to run the experiments per-seed and to launch
many seeds in parallel from a POSIX shell.

Files
- scripts/run_single_seed.py: Python script that runs all run-configs for a single seed and writes a per-seed CSV to `experiments/results/gmm_em_results_seed_<seed>.csv`.
- run_multiple_seeds.sh: Bash script (in repo root) that runs multiple seeds in parallel and merges per-seed CSVs into a single `experiments/results/gmm_em_results.csv`.

Quick example (from repo root):

```bash
# run seeds 1..100 with 8 parallel jobs
./run_multiple_seeds.sh --seeds 1-100 --jobs 8 --out experiments/results/gmm_em_results.csv
```

```bash
# alternative cmd
chmod +x run_multiple_seeds.sh
./run_multiple_seeds.sh --seeds 1-100 --jobs 8 --out experiments/results/gmm_em_results.csv
```

Notes
- The Python runner imports the package under `src/` so run from the repository root so imports resolve.
- `run_multiple_seeds.sh` merges CSVs and performs a simple deduplication by logical run key.

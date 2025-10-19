#!/usr/bin/env bash
# Run a range of seeds in parallel and merge results into one CSV.
# Usage:
#   ./run_multiple_seeds.sh --config experiments/configs/gmm_em.yaml --seeds 1-100 --jobs 8 --out experiments/results/gmm_em_results.csv

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT="$SCRIPT_DIR"
PY_RUNNER="$REPO_ROOT/scripts/run_single_seed.py"

print_usage() {
  cat <<'USAGE'
Usage: run_multiple_seeds.sh [--config <yaml>] [--seeds <a-b|list>] [--jobs N] [--out <csv>]

Examples:
  ./run_multiple_seeds.sh --seeds 1-100 --jobs 8 --out experiments/results/gmm_em_results.csv
  ./run_multiple_seeds.sh --seeds 1,3,5,7 --jobs 4
USAGE
}

CONFIG="experiments/configs/gmm_em.yaml"
SEEDS="1-2"
JOBS=4
OUT="experiments/results/gmm_em_results.csv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"; shift 2;;
    --seeds)
      SEEDS="$2"; shift 2;;
    --jobs)
      JOBS="$2"; shift 2;;
    --out)
      OUT="$2"; shift 2;;
    -h|--help)
      print_usage; exit 0;;
    *)
      echo "Unknown arg: $1"; print_usage; exit 1;;
  esac
done

mkdir -p "$(dirname "$OUT")"

# parse seeds string
parse_seeds() {
  local s="$1"
  if [[ "$s" == *","* ]]; then
    IFS=',' read -r -a arr <<< "$s"
  elif [[ "$s" == *"-"* ]]; then
    IFS='-' read -r a b <<< "$s"
    arr=()
    for ((i=a;i<=b;i++)); do arr+=("$i"); done
  else
    arr=("$s")
  fi
  echo "${arr[@]}"
}

SEED_LIST=( $(parse_seeds "$SEEDS") )

echo "Running seeds: ${SEED_LIST[*]}"
echo "Using $JOBS parallel jobs"

# Run jobs (background) with a simple job slot semaphore
pids=()
for seed in "${SEED_LIST[@]}"; do
  python3 "$PY_RUNNER" --config "$CONFIG" --seed "$seed" --out-dir "$(dirname "$OUT")" &
  pids+=("$!")
  # control concurrency
  while [[ "${#pids[@]}" -ge $JOBS ]]; do
    # wait for first pid
    wait "${pids[0]}" || true
    # drop finished pids
    new=()
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then new+=("$pid"); fi
    done
    pids=("${new[@]}")
  done
done

# wait for remaining
for pid in "${pids[@]}"; do
  wait "$pid" || true
done

echo "All per-seed runs complete. Merging CSVs..."

TMP_OUT="${OUT}.tmp"
rm -f "$TMP_OUT"

# find per-seed CSVs
CSV_DIR=$(dirname "$OUT")
FILES=("$CSV_DIR"/gmm_em_results_seed_*.csv)

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No per-seed CSVs found in $CSV_DIR"; exit 1
fi

# Write header from the first file, then append the rest skipping headers
head -n1 "${FILES[0]}" > "$TMP_OUT"
for f in "${FILES[@]}"; do
  tail -n +2 "$f" >> "$TMP_OUT"
done

# Optionally deduplicate logical runs by key columns: seed,K,delta,beta_spread,use_x_in_em,em_iter,em_loglik
# We use awk to do a simple dedup; this performs string-level de-duplication of the chosen key
awk -v OFS="," 'NR==1 {print; next} { key = $15","$16","$17","$18","$19","$21","$22; if (!seen[key]++) print }' "$TMP_OUT" > "${OUT}.final"
mv "${OUT}.final" "$OUT"
rm -f "$TMP_OUT"

echo "Merged output saved to $OUT"

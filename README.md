Default configs live in `experiments/configs/gmm_em.yaml`.

## Running

```bash
# Linux / macOS
chmod +x run.sh
./run.sh

# running 100 trials
bash run.sh experiments/configs/gmm_em.yaml 100 experiments/plots_test experiments/results/gmm_em_results_test.csv

# Windows PowerShell
python main.py --config experiments/configs/gmm_em.yaml
```

## Visualizing results

After simulations finish, generate summary figures:

```bash
# Linux / macOS
python plot_results.py --config experiments/configs/gmm_em.yaml

# Windows PowerShell
python plot_results.py --config experiments/configs/gmm_em.yaml
```

Figures are stored under `experiments/plots/` with descriptive filenames such as `coverage_vs_delta.png`, `length_vs_delta.png`, and `imputation_metrics_vs_delta.png`.

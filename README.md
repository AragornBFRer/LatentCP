Default configs live in `experiments/configs/gmm_em.yaml`.

## Running the full sweep (100 seeds)

```bash
# Linux / macOS
chmod +x run.sh
./run.sh

# Windows PowerShell
python main.py --config experiments/configs/gmm_em.yaml
```

## Running a single trial (edit the config to limit seeds)

Create a lightweight override, for example `experiments/configs/single.yaml`:

```yaml
global:
	seeds:
		values: [1]
```

Then execute either helper:

```bash
# Linux / macOS
./run.sh experiments/configs/single.yaml

# Windows PowerShell
python main.py --config experiments/configs/single.yaml
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

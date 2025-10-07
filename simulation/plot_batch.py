from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOT_COLORS = {
    "SCP": "#1f77b4",
    "PCP": "#ff7f0e",
    "CQR": "#2ca02c",
    "QR": "#d62728",
    "CRR": "#9467bd",
}


def _ensure_axes(axes: Iterable, count: int):
    axes = list(axes) if isinstance(axes, Iterable) else [axes]
    if len(axes) < count:
        axes.extend([axes[-1]] * (count - len(axes)))
    return axes[:count]


def _method_order(methods: Iterable[str]) -> list[str]:
    preference = ["SCP", "PCP", "CQR", "QR", "CRR"]
    preferred = [m for m in preference if m in methods]
    remaining = sorted(set(methods) - set(preference))
    return preferred + remaining


def plot_coverage_and_efficiency(
    eval_df: pd.DataFrame,
    output_path: Path,
    title_note: Optional[str] = None,
    footer_note: Optional[str] = None,
    rotate_xticks: bool = False,
) -> None:
    tasks = sorted(eval_df["task"].unique())
    # Create subplots: 2 rows (coverage, efficiency) x len(tasks) columns
    fig, axes = plt.subplots(2, len(tasks), figsize=(6 * len(tasks), 10), 
                           gridspec_kw={'hspace': 0.3})
    if len(tasks) == 1:
        axes = axes.reshape(-1, 1)  # Ensure 2D array for single task
    
    coverage_axes = axes[0] if len(tasks) > 1 else [axes[0, 0]]
    efficiency_axes = axes[1] if len(tasks) > 1 else [axes[1, 0]]
    
    coverage_axes = _ensure_axes(coverage_axes, len(tasks))
    efficiency_axes = _ensure_axes(efficiency_axes, len(tasks))

    # Plot coverage
    for ax, task in zip(coverage_axes, tasks):
        subset = eval_df[eval_df["task"] == task]
        ordered_methods = _method_order(subset["method"].unique())
        data = [subset[subset["method"] == method]["coverage"].dropna().values for method in ordered_methods]

        bp = ax.boxplot(
            data,
            labels=ordered_methods,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.0},
        )

        for patch, method in zip(bp["boxes"], ordered_methods):
            patch.set_facecolor(PLOT_COLORS.get(method, "#cccccc"))
            patch.set_alpha(0.6)

        for idx, method in enumerate(ordered_methods, start=1):
            y_values = subset[subset["method"] == method]["coverage"].dropna().values
            if y_values.size == 0:
                continue
            x_values = np.random.normal(idx, 0.05, size=len(y_values))
            ax.scatter(x_values, y_values, color=PLOT_COLORS.get(method, "#333333"), s=18, alpha=0.7)

        title = f"Coverage - {task.capitalize()}"
        if title_note:
            title += f" ({title_note})"
        ax.set_title(title)
        ax.set_ylabel("Coverage")
        ax.set_ylim(0.6, 1.05)
        target = 1 - subset["alpha"].median() if "alpha" in subset and subset["alpha"].notna().any() else None
        if target:
            ax.axhline(target, color="k", linestyle="--", linewidth=1.0, alpha=0.6, label=f"Target {(target*100):.1f}%")
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

        if rotate_xticks:
            ax.set_xticks(range(1, len(ordered_methods) + 1))
            ax.set_xticklabels(ordered_methods, rotation=45, ha="right")
        
        # Add average noise level annotation
        if "noisy_accuracy" in subset.columns:
            avg_noise = 1 - subset["noisy_accuracy"].mean()  # noise rate = 1 - accuracy
            ax.text(0.02, 0.98, f"Avg noise: {avg_noise:.2%}", 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Plot length efficiency
    for ax, task in zip(efficiency_axes, tasks):
        subset = eval_df[eval_df["task"] == task]
        ordered_methods = _method_order(subset["method"].unique())
        
        # Filter out methods without efficiency data
        methods_with_data = []
        efficiency_data = []
        for method in ordered_methods:
            method_data = subset[subset["method"] == method]["length_efficiency_vs_scp"].dropna().values
            if method_data.size > 0:
                methods_with_data.append(method)
                efficiency_data.append(method_data)
        
        if efficiency_data:
            bp = ax.boxplot(
                efficiency_data,
                labels=methods_with_data,
                patch_artist=True,
                medianprops={"color": "black", "linewidth": 1.0},
            )

            for patch, method in zip(bp["boxes"], methods_with_data):
                patch.set_facecolor(PLOT_COLORS.get(method, "#cccccc"))
                patch.set_alpha(0.6)

            for idx, method in enumerate(methods_with_data, start=1):
                y_values = subset[subset["method"] == method]["length_efficiency_vs_scp"].dropna().values
                if y_values.size == 0:
                    continue
                x_values = np.random.normal(idx, 0.05, size=len(y_values))
                ax.scatter(x_values, y_values, color=PLOT_COLORS.get(method, "#333333"), s=18, alpha=0.7)

        ax.set_title(f"Length Efficiency vs SCP - {task.capitalize()}")
        ax.set_ylabel("Length Efficiency Ratio")
        ax.axhline(1.0, color="k", linestyle="--", linewidth=1.0, alpha=0.6, label="SCP baseline")
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

        if rotate_xticks and methods_with_data:
            ax.set_xticks(range(1, len(methods_with_data) + 1))
            ax.set_xticklabels(methods_with_data, rotation=45, ha="right")

    if len(tasks) == 1:
        coverage_axes[0].legend(loc="lower left")
        efficiency_axes[0].legend(loc="upper right")

    if footer_note:
        fig.text(0.5, 0.02, footer_note, ha="center", va="bottom", fontsize=10)

    fig.tight_layout(rect=[0, 0.04 if footer_note else 0, 1, 1])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

# read from the files generated in test_batch.py
alpha = 0.1
num_samples = 2000
num_trials = 100
gmm_params = {
    "temperature": 0.75,
    "deterministic_margin": 0.2,
    "K": 3,
    "d": 4,
}

output_dir = Path("out") / "low dim (large batch)"
identifier = (
    f"samples{num_samples}_trials{num_trials}_"
    f"temperature{gmm_params['temperature']}_margin{gmm_params['deterministic_margin']}_"
    f"K{gmm_params['K']}_d{gmm_params['d']}"
)
# TODO read from filesystem and get the parameters, ask user which to visualize if multiple exist

# Read the merged results file
results_path = output_dir / f"result_{identifier}.csv"
results_df = pd.read_csv(results_path)

# Prepare data for plotting
plot_df = results_df.copy()
plot_df["task"] = "low-dim"
plot_df["coverage"] = plot_df["coverage_rate"]
plot_df["alpha"] = alpha

coverage_plot_path = output_dir / f"coverage_and_efficiency_{identifier}.png"
plot_coverage_and_efficiency(
    plot_df,
    coverage_plot_path,
    title_note=f"{num_trials} trials",
    footer_note=f"Sample size {num_samples}, alpha={alpha}",
    rotate_xticks=True,
)

print(f"Saved coverage and efficiency plot at: {coverage_plot_path}")

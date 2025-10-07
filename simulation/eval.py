"""
Simulation study comparing multiple conformal prediction methods.
1. SCP (Split Conformal Prediction) - standard conformal prediction baseline
2. Cluster-wise Oracle - conditional on discrete feature clusters (Equation 5)
3. PCP - complete posterior conformal prediction implementation
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings("ignore")

from pcp import simulate_data, train_val_test_split, SCP, PCP
from eval_const import OUTPATH_FIG
from eval_utils import print_results, plot_results


class ClusterwiseOracle:
    """Oracle method that conditions on the latent clusters used to generate the data."""

    def __init__(self, true_clusters):
        self.name = "Cluster-wise Oracle"
        self.true_clusters = np.asarray(true_clusters)
        self.cluster_residuals = {}
        self.alpha_used = None
        self.missing_clusters_ = set()

    def fit_clusters(self, R_val, val_indices):
        cluster_ids = self.true_clusters[val_indices]
        self.cluster_residuals = {}
        for cluster_id, residual in zip(cluster_ids, R_val):
            self.cluster_residuals.setdefault(int(cluster_id), []).append(residual)
        for cluster_id in self.cluster_residuals:
            self.cluster_residuals[cluster_id] = np.asarray(self.cluster_residuals[cluster_id])

    def calibrate(self, R_val, R_test, alpha, val_indices, test_indices):
        self.alpha_used = alpha
        self.fit_clusters(R_val, val_indices)

        global_quantile = np.quantile(R_val, 1 - alpha)
        quantiles = []
        coverage = []
        missing_clusters = set()

        test_cluster_ids = self.true_clusters[test_indices]

        for i, cluster_id in enumerate(test_cluster_ids):
            residuals = self.cluster_residuals.get(int(cluster_id))
            if residuals is None or residuals.size == 0:
                raise ValueError(f"Cluster {cluster_id} not found in validation set.")
            elif residuals.size == 1:
                quantile = residuals[0]
            else:
                quantile = np.quantile(residuals, 1 - alpha)

            quantiles.append(float(quantile))
            coverage.append(float(R_test[i] <= quantile))

        self.missing_clusters_ = missing_clusters
        return quantiles, coverage


def run_simulation(num_samples=1200, alpha=0.1, random_seed=42, pcp_fold=20, pcp_grid=20,
                   K=3, d=2, mean_scale=4.0, temperature=1.0, cluster_spread=1.0, response_noise=0.5,
                   deterministic_margin=0.2, n_estimators=200, max_features='sqrt', max_depth=None,
                   min_samples_leaf=5, min_samples_split=10, n_jobs=-1,
                   train_ratio=0.4, val_ratio=0.3):
    """Run the GMM-based simulation comparing SCP, Oracle, and PCP."""

    print(f"Running GMM simulation with {num_samples} samples, alpha={alpha}, K={K}, d={d}, temperature={temperature}")
    print("=" * 70)

    X, Y, meta = simulate_data(
        num_samples,
        K=K,
        d=d,
        mean_scale=mean_scale,
        temperature=temperature,
        cluster_spread=cluster_spread,
        response_noise=response_noise,
        deterministic_margin=deterministic_margin,
        random_state=random_seed,
        return_meta=True,
    )

    X_train, X_val, X_test, Y_train, Y_val, Y_test, \
    X_val_0, idx_val, X_test_0, idx_test = train_val_test_split(
        X, Y, p=train_ratio, p2=val_ratio, return_index=True, random_state=random_seed
    )

    print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    rf = RandomForestRegressor(
        random_state=random_seed,
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        n_jobs=n_jobs,
    )
    rf.fit(X_train, Y_train)

    predictions_val = rf.predict(X_val)
    R_val = np.abs(Y_val - predictions_val)
    predictions_test = rf.predict(X_test)
    R_test = np.abs(Y_test - predictions_test)

    print(f"Model trained. Val RMSE: {np.sqrt(np.mean((Y_val - predictions_val) ** 2)):.3f}")
    print(f"Test RMSE: {np.sqrt(np.mean((Y_test - predictions_test) ** 2)):.3f}")

    results = {}

    # Baseline: standard split conformal
    print("\n-- Running SCP...")
    scp_quantiles, scp_coverage = SCP(R_val, R_test, alpha)
    results['SCP'] = {
        'quantiles': scp_quantiles,
        'coverage': scp_coverage,
        'avg_length': np.mean(scp_quantiles) * 2,
        'coverage_rate': np.mean(scp_coverage),
    }

    # Oracle that sees the true latent clusters
    print("-- Running Cluster-wise Oracle...")
    cluster_oracle = ClusterwiseOracle(meta['true_clusters'])
    cluster_quantiles, cluster_coverage = cluster_oracle.calibrate(
        R_val, R_test, alpha, idx_val, idx_test
    )

    val_true_clusters = meta['true_clusters'][idx_val]
    test_true_clusters = meta['true_clusters'][idx_test]
    val_noisy = meta['noisy_labels'][idx_val]
    test_noisy = meta['noisy_labels'][idx_test]
    val_ambiguous = meta['ambiguous_mask'][idx_val]
    test_ambiguous = meta['ambiguous_mask'][idx_test]

    val_unique, val_counts = np.unique(val_true_clusters, return_counts=True)
    test_unique, test_counts = np.unique(test_true_clusters, return_counts=True)

    print(f"   Validation clusters: {len(val_unique)} (avg size {np.mean(val_counts):.1f})")
    print(f"   Test clusters:       {len(test_unique)} (avg size {np.mean(test_counts):.1f})")
    print(f"   Noisy label accuracy (val/test): {np.mean(val_noisy == val_true_clusters):.3f} / {np.mean(test_noisy == test_true_clusters):.3f}")
    print(f"   Ambiguous region rate (val/test): {np.mean(val_ambiguous):.3f} / {np.mean(test_ambiguous):.3f}")
    if cluster_oracle.missing_clusters_:
        print(f"   Warning: Missing clusters in validation set {sorted(cluster_oracle.missing_clusters_)}; using global quantile fallback.")

    results['Cluster-wise Oracle'] = {
        'quantiles': cluster_quantiles,
        'coverage': cluster_coverage,
        'avg_length': np.mean(cluster_quantiles) * 2,
        'coverage_rate': np.mean(cluster_coverage),
    }

    # PCP with externally supplied (noisy) cluster probabilities
    print("-- Running PCP (with pre-labeled clusters)...")
    val_cluster_probs = meta['noisy_membership'][idx_val]
    test_cluster_probs = meta['noisy_membership'][idx_test]

    pcp_model = PCP(fold=pcp_fold, grid=pcp_grid)
    pcp_model.train(X_val, R_val, info=False, cluster_probs=val_cluster_probs)

    pcp_quantiles, pcp_coverage = pcp_model.calibrate(
        X_val,
        R_val,
        X_test,
        R_test,
        alpha,
        return_pi=False,
        finite=True,
        max_iter=5,
        tol=0.01,
        cluster_probs_val=val_cluster_probs,
        cluster_probs_test=test_cluster_probs,
    )

    results['PCP'] = {
        'quantiles': pcp_quantiles,
        'coverage': pcp_coverage,
        'avg_length': np.mean(pcp_quantiles) * 2,
        'coverage_rate': np.mean(pcp_coverage),
    }
    print("   PCP completed successfully!")

    diagnostics = {
        'true_clusters_test': test_true_clusters,
        'ambiguous_mask_test': test_ambiguous,
        'noisy_labels_test': test_noisy,
        'temperature': temperature,
    }

    return results, X_test_0, Y_test, predictions_test, diagnostics


def main():
    """Main simulation function"""
    # Simulation parameters
    num_samples = 1500
    alpha = 0.1
    random_seed = 42
    K = 3
    d = 2
    temperature = 1.0
    n_estimators = 200
    
    print("Simulation Study: Comparing Conformal Prediction Methods")
    print(f"Sample size: {num_samples}")
    print(f"Alpha (significance level): {alpha}")
    print(f"Gaussian mixture: K={K}, d={d}, temperature={temperature}")
    
    # Run simulation
    results, X_test_0, Y_test, predictions_test, diagnostics = run_simulation(
        num_samples=num_samples, 
        alpha=alpha, 
        random_seed=random_seed,
        K=K,
        d=d,
        temperature=temperature,
        n_estimators=n_estimators,
    )
    
    # Print results
    print_results(results, alpha)
    
    # Generate analysis
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")
    
    scp_coverage = results['SCP']['coverage_rate']
    scp_length = results['SCP']['avg_length']
    
    print(f"- SCP (baseline): {scp_coverage:.3f} coverage, {scp_length:.1f} average length")
    
    for name, result in results.items():
        if name != 'SCP':
            cov_diff = result['coverage_rate'] - scp_coverage
            len_ratio = scp_length / result['avg_length']
            print(f"- {name}: {result['coverage_rate']:.3f} coverage (+{cov_diff:+.3f}), "
                  f"{len_ratio:.2f}x length efficiency")
    
    print(f"- Cluster-wise Oracle conditions on true latent clusters")
    if 'PCP' in results:
        print(f"- PCP consumes externally supplied cluster labels (no k-means stage)")
    else:
        print(f"- PCP failed due to numerical instability with this sample size")

    # Plot results
    print(f"\nGenerating plots...")
    plot_results(
        results,
        X_test_0,
        Y_test,
        predictions_test,
        num_samples=num_samples,
        n_tree=n_estimators,
        seed=random_seed,
        setting=f"gmm_K{K}_d{d}",
        temperature=temperature,
        true_clusters=diagnostics['true_clusters_test'],
        ambiguous_mask=diagnostics['ambiguous_mask_test'],
    )

    out_fig = OUTPATH_FIG.format(num_samples=num_samples)
    print(f"\nSimulation completed! Check '{out_fig}' for plots.")

    return results


if __name__ == "__main__":
    results = main()
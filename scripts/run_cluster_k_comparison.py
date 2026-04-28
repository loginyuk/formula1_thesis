import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    SUMMARIES_DIR, DATASET_ALL, CLUSTER_FEATURES, RANDOM_STATE,
    RESULTS_CLUSTERING_K_COMPARISON_DIR,
)
from src.utils import log, write_summary
from src.clustering.gmm import fit_gmm
from src.clustering.normalization import normalize_lap_features
from src.clustering.k_comparison_plots import plot_centroid_radar, plot_cluster_sizes, plot_pca_scatter

warnings.filterwarnings("ignore")

K_RANGE = [2, 3, 4, 5]

if __name__ == "__main__":
    start_all = time.time()
    summary_lines = []
    summary_path = os.path.join(SUMMARIES_DIR, "summary_cluster_k_comparison.txt")
    out_dir = RESULTS_CLUSTERING_K_COMPARISON_DIR
    os.makedirs(out_dir, exist_ok=True)

    log(summary_lines, f"Cluster K Comparison\n")
    log(summary_lines, f"K values: {K_RANGE}\n")

    # load existing clustering features
    df = pd.read_csv(DATASET_ALL)
    cluster_feature_means = [f'Mean_{f}' for f in CLUSTER_FEATURES]
    df = df.dropna(subset=cluster_feature_means).reset_index(drop=True)
    df, z_cols = normalize_lap_features(df, cluster_feature_means)
    X = df[z_cols].values

    log(summary_lines, f"Data: {len(df)} laps")
    log(summary_lines, f"Features: {z_cols}\n")

    # fit GMM for each K
    rows = []
    fitted = {}

    for k in K_RANGE:
        gmm, proba, labels, centroids = fit_gmm(X, k)
        fitted[k] = {'gmm': gmm, 'labels': labels, 'proba': proba, 'centroids': centroids}

        # compute metrics
        sil = silhouette_score(X, labels, sample_size=min(5000, len(X)), random_state=RANDOM_STATE)
        db = davies_bouldin_score(X, labels)
        bic = gmm.bic(X)
        aic = gmm.aic(X)
        ll = gmm.score(X) * len(X)

        sizes = np.bincount(labels, minlength=k)
        balance = sizes.min() / sizes.max()

        rows.append({
            'K': k,
            'BIC': round(bic, 1),
            'AIC': round(aic, 1),
            'Silhouette': round(sil, 4),
            'Davies_Bouldin': round(db, 4),
            'Log_Likelihood': round(ll, 1),
            'Balance (min/max)': round(balance, 3),
            'Sizes': ', '.join(str(s) for s in sizes),
        })

    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(out_dir, "metrics_table.csv"), index=False)

    feature_labels = [f.replace('Mean_', '') for f in cluster_feature_means]
    plot_centroid_radar(fitted, K_RANGE, feature_labels, out_dir)
    plot_cluster_sizes(fitted, K_RANGE, out_dir)
    plot_pca_scatter(X, fitted, K_RANGE, out_dir, random_state=RANDOM_STATE)

    # centroid values
    log(summary_lines, "\nCentroid values per K:")
    for k in K_RANGE:
        centroids = fitted[k]['centroids']
        log(summary_lines, f"\nK={k}:")
        for cid in range(k):
            vals = ', '.join(
                f'{feature_labels[j]}={centroids[cid, j]:+.3f}'
                for j in range(len(feature_labels))
            )
            log(summary_lines, f"  C{cid}: {vals}")

    log(summary_lines, f"\nTotal time: {time.time() - start_all:.1f} s")
    write_summary(summary_lines, summary_path)

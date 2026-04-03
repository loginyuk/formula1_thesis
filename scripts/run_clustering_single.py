"""
run_clustering_single.py
────────────────────────
Runs the clustering pipeline on a single race for inspection/debugging.
Saves lap-level cluster results and plots.

Run from project root:
    python scripts/run_clustering_single.py
    python scripts/run_clustering_single.py --year 2023 --location Silverstone
"""

import os
import sys
import time
import argparse

import fastf1

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CACHE_DIR, N_CLUSTERS, CLUSTER_FEATURES, RESULTS_CLUSTERING_DIR
from src.utils import log, write_summary
from src.clustering.corners import build_corner_zones, build_corner_database
from src.clustering.gmm import calculate_circuit_weights, aggregate_corners_to_laps, normalize_lap_features, cluster_laps, silhouette_scoring
from src.clustering.plots import plot_lap_clusters_scatter, plot_race_timeline, plot_probability_distributions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2023)
    parser.add_argument('--location', type=str, default='Silverstone')
    args = parser.parse_args()

    out_dir = os.path.join(RESULTS_CLUSTERING_DIR, f"{args.year}_{args.location.replace(' ', '_')}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)

    start_time = time.time()
    summary_lines = []
    summary_path = os.path.join(out_dir, "summary.txt")

    session = fastf1.get_session(args.year, args.location, 'R')
    session.load(telemetry=True, weather=False, messages=False)
    all_laps = session.laps.pick_quicklaps().pick_wo_box().reset_index(drop=True)

    log(summary_lines, f"Session: {session.event['EventName']} {session.event.year}")

    corner_zones = build_corner_zones(session)

    log(summary_lines, "\nExtracting corner features")
    df_corners = build_corner_database(session, all_laps, corner_zones=corner_zones)
    log(summary_lines, f"  {len(df_corners)} corners")

    log(summary_lines, "Aggregating corners to laps")
    corner_weights = calculate_circuit_weights(session, corner_zones=corner_zones)
    df_laps = aggregate_corners_to_laps(df_corners, corner_weights)
    log(summary_lines, f"  {len(df_laps)} laps, {len(df_laps['Driver'].unique())} drivers")

    cluster_feature_means = [f'Mean_{f}' for f in CLUSTER_FEATURES]
    df_laps_norm, z_features = normalize_lap_features(df_laps, cluster_feature_means)
    log(summary_lines, "\nFeature ranges after normalisation:")
    log(summary_lines, df_laps_norm[z_features].describe().loc[['min', 'max', 'std']].round(2).to_string())

    silhouette_scoring(df_laps_norm[z_features].values, max_k=4)

    log(summary_lines, f"\nFitting GMM with k={N_CLUSTERS} clusters")
    df_laps_clustered = cluster_laps(df_laps_norm, z_features, N_CLUSTERS)

    p_cols = sorted([c for c in df_laps_clustered.columns if c.startswith('P_') and c[2:].isdigit()])
    log(summary_lines, "\nSample output:")
    cols = ['Driver', 'LapNumber'] + p_cols + ['Style_Entropy', 'Style_Cluster_ID']
    log(summary_lines, df_laps_clustered[df_laps_clustered['Driver'] == 'VER'][cols].head(10).to_string())

    log(summary_lines, "\nCluster distribution:")
    log(summary_lines, df_laps_clustered['Style_Cluster_ID'].value_counts().to_string())

    centroid_cols = z_features + ['Style_Cluster_ID']
    log(summary_lines, "\nCluster centroids (Z-scored features):")
    log(summary_lines, df_laps_clustered[centroid_cols].groupby('Style_Cluster_ID').mean().round(3).to_string())

    raw_mean_cols = [f'Mean_{f}' for f in CLUSTER_FEATURES]
    log(summary_lines, "\nCluster centroids (raw feature means):")
    log(summary_lines, df_laps_clustered[raw_mean_cols + ['Style_Cluster_ID']].groupby('Style_Cluster_ID').mean().round(4).to_string())

    csv_path = os.path.join(out_dir, "lap_clusters.csv")
    df_laps_clustered.to_csv(csv_path, index=False)
    log(summary_lines, f"\nLap-level results saved to {csv_path}")
    log(summary_lines, f"\nClustering completed in {time.time() - start_time:.2f} seconds")

    write_summary(summary_lines, summary_path)

    # plots
    drivers_dir = os.path.join(out_dir, "drivers")
    plot_lap_clusters_scatter(df_laps_clustered, 'Z_Mean_Apex_Speed_Ratio', 'Z_Mean_Throttle_Integral_Norm', 'VER', out_dir=drivers_dir)
    for drv in df_laps_clustered['Driver'].unique():
        plot_race_timeline(df_laps_clustered, drv, out_dir=drivers_dir)
    plot_probability_distributions(df_laps_clustered, out_dir=out_dir)

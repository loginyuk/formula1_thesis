import os
import sys
import time
import fastf1
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CACHE_DIR, N_CLUSTERS, CLUSTER_FEATURES, RESULTS_CLUSTERING_DIR
from src.utils import log, write_summary
from src.clustering.corners import build_corner_zones, build_corner_database
from src.clustering.weighting import calculate_circuit_weights
from src.clustering.aggregation import aggregate_corners_to_laps
from src.clustering.normalization import normalize_lap_features
from src.clustering.gmm import cluster_laps
from src.clustering.plots import (plot_race_timeline, plot_centroid_profiles, plot_feature_space, 
                            plot_driver_composition, plot_race_evolution, plot_laptime_by_cluster)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2023)
    parser.add_argument('--location', type=str, default='Silverstone')
    parser.add_argument('--k', type=int, default=N_CLUSTERS, help='Number of GMM clusters')
    args = parser.parse_args()

    n_clusters = args.k

    out_dir = os.path.join(RESULTS_CLUSTERING_DIR, f"{args.year}_{args.location.replace(' ', '_')}_k{args.k if args.k != N_CLUSTERS else N_CLUSTERS}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)

    start_time = time.time()
    summary_lines = []
    summary_path = os.path.join(out_dir, "summary.txt")

    # load session
    session = fastf1.get_session(args.year, args.location, 'R')
    session.load(telemetry=True, weather=False, messages=False)

    df_laps = session.laps

    # filter laps for clustering
    clustering_laps = df_laps[
        (df_laps['TrackStatus'] == '1') &
        (df_laps['PitInTime'].isna()) &
        (df_laps['PitOutTime'].isna()) &
        (df_laps.get('FastF1Generated', False) == False)
    ].reset_index(drop=True)

    log(summary_lines, f"Session: {session.event['EventName']} {session.event.year}")

    # build corner database and aggregate to lap-level features
    corner_zones = build_corner_zones(session)

    df_corners = build_corner_database(session, clustering_laps, corner_zones=corner_zones)
    corner_weights = calculate_circuit_weights(session, corner_zones=corner_zones)
    df_laps = aggregate_corners_to_laps(df_corners, corner_weights)
    log(summary_lines, f"Data: {len(df_laps)} laps, {len(df_corners)} corners, {len(df_laps['Driver'].unique())} drivers")
    
    cluster_feature_means = [f'Mean_{f}' for f in CLUSTER_FEATURES]
    df_laps_norm, z_features = normalize_lap_features(df_laps, cluster_feature_means)
    log(summary_lines, "\nFeature ranges after normalisation:")
    log(summary_lines, df_laps_norm[z_features].describe().loc[['min', 'max', 'std']].round(2).to_string())

    # clustering
    log(summary_lines, f"\nFitting GMM with k={n_clusters} clusters")
    df_laps_clustered = cluster_laps(df_laps_norm, z_features, n_clusters)

    # cluster distribution and centroids
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
    log(summary_lines, f"\nClustering completed in {time.time() - start_time:.2f} seconds")

    write_summary(summary_lines, summary_path)

    # per-driver timelines
    drivers_dir = os.path.join(out_dir, "drivers")
    for drv in df_laps_clustered['Driver'].unique():
        plot_race_timeline(df_laps_clustered, drv, out_dir=drivers_dir)

    # other visualizations
    viz_dir = os.path.join(out_dir, "visualizations")
    plot_centroid_profiles(df_laps_clustered, z_features, args.location, out_dir=viz_dir)
    plot_feature_space(df_laps_clustered, z_features, args.location, out_dir=viz_dir)
    plot_driver_composition(df_laps_clustered, args.location, out_dir=viz_dir)
    plot_race_evolution(df_laps_clustered, args.location, out_dir=viz_dir)
    plot_laptime_by_cluster(df_laps_clustered, args.location, out_dir=viz_dir)

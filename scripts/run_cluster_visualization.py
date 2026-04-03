"""
run_cluster_visualization.py
──────────────────
Generates 6 cluster visualization plots for a given race.

Run from project root:
    python scripts/run_cluster_visualization.py
    python scripts/run_cluster_visualization.py --location Silverstone --driver VER --year 2023
"""

import os
import sys
import time
import argparse

import fastf1
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CACHE_DIR, CLUSTER_FEATURES, RESULTS_VISUALIZATIONS_DIR, DATASET_ALL
from src.clustering.gmm import normalize_lap_features
from src.visualization.clustering_plots import (
    plot_centroid_profiles, plot_feature_space, plot_driver_composition,
    plot_race_evolution, plot_driver_enhanced_timeline, plot_laptime_by_cluster,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', type=str, default='Jeddah')
    parser.add_argument('--driver', type=str, default='VER')
    parser.add_argument('--year', type=int, default=None)
    args = parser.parse_args()

    t0 = time.time()

    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)

    year_str = str(args.year) if args.year else "all years"
    print(f"Loading {args.location} ({year_str}) data from dataset...")
    df_season = pd.read_csv(DATASET_ALL)
    df_loc = df_season[df_season['Location'] == args.location]
    if args.year is not None:
        df_loc = df_loc[df_loc['Year'] == args.year]
    df_loc = df_loc.reset_index(drop=True)

    if df_loc.empty:
        year_hint = f" for year {args.year}" if args.year else ""
        raise ValueError(f"{args.location}{year_hint} not found in dataset_all.csv")

    cluster_feature_means = [f'Mean_{f}' for f in CLUSTER_FEATURES]
    df_clustered, z_features = normalize_lap_features(df_loc, cluster_feature_means)
    df_clustered['Style_Cluster_ID'] = df_clustered['Style_Cluster_ID'].astype(int)

    year_suffix = f"_{args.year}" if args.year else ""
    out_dir = os.path.join(RESULTS_VISUALIZATIONS_DIR, f"{args.location.replace(' ', '_')}{year_suffix}")

    print(f"\nSaving plots to {out_dir}/")
    plot_centroid_profiles(df_clustered, z_features, args.location, out_dir=out_dir)
    plot_feature_space(df_clustered, z_features, args.location, out_dir=out_dir)
    plot_driver_composition(df_clustered, args.location, out_dir=out_dir)
    plot_race_evolution(df_clustered, args.location, out_dir=out_dir)
    plot_driver_enhanced_timeline(df_clustered, args.driver, args.location, out_dir=out_dir)
    plot_laptime_by_cluster(df_clustered, args.location, out_dir=out_dir)

    print(f"\nDone in {time.time() - t0:.1f}s")

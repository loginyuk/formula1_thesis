import os
import sys
import time
import fastf1
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CACHE_DIR, DATASET_ALL, ERRORS_LOG, SUMMARIES_DIR
from src.utils import log, write_summary
from src.clustering.pipeline import run_clustering_features

K2_COLS = ['P_0', 'P_1', 'Style_Cluster_ID', 'Style_Entropy']
RENAME = {c: f'{c}_k2' for c in K2_COLS}

if __name__ == "__main__":
    start_time = time.time()

    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)

    summary_lines = []
    summary_path = os.path.join(SUMMARIES_DIR, "summary_clustering_k2.txt")

    # get list of races
    df = pd.read_csv(DATASET_ALL)
    races = df[['Year', 'Location']].drop_duplicates().values.tolist()
    log(summary_lines, f"K=2 clustering generation\n")
    log(summary_lines, f"Data: {len(df)} laps, {len(races)} races\n")

    all_k2 = []

    for year, location in races:
        log(summary_lines, f"\n{'-' * 55}")
        log(summary_lines, f"{year} - {location}")

        try:
            session = fastf1.get_session(int(year), location, 'R')
            session.load(telemetry=True, weather=False, messages=False)

            # filter laps for clustering
            laps = session.laps
            clustering_laps = laps[
                (laps['TrackStatus'] == '1') &
                (laps['PitInTime'].isna()) &
                (laps['PitOutTime'].isna()) &
                (laps.get('FastF1Generated', False) == False)
            ].reset_index(drop=True)

            df_clustered = run_clustering_features(
                session, clustering_laps, n_clusters=2,
                circuit_info=session.get_circuit_info(),
            )
            if df_clustered.empty:
                log(summary_lines, f"No laps clustered for K=2")
                continue

            # rename columns for merging back to main dataset
            keep = ['Driver', 'LapNumber'] + K2_COLS
            df_k2 = df_clustered[[c for c in keep if c in df_clustered.columns]].copy()
            df_k2 = df_k2.rename(columns=RENAME)
            df_k2['Year'] = int(year)
            df_k2['Location'] = location

            log(summary_lines, f"Cluster sizes: {df_k2['Style_Cluster_ID_k2'].value_counts().sort_index().to_dict()}")
            all_k2.append(df_k2)

        except Exception as e:
            log(summary_lines, f"Error processing {year} {location}, skipping K=2 clustering")
            continue

    if not all_k2:
        log(summary_lines, f"No K=2 clusters generated for any race")
        write_summary(summary_lines, summary_path)
        sys.exit(1)

    k2_df = pd.concat(all_k2, ignore_index=True)
    log(summary_lines, f"\n{'-' * 55}")
    log(summary_lines, f"Total K=2 cluster rows: {len(k2_df)}")

    # drop existing k2 columns if they exist
    for col in [f'{c}_k2' for c in K2_COLS]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # merge 2 clusters results back to main dataset
    merged = df.merge(k2_df, on=['Year', 'Location', 'Driver', 'LapNumber'], how='left')
    n_with_k2 = merged['Style_Cluster_ID_k2'].notna().sum()

    merged.to_csv(DATASET_ALL, index=False)
    log(summary_lines, f"\n{DATASET_ALL} updated with K=2 cluster columns")
    log(summary_lines, f"Total time: {(time.time() - start_time) / 60:.2f} min")

    write_summary(summary_lines, summary_path)

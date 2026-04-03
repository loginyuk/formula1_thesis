"""
run_data_pipeline.py
────────────────────
Full data pipeline: loads FastF1 sessions for all years/locations,
generates telemetry features, runs clustering, cleans data, and saves
a single combined dataset to data/processed/dataset_all.csv.

Run from project root:
    python scripts/run_data_pipeline.py
"""

import os
import sys
import time
import logging

import fastf1
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CACHE_DIR, DATA_PROCESSED_DIR, SUMMARIES_DIR, ERRORS_LOG, TRACK_PARAMETERS_FILE, RESULTS_CLUSTERING_VERIFICATION_DIR
from src.utils import log, write_summary
from src.logging_setup import setup_file_logging
from src.data_preparation.loading import prepare_race, merge_weather, get_pirelli_press_data
from src.data_preparation.physics import calculate_physics_fuel_load, add_physics_track_evolution
from src.data_preparation.cleaning import clean_laps, add_lag_features, encode_categorical_features
from src.telemetry.pipeline import run_telemetry_feature_generation
from src.clustering.plots import plot_cluster_verification

if __name__ == "__main__":
    start_time = time.time()

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)

    race_filter = setup_file_logging(ERRORS_LOG)

    summary_lines = []
    summary_path = os.path.join(SUMMARIES_DIR, "summary_data_pipeline.txt")

    df_pirelli_all = get_pirelli_press_data(TRACK_PARAMETERS_FILE)

    YEARS = [2022, 2023, 2024, 2025]
    full_dataset = []

    for YEAR in YEARS:
        df_pirelli = df_pirelli_all[df_pirelli_all['Year'] == YEAR].copy()
        locations = df_pirelli['Location'].unique()

        for location in locations:
            race_filter.race = f"{YEAR} {location}"
            log(summary_lines, f"\n{'-'*55}")
            log(summary_lines, f"Processing: {YEAR} - {location}\n")

            try:
                session = fastf1.get_session(YEAR, location, 'R')
                for attempt in range(3):
                    try:
                        session.load()
                        break
                    except Exception as e:
                        if attempt == 2:
                            raise
                        logging.warning(f"{YEAR} {location}: session.load() attempt {attempt+1} failed: {e}, retrying...")
                        time.sleep(5)

                circuit_info = session.get_circuit_info()

                laps, drivers, stints = prepare_race(session)
                df_weather = merge_weather(session, laps)
                df_fuel = calculate_physics_fuel_load(session, df_weather)
                df_track = add_physics_track_evolution(df_fuel, df_pirelli)

                df_telemetry = run_telemetry_feature_generation(session, df_track, circuit_info=circuit_info)

                df_clean = clean_laps(df_telemetry, summary_lines)
                df_lag = add_lag_features(df_clean, summary_lines)

                full_dataset.append(df_lag)
                log(summary_lines, f"{YEAR} {location} processed {len(df_lag)} rows")

            except Exception as e:
                logging.error(f"{YEAR} {location}: {e}", exc_info=True)
                log(summary_lines, f"Could not process {YEAR} {location}. Error: {e}")
                continue

    if full_dataset:
        df_all = pd.concat(full_dataset, ignore_index=True)

        # encode once across all years
        df_all, label_encoders = encode_categorical_features(df_all)
        output_path = os.path.join(DATA_PROCESSED_DIR, 'dataset_all.csv')
        df_all.to_csv(output_path, index=False)

        if 'Style_Cluster_ID' in df_all.columns:
            plot_cluster_verification(df_all)
            log(summary_lines, f"Verification plots saved to {RESULTS_CLUSTERING_VERIFICATION_DIR}/")
        else:
            log(summary_lines, "Skipping cluster verification — clustering columns missing")

        total_time = (time.time() - start_time) / 60
        log(summary_lines, f"Total time taken: {total_time:.2f} minutes")
        log(summary_lines, f"Total laps: {len(df_all)}")
    else:
        log(summary_lines, "No races were successfully processed")

    write_summary(summary_lines, summary_path)

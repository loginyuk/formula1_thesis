"""
run_model_training.py
─────────────────────
XGBoost walk-forward training on the full processed dataset.
Telemetry features are shifted by 1 lap to remove data leakage (forecasting model).

Run from project root:
    python scripts/run_model_training.py
"""

import os
import sys
import time

import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SUMMARIES_DIR, MODEL_FEATURES, DATASET_ALL
from src.utils import log, write_summary
from src.modeling.training import run_season_walk_forward, convert_deltas_to_absolute_times, shift_telemetry_features
from src.modeling.analysis import plot_feature_importance
from src.modeling.plots import plot_full_season_slopes

if __name__ == "__main__":
    start = time.time()
    summary_lines = []
    summary_path = os.path.join(SUMMARIES_DIR, "summary_model_training.txt")

    df = pd.read_csv(DATASET_ALL)

    df = shift_telemetry_features(df)

    log(summary_lines, f"Features used for training ({len(MODEL_FEATURES)}):")
    log(summary_lines, f"{MODEL_FEATURES}\n")

    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)

    simulation_results = run_season_walk_forward(
        df, MODEL_FEATURES, model, summary_lines,
        target='Target_Delta', min_train_races=20, print_progress=True
    )
    simulation_results = convert_deltas_to_absolute_times(simulation_results, df)

    plot_feature_importance(df, MODEL_FEATURES)
    plot_full_season_slopes(simulation_results, 'VER')

    write_summary(summary_lines, summary_path)

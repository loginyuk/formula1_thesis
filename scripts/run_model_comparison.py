"""
run_model_comparison.py
───────────────────────
Walk-forward validation across multiple model families.
Each model is trained and evaluated identically using the same folds.

Run from project root:
    python scripts/run_model_comparison.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SUMMARIES_DIR, RESULTS_MODEL_DIR, MODEL_FEATURES, DATASET_ALL
from src.utils import log, write_summary
from src.modeling.training import run_season_walk_forward, convert_deltas_to_absolute_times, shift_telemetry_features
from src.modeling.plots import plot_full_season_slopes
from src.modeling.analysis import plot_model_comparison
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODELS = {
    "Ridge": Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', Ridge()),
    ]),
    "RandomForest": Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestRegressor(
            n_estimators=100, max_features='sqrt', max_depth=15,
            n_jobs=-1, random_state=42
        )),
    ]),
    "XGBoost": XGBRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        n_jobs=-1, random_state=42, verbosity=0
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=100, max_depth=5, num_leaves=31,
        n_jobs=-1, random_state=42, verbose=-1
    ),
    "CatBoost": CatBoostRegressor(
        iterations=100, depth=5, learning_rate=0.1,
        random_state=42, verbose=0
    ),
}

MIN_TRAIN_RACES = 20

if __name__ == "__main__":
    start_all = time.time()
    summary_lines = []
    summary_path = os.path.join(SUMMARIES_DIR, "summary_model_comparison.txt")

    df = pd.read_csv(DATASET_ALL)
    df = shift_telemetry_features(df)

    comparison_rows = []

    for name, model in MODELS.items():
        log(summary_lines, f"\n{'='*40}")
        log(summary_lines, f"Model: {name}")
        log(summary_lines, f"{'='*40}")

        t0 = time.time()
        results = run_season_walk_forward(
            df, MODEL_FEATURES, model, summary_lines,
            target='Target_Delta', min_train_races=MIN_TRAIN_RACES,
            print_progress=False
        )
        results = convert_deltas_to_absolute_times(results, df)

        mae  = mean_absolute_error(results['Actual'], results['Predicted'])
        rmse = np.sqrt(mean_squared_error(results['Actual'], results['Predicted']))
        elapsed = time.time() - t0

        log(summary_lines, f"  MAE:  {mae:.3f} s")
        log(summary_lines, f"  RMSE: {rmse:.3f} s")
        log(summary_lines, f"  Time: {elapsed:.1f} s")

        comparison_rows.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'Time_s': elapsed})

        model_dir = os.path.join(RESULTS_MODEL_DIR, name)
        os.makedirs(model_dir, exist_ok=True)
        results.to_csv(os.path.join(model_dir, f"results_{name}.csv"), index=False)
        plot_full_season_slopes(results, 'VER', out_dir=model_dir)

    log(summary_lines, f"\n{'='*40}")
    log(summary_lines, "Final Comparison")
    log(summary_lines, f"{'='*40}")
    comp_df = plot_model_comparison(comparison_rows, out_dir=RESULTS_MODEL_DIR)
    log(summary_lines, comp_df.to_string(index=False))
    log(summary_lines, f"\nTotal time: {time.time() - start_all:.1f} s")

    write_summary(summary_lines, summary_path)
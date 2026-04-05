"""
run_model_comparison.py
───────────────────────
Walk-forward validation across multiple model families.
Uses tuned params from best_params.json where available, falls back to MODEL_DEFAULTS.

Run from project root:
    python scripts/run_model_comparison.py
"""

import os
import sys
import json
import time

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

from src.config import (
    SUMMARIES_DIR, RESULTS_MODEL_DIR, MODEL_FEATURES, DATASET_ALL,
    MIN_TRAIN_RACES, BEST_PARAMS_FILE, MODEL_DEFAULTS,
)
from src.utils import log, write_summary
from src.modeling.training import run_season_walk_forward, convert_deltas_to_absolute_times, shift_telemetry_features, compute_metrics
from src.modeling.plots import plot_full_season_slopes
from src.modeling.analysis import plot_model_comparison

MODELS_TO_COMPARE = ["Ridge", "RandomForest", "XGBoost", "LightGBM", "CatBoost"]


def build_model(name, params):
    if name == "XGBoost":
        return XGBRegressor(**params, n_jobs=-1)
    if name == "LightGBM":
        return LGBMRegressor(**params, n_jobs=-1, verbose=-1)
    if name == "CatBoost":
        return CatBoostRegressor(**params, verbose=0, silent=True)
    if name == "RandomForest":
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestRegressor(**params, n_jobs=-1)),
        ])
    if name == "Ridge":
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', Ridge(**params)),
        ])
    raise ValueError(f"Unknown model: {name}")


if __name__ == "__main__":
    start_all = time.time()
    summary_lines = []
    summary_path = os.path.join(SUMMARIES_DIR, "summary_model_comparison.txt")

    best_params = {}
    if os.path.exists(BEST_PARAMS_FILE):
        with open(BEST_PARAMS_FILE) as f:
            best_params = json.load(f)

    df = pd.read_csv(DATASET_ALL)
    df = shift_telemetry_features(df)

    comparison_rows = []

    for name in MODELS_TO_COMPARE:
        params = best_params.get(name, MODEL_DEFAULTS[name])
        source = "tuned" if name in best_params else "default"

        log(summary_lines, f"\n{'-'*40}")
        log(summary_lines, f"Model: {name} ({source} params)")
        log(summary_lines, f"{'-'*40}")

        model = build_model(name, params)

        t0 = time.time()
        results = run_season_walk_forward(
            df, MODEL_FEATURES, model, summary_lines,
            target='Target_Delta', min_train_races=MIN_TRAIN_RACES,
            print_progress=False
        )
        results = convert_deltas_to_absolute_times(results, df)

        metrics = compute_metrics(results['Actual'].values, results['Predicted'].values)
        elapsed = time.time() - t0

        log(summary_lines, f"  MAE:  {metrics['MAE']:.3f} s")
        log(summary_lines, f"  RMSE: {metrics['RMSE']:.3f} s")
        log(summary_lines, f"  R2:   {metrics['R2']:.4f}")
        log(summary_lines, f"  MAPE: {metrics['MAPE']:.2f} %")
        log(summary_lines, f"  Time: {elapsed:.1f} s")

        comparison_rows.append({'Model': name, **metrics, 'Time_s': elapsed})

        model_dir = os.path.join(RESULTS_MODEL_DIR, name)
        os.makedirs(model_dir, exist_ok=True)
        results.to_csv(os.path.join(model_dir, f"results_{name}.csv"), index=False)
        plot_full_season_slopes(results, 'VER', out_dir=model_dir)

    log(summary_lines, f"\n{'-'*40}")
    log(summary_lines, "Final Comparison")
    log(summary_lines, f"{'-'*40}")
    comp_df = plot_model_comparison(comparison_rows, out_dir=RESULTS_MODEL_DIR)
    log(summary_lines, comp_df.to_string(index=False))
    log(summary_lines, f"\nTotal time: {time.time() - start_all:.1f} s")

    write_summary(summary_lines, summary_path)

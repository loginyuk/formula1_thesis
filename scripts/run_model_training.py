"""
run_model_training.py
─────────────────────
Walk-forward training using the primary model defined in config.py.
Model and hyperparameters are loaded from best_params.json if available,
otherwise falls back to sensible defaults.

Run from project root:
    python scripts/run_model_training.py
"""

import os
import sys
import json
import time

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    SUMMARIES_DIR, MODEL_FEATURES, DATASET_ALL,
    MIN_TRAIN_RACES, PRIMARY_MODEL, BEST_PARAMS_FILE, MODEL_DEFAULTS,
)
from src.utils import log, write_summary
from src.modeling.training import run_season_walk_forward, convert_deltas_to_absolute_times, shift_telemetry_features, compute_metrics
from src.modeling.analysis import plot_feature_importance
from src.modeling.plots import (
    plot_full_season_slopes, plot_predicted_vs_actual,
    plot_residual_analysis, plot_per_race_mae, plot_compound_breakdown, plot_driver_mae
)


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
    start = time.time()
    summary_lines = []
    summary_path = os.path.join(SUMMARIES_DIR, "summary_model_training.txt")

    # load best params if available
    if os.path.exists(BEST_PARAMS_FILE):
        with open(BEST_PARAMS_FILE) as f:
            all_best = json.load(f)
        if PRIMARY_MODEL in all_best:
            params = all_best[PRIMARY_MODEL]
            log(summary_lines, f"Loaded tuned params for {PRIMARY_MODEL} from {BEST_PARAMS_FILE}")
        else:
            params = MODEL_DEFAULTS[PRIMARY_MODEL]
            log(summary_lines, f"No tuned params found for {PRIMARY_MODEL}, using defaults")
    else:
        params = MODEL_DEFAULTS[PRIMARY_MODEL]
        log(summary_lines, f"best_params.json not found, using defaults for {PRIMARY_MODEL}")

    log(summary_lines, f"Model: {PRIMARY_MODEL}")
    log(summary_lines, f"Params: {params}\n")

    model = build_model(PRIMARY_MODEL, params)

    df = pd.read_csv(DATASET_ALL)
    df = shift_telemetry_features(df)

    log(summary_lines, f"Features used for training ({len(MODEL_FEATURES)}):")
    log(summary_lines, f"{MODEL_FEATURES}\n")

    simulation_results = run_season_walk_forward(
        df, MODEL_FEATURES, model, summary_lines,
        target='Target_Delta', min_train_races=MIN_TRAIN_RACES, print_progress=True
    )
    simulation_results = convert_deltas_to_absolute_times(simulation_results, df)

    metrics = compute_metrics(simulation_results['Actual'].values, simulation_results['Predicted'].values)
    log(summary_lines, f"\nMetrics on absolute lap times")
    log(summary_lines, f"MAE:  {metrics['MAE']:.3f} s")
    log(summary_lines, f"RMSE: {metrics['RMSE']:.3f} s")
    log(summary_lines, f"R2:   {metrics['R2']:.4f}")
    log(summary_lines, f"MAPE: {metrics['MAPE']:.2f} %")

    plot_feature_importance(df, MODEL_FEATURES, model)
    plot_full_season_slopes(simulation_results, 'VER')
    plot_predicted_vs_actual(simulation_results)
    plot_residual_analysis(simulation_results)
    plot_per_race_mae(simulation_results)
    plot_compound_breakdown(simulation_results)
    plot_driver_mae(simulation_results)

    log(summary_lines, f"\nTotal time: {time.time() - start:.1f} s")
    write_summary(summary_lines, summary_path)
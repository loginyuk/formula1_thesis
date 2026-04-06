"""
run_model_no_clustering.py
───────────────
No clustering study: measures prediction improvement from the GMM clustering pipeline.

Runs the primary model twice:
  1. Full feature set (MODEL_FEATURES)
  2. Without clustering features (MODEL_FEATURES_NO_CLUSTER)

Saves comparison plots and a summary.

Run from project root:
    python scripts/run_model_no_clustering.py
"""

import os
import sys
import json
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    SUMMARIES_DIR, RESULTS_MODEL_DIR, DATASET_ALL,
    MODEL_FEATURES, MODEL_FEATURES_NO_CLUSTER, CLUSTER_PIPELINE_FEATURES,
    MIN_TRAIN_RACES, PRIMARY_MODEL, BEST_PARAMS_FILE, MODEL_DEFAULTS,
)
from src.utils import log, write_summary
from src.modeling.training import run_season_walk_forward, convert_deltas_to_absolute_times, shift_telemetry_features, compute_metrics
from src.modeling.plots import plot_no_clustering

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


def build_model(name, params):
    if name == "XGBoost":
        return XGBRegressor(**params, n_jobs=-1)
    if name == "LightGBM":
        return LGBMRegressor(**params, n_jobs=-1, verbose=-1)
    if name == "CatBoost":
        return CatBoostRegressor(**params, silent=True)
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
    summary_path = os.path.join(SUMMARIES_DIR, "summary_no_clustering.txt")
    out_dir = RESULTS_MODEL_DIR

    best_params = {}
    if os.path.exists(BEST_PARAMS_FILE):
        with open(BEST_PARAMS_FILE) as f:
            best_params = json.load(f)

    params = best_params.get(PRIMARY_MODEL, MODEL_DEFAULTS[PRIMARY_MODEL])
    source = "tuned" if PRIMARY_MODEL in best_params else "default"

    log(summary_lines, f"No clustering study — model: {PRIMARY_MODEL} ({source} params)")
    log(summary_lines, f"{len(CLUSTER_PIPELINE_FEATURES)} removed features: ")
    log(summary_lines, f"{CLUSTER_PIPELINE_FEATURES}")
    log(summary_lines, f"Full feature set:    {len(MODEL_FEATURES)} features")
    log(summary_lines, f"Reduced feature set: {len(MODEL_FEATURES_NO_CLUSTER)} features\n")

    df = pd.read_csv(DATASET_ALL)
    df = shift_telemetry_features(df)

    ablation_rows = []

    for label, features in [
        ("With clustering", MODEL_FEATURES),
        ("Without clustering", MODEL_FEATURES_NO_CLUSTER),
    ]:
        log(summary_lines, f"\n{'-'*40}")
        log(summary_lines, f"{label}  ({len(features)} features)")
        log(summary_lines, f"{'-'*40}")

        model = build_model(PRIMARY_MODEL, params)

        t0 = time.time()
        results = run_season_walk_forward(
            df, features, model, summary_lines,
            target='Target_Delta', min_train_races=MIN_TRAIN_RACES, print_progress=False
        )
        results = convert_deltas_to_absolute_times(results, df)
        elapsed = time.time() - t0

        metrics = compute_metrics(results['Actual'].values, results['Predicted'].values)
        log(summary_lines, f"  MAE:  {metrics['MAE']:.4f} s")
        log(summary_lines, f"  RMSE: {metrics['RMSE']:.4f} s")
        log(summary_lines, f"  R2:   {metrics['R2']:.6f}")
        log(summary_lines, f"  MAPE: {metrics['MAPE']:.3f} %")
        log(summary_lines, f"  Time: {elapsed:.1f} s")

        ablation_rows.append({'Features': label, **metrics})

    # calculate delta
    with_m = ablation_rows[0]
    without_m = ablation_rows[1]
    log(summary_lines, f"\n{'-'*40}")
    log(summary_lines, "Improvement from clustering features (negative = clustering helps)")
    log(summary_lines, f"{'-'*40}")
    log(summary_lines, f" MAE:  {without_m['MAE']  - with_m['MAE']:+.4f} s")
    log(summary_lines, f" RMSE: {without_m['RMSE'] - with_m['RMSE']:+.4f} s")
    log(summary_lines, f" R2:   {without_m['R2']   - with_m['R2']:+.6f}")

    plot_no_clustering(ablation_rows, out_dir=out_dir)

    log(summary_lines, f"\nTotal time: {time.time() - start_all:.1f} s")
    write_summary(summary_lines, summary_path)
    print(f"Saved to {out_dir}/model_no_clustering.png")
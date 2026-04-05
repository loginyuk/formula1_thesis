"""
run_hyperparameter_tuning.py
────────────────────────────
Hyperparameter tuning for top boosting models using Optuna.

Tuning using TimeSeriesSplit without data leakage with full walk-forward as final evaluation.

Run from project root:
    python scripts/run_hyperparameter_tuning.py
    python scripts/run_hyperparameter_tuning.py --trials 50
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SUMMARIES_DIR, RESULTS_MODEL_DIR, MODEL_FEATURES, DATASET_ALL, MIN_TRAIN_RACES, CV_N_SPLITS, CV_N_JOBS
from src.utils import log, write_summary
from src.modeling.training import run_season_walk_forward, convert_deltas_to_absolute_times, shift_telemetry_features, compute_metrics
from src.modeling.analysis import plot_model_comparison



def make_objective(model_name, X, y):
    tscv = TimeSeriesSplit(n_splits=CV_N_SPLITS)

    def objective(trial):
        if model_name == "RandomForest":
            model = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('model', RandomForestRegressor(
                    n_estimators     = trial.suggest_int("n_estimators", 50, 300),
                    max_depth        = trial.suggest_int("max_depth", 5, 20),
                    max_features     = trial.suggest_float("max_features", 0.1, 0.8),
                    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20),
                    n_jobs=-1, random_state=42,
                )),
            ])
        elif model_name == "XGBoost":
            model = XGBRegressor(
                n_estimators     = trial.suggest_int("n_estimators", 50, 400),
                max_depth        = trial.suggest_int("max_depth", 3, 8),
                learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample        = trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
                min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
                n_jobs=-1, random_state=42, verbosity=0,
            )
        elif model_name == "LightGBM":
            model = LGBMRegressor(
                n_estimators      = trial.suggest_int("n_estimators", 50, 400),
                max_depth         = trial.suggest_int("max_depth", 3, 8),
                num_leaves        = trial.suggest_int("num_leaves", 15, 127),
                learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample         = trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
                min_child_samples = trial.suggest_int("min_child_samples", 5, 50),
                n_jobs=-1, random_state=42, verbose=-1,
            )
        elif model_name == "CatBoost":
            model = CatBoostRegressor(
                iterations    = trial.suggest_int("iterations", 50, 400),
                depth         = trial.suggest_int("depth", 4, 8),
                learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                l2_leaf_reg   = trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                random_state=42, verbose=0,
            )

        scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=CV_N_JOBS)
        return -scores.mean()

    return objective


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=50)
    args = parser.parse_args()

    start_all = time.time()
    summary_lines = []
    summary_path = os.path.join(SUMMARIES_DIR, "summary_hyperparameter_tuning.txt")

    df = pd.read_csv(DATASET_ALL)
    df = shift_telemetry_features(df)

    df_sorted = df.sort_values(['Year', 'RoundNumber']).reset_index(drop=True)
    X = df_sorted[MODEL_FEATURES].values
    y = df_sorted['Target_Delta'].values

    MODELS_TO_TUNE = ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]

    best_params_all = {}
    comparison_rows = []

    for model_name in MODELS_TO_TUNE:
        log(summary_lines, f"\n{'-'*45}")
        log(summary_lines, f"Tuning: {model_name} ({args.trials} trials)")
        log(summary_lines, f"{'-'*45}")

        study = optuna.create_study(direction='minimize')
        t0 = time.time()
        study.optimize(make_objective(model_name, X, y), n_trials=args.trials, show_progress_bar=True)
        tune_time = time.time() - t0

        best_params = study.best_params
        best_params_all[model_name] = best_params
        log(summary_lines, f"Tuning completed in {tune_time:.1f} s")
        log(summary_lines, f"Best MAE: {study.best_value:.4f} s")
        log(summary_lines, f"Best params: {json.dumps(best_params, indent=2)}")

        # final walk-forward with best params
        log(summary_lines, f"\nFinal walk-forward validation with best params")
        if model_name == "RandomForest":
            model = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('model', RandomForestRegressor(**best_params, n_jobs=-1, random_state=42)),
            ])
        elif model_name == "XGBoost":
            model = XGBRegressor(**best_params, n_jobs=-1, random_state=42, verbosity=0)
        elif model_name == "LightGBM":
            model = LGBMRegressor(**best_params, n_jobs=-1, random_state=42, verbose=-1)
        elif model_name == "CatBoost":
            model = CatBoostRegressor(**best_params, random_state=42, verbose=0)

        t1 = time.time()
        results = run_season_walk_forward(
            df, MODEL_FEATURES, model, summary_lines,
            target='Target_Delta', min_train_races=MIN_TRAIN_RACES, print_progress=False
        )
        results = convert_deltas_to_absolute_times(results, df)
        wf_time = time.time() - t1

        metrics = compute_metrics(results['Actual'].values, results['Predicted'].values)
        log(summary_lines, f"Walk-forward MAE:  {metrics['MAE']:.3f} s")
        log(summary_lines, f"Walk-forward RMSE: {metrics['RMSE']:.3f} s")
        log(summary_lines, f"Walk-forward R2:   {metrics['R2']:.4f}")
        log(summary_lines, f"Walk-forward MAPE: {metrics['MAPE']:.2f} %")
        log(summary_lines, f"Walk-forward time: {wf_time:.1f} s")

        comparison_rows.append({'Model': f"{model_name} (tuned)", **metrics, 'Time_s': wf_time})

    # save all best params
    params_path = os.path.join(RESULTS_MODEL_DIR, "best_params.json")
    with open(params_path, 'w') as f:
        json.dump(best_params_all, f, indent=2)
    log(summary_lines, f"\nBest params saved to {params_path}")

    log(summary_lines, f"\n{'-'*50}")
    log(summary_lines, "Tuned Model Comparison")
    log(summary_lines, f"{'-'*50}")
    comp_df = plot_model_comparison(comparison_rows, out_dir=RESULTS_MODEL_DIR)
    log(summary_lines, comp_df.to_string(index=False))
    log(summary_lines, f"\nTotal time: {time.time() - start_all:.1f} s")

    write_summary(summary_lines, summary_path)

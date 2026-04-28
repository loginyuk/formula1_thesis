import os
import sys
import json
import time
import argparse
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
    RESULTS_MODEL_DIR, MODEL_FEATURES, DATASET_ALL,
    MIN_TRAIN_RACES, PRIMARY_MODEL, BEST_PARAMS_FILE, MODEL_DEFAULTS,
)
from src.utils import log, write_summary
from src.modeling.training import run_season_walk_forward, convert_deltas_to_absolute_times, shift_telemetry_features, compute_metrics
from src.modeling.analysis_plots import plot_feature_importance
from src.modeling.plots import (
    plot_full_season_slopes, plot_pred_vs_act_errors,
    plot_per_race_mae, plot_compound_breakdown, plot_driver_mae
)


def build_model(name, params):
    """
    Loads a model with the given name and parameters
    """
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-shift', action='store_true')
    args = parser.parse_args()

    no_shift = args.no_shift

    start = time.time()
    summary_lines = []
    if no_shift:
        out_dir = os.path.join(RESULTS_MODEL_DIR, PRIMARY_MODEL, "no_shift")
    else:
        out_dir = os.path.join(RESULTS_MODEL_DIR, PRIMARY_MODEL)
        
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.txt")

    if no_shift:
        log(summary_lines, f"No shift telemetry features experiment\n")

    # load best params if available
    best_params = {}
    if os.path.exists(BEST_PARAMS_FILE):
        with open(BEST_PARAMS_FILE) as f:
            best_params = json.load(f)

    params = best_params.get(PRIMARY_MODEL, MODEL_DEFAULTS[PRIMARY_MODEL])
    source = "tuned" if PRIMARY_MODEL in best_params else "default"

    log(summary_lines, f"Model: {PRIMARY_MODEL} ({source} params)")
    log(summary_lines, f"Params: {params}\n")

    model = build_model(PRIMARY_MODEL, params)

    # load data
    df = pd.read_csv(DATASET_ALL)
    if not no_shift:
        df = shift_telemetry_features(df)

    log(summary_lines, f"Features ({len(MODEL_FEATURES)}): {MODEL_FEATURES}")

    # run walk-forward validation
    simulation_results = run_season_walk_forward(
        df, MODEL_FEATURES, model, summary_lines,
        target='Target_Delta', min_train_races=MIN_TRAIN_RACES, print_progress=True
    )
    simulation_results = convert_deltas_to_absolute_times(simulation_results, df)

    metrics = compute_metrics(simulation_results['Actual'].values, simulation_results['Predicted'].values)
    log(summary_lines, f"\nMetrics:")
    log(summary_lines, f"MAE:  {metrics['MAE']:.3f} s")
    log(summary_lines, f"RMSE: {metrics['RMSE']:.3f} s")
    log(summary_lines, f"R2:   {metrics['R2']:.4f}")
    log(summary_lines, f"MAPE: {metrics['MAPE']:.2f} %")

    plot_feature_importance(df, MODEL_FEATURES, model, out_dir=out_dir)
    plot_full_season_slopes(simulation_results, 'VER', out_dir=out_dir)
    plot_pred_vs_act_errors(simulation_results, out_dir=out_dir)
    plot_per_race_mae(simulation_results, out_dir=out_dir)
    plot_compound_breakdown(simulation_results, out_dir=out_dir)
    plot_driver_mae(simulation_results, out_dir=out_dir)

    # calculate delta
    if no_shift:
        baseline_path = os.path.join(RESULTS_MODEL_DIR, "model_comparison.csv")
        if os.path.exists(baseline_path):
            baseline = pd.read_csv(baseline_path)
            row = baseline[baseline['Model'] == PRIMARY_MODEL]
            if not row.empty:
                shift_m = row.iloc[0]
                delta = pd.DataFrame([
                    {'Metric': 'MAE',  'Shift': round(shift_m['MAE'], 4),  'NoShift': round(metrics['MAE'], 4),  'Delta': round(shift_m['MAE']  - metrics['MAE'], 4)},
                    {'Metric': 'RMSE', 'Shift': round(shift_m['RMSE'], 4), 'NoShift': round(metrics['RMSE'], 4), 'Delta': round(shift_m['RMSE'] - metrics['RMSE'], 4)},
                    {'Metric': 'R2',   'Shift': round(shift_m['R2'], 6),   'NoShift': round(metrics['R2'], 6),   'Delta': round(metrics['R2'] - shift_m['R2'], 6)},
                    {'Metric': 'MAPE', 'Shift': round(shift_m['MAPE'], 4), 'NoShift': round(metrics['MAPE'], 4), 'Delta': round(shift_m['MAPE'] - metrics['MAPE'], 4)},
                ])
                delta_path = os.path.join(out_dir, "shift_vs_no_shift.csv")
                delta.to_csv(delta_path, index=False)

                log(summary_lines, f"\n{'-' * 40}")
                log(summary_lines, "Delta:")
                log(summary_lines, f"{'-' * 40}")
                log(summary_lines, delta.to_string(index=False))
            else:
                log(summary_lines, f"\n{PRIMARY_MODEL} row not found in {baseline_path}, skipping delta")
        else:
            log(summary_lines, f"\nNo {baseline_path} found, skipping delta")

    log(summary_lines, f"\nTotal time: {time.time() - start:.1f} s")
    write_summary(summary_lines, summary_path)
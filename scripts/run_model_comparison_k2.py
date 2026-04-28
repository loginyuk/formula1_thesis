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
    SUMMARIES_DIR, RESULTS_MODEL_DIR, MODEL_FEATURES_K2, DATASET_ALL,
    MIN_TRAIN_RACES, BEST_PARAMS_FILE, MODEL_DEFAULTS, MODELS_TO_COMPARE,
)
from src.utils import log, write_summary
from src.modeling.training import (
    run_season_walk_forward, convert_deltas_to_absolute_times,
    shift_telemetry_features, compute_metrics,
)
from src.modeling.plots import (
    plot_full_season_slopes, plot_pred_vs_act_errors,
    plot_per_race_mae, plot_compound_breakdown, plot_driver_mae,
)
from src.modeling.analysis_plots import plot_feature_importance, plot_model_comparison

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
    start_all = time.time()
    summary_lines = []
    summary_path = os.path.join(SUMMARIES_DIR, "summary_model_comparison_k2.txt")

    k2_root = os.path.join(RESULTS_MODEL_DIR, "k2")
    os.makedirs(k2_root, exist_ok=True)

    best_params = {}
    if os.path.exists(BEST_PARAMS_FILE):
        with open(BEST_PARAMS_FILE) as f:
            best_params = json.load(f)

    # load data with K=2 features
    df = pd.read_csv(DATASET_ALL)

    log(summary_lines, "Model Comparison (K=2 cluster features)")

    missing_k2 = [c for c in ['P_0_k2', 'P_1_k2', 'Style_Cluster_ID_k2', 'Style_Entropy_k2'] if c not in df.columns]
    if missing_k2:
        log(summary_lines, f"K=2 columns missing from dataset: {missing_k2}")
        write_summary(summary_lines, summary_path)
        sys.exit(1)

    before = len(df)
    df = df.dropna(subset=['P_0_k2', 'P_1_k2', 'Style_Cluster_ID_k2', 'Style_Entropy_k2']).reset_index(drop=True)
    if len(df) < before:
        log(summary_lines, f"Dropped {before - len(df)} laps with NaN in K=2 columns")

    df = shift_telemetry_features(df)

    comparison_rows = []

    for name in MODELS_TO_COMPARE:
        params = best_params.get(name, MODEL_DEFAULTS[name])
        source = "tuned" if name in best_params else "default"

        log(summary_lines, f"\n{'-' * 40}")
        log(summary_lines, f"Model: {name} ({source} params)")
        log(summary_lines, f"{'-' * 40}")

        # build model with parameters and run walk-forward validation
        model = build_model(name, params)

        t0 = time.time()
        results = run_season_walk_forward(
            df, MODEL_FEATURES_K2, model, summary_lines,
            target='Target_Delta', min_train_races=MIN_TRAIN_RACES,
            print_progress=False
        )
        results = convert_deltas_to_absolute_times(results, df)
        elapsed = time.time() - t0

        metrics = compute_metrics(results['Actual'].values, results['Predicted'].values)
        log(summary_lines, "Metrics:")
        log(summary_lines, f"  MAE:  {metrics['MAE']:.3f} s")
        log(summary_lines, f"  RMSE: {metrics['RMSE']:.3f} s")
        log(summary_lines, f"  R2:   {metrics['R2']:.4f}")
        log(summary_lines, f"  MAPE: {metrics['MAPE']:.2f} %")
        log(summary_lines, f"  Time: {elapsed:.1f} s")

        comparison_rows.append({'Model': name, **metrics, 'Time_s': elapsed})

        model_dir = os.path.join(RESULTS_MODEL_DIR, name, "k2")
        os.makedirs(model_dir, exist_ok=True)
        results.to_csv(os.path.join(model_dir, f"results_{name}.csv"), index=False)
        plot_feature_importance(df, MODEL_FEATURES_K2, model, out_dir=model_dir)
        plot_full_season_slopes(results, 'VER', out_dir=model_dir)
        plot_pred_vs_act_errors(results, out_dir=model_dir)
        plot_per_race_mae(results, out_dir=model_dir)
        plot_compound_breakdown(results, out_dir=model_dir)
        plot_driver_mae(results, out_dir=model_dir)

    # K=2 comparison plot + csv
    log(summary_lines, f"\n{'-' * 40}")
    log(summary_lines, "K=2 Summary:")
    log(summary_lines, f"{'-' * 40}")
    comp_k2 = plot_model_comparison(comparison_rows, out_dir=k2_root)
    log(summary_lines, comp_k2.to_string(index=False))

    # compare with K=3 results
    k3_path = os.path.join(RESULTS_MODEL_DIR, "model_comparison.csv")
    if os.path.exists(k3_path):
        comp_k3 = pd.read_csv(k3_path)

        merged = comp_k2.merge(comp_k3, on='Model', suffixes=('_k2', '_k3'))
        delta = pd.DataFrame({
            'Model': merged['Model'],
            'MAE_k3': merged['MAE_k3'].round(4),
            'MAE_k2': merged['MAE_k2'].round(4),
            'MAE_delta': (merged['MAE_k3'] - merged['MAE_k2']).round(4),
            'RMSE_k3': merged['RMSE_k3'].round(4),
            'RMSE_k2': merged['RMSE_k2'].round(4),
            'RMSE_delta': (merged['RMSE_k3'] - merged['RMSE_k2']).round(4),
            'R2_k3': merged['R2_k3'].round(6),
            'R2_k2': merged['R2_k2'].round(6),
            'R2_delta': (merged['R2_k2'] - merged['R2_k3']).round(6),
        })
        delta_path = os.path.join(RESULTS_MODEL_DIR, "k2_vs_k3_delta.csv")
        delta.to_csv(delta_path, index=False)

        log(summary_lines, f"\n{'-' * 40}")
        log(summary_lines, "K=2 vs K=3 delta (K=3 - K=2):")
        log(summary_lines, f"{'-' * 40}")
        log(summary_lines, delta.to_string(index=False))
    else:
        log(summary_lines, f"\nNo K=3 results found")

    log(summary_lines, f"\nTotal time: {time.time() - start_all:.1f} s")
    write_summary(summary_lines, summary_path)

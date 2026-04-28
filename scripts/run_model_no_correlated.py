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
    RESULTS_MODEL_DIR, RESULTS_CORRELATION_DIR, MODEL_FEATURES, DATASET_ALL,
    MIN_TRAIN_RACES, PRIMARY_MODEL, BEST_PARAMS_FILE, MODEL_DEFAULTS,
)
from src.utils import log, write_summary
from src.modeling.training import (
    run_season_walk_forward, convert_deltas_to_absolute_times,
    shift_telemetry_features, compute_metrics,
)
from src.modeling.analysis_plots import plot_feature_importance

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


def select_features_to_drop(pairs_df, all_features):
    """
    Selects which features to drop based on the correlation pairs.
    drops one of each pair
    """
    kept = set(all_features)
    dropped = []
    for _, row in pairs_df.sort_values('abs_r', ascending=False).iterrows():
        a, b = row['Feature_A'], row['Feature_B']
        if a in kept and b in kept:
            kept.discard(b)
            dropped.append((b, a, row['abs_r']))
    return dropped


if __name__ == "__main__":
    start = time.time()
    summary_lines = []
    out_dir = os.path.join(RESULTS_MODEL_DIR, PRIMARY_MODEL, "no_correlated")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.txt")

    # load best params if available
    best_params = {}
    if os.path.exists(BEST_PARAMS_FILE):
        with open(BEST_PARAMS_FILE) as f:
            best_params = json.load(f)
    params = best_params.get(PRIMARY_MODEL, MODEL_DEFAULTS[PRIMARY_MODEL])
    source = "tuned" if PRIMARY_MODEL in best_params else "default"

    log(summary_lines, f"No-correlated-features")
    log(summary_lines, f"Model: {PRIMARY_MODEL} ({source} params)")

    # load high correlation pairs
    pairs_path = os.path.join(RESULTS_CORRELATION_DIR, "high_correlation_pairs.csv")
    if not os.path.exists(pairs_path):
        log(summary_lines, f"Missing {pairs_path}")
        write_summary(summary_lines, summary_path)
        sys.exit(1)

    pairs_df = pd.read_csv(pairs_path)
    pairs_in_features = pairs_df[
        pairs_df['Feature_A'].isin(MODEL_FEATURES) & pairs_df['Feature_B'].isin(MODEL_FEATURES)
    ].reset_index(drop=True)
    log(summary_lines, f"High correlation pairs: {len(pairs_in_features)}")

    # select features to drop
    dropped = select_features_to_drop(pairs_in_features, MODEL_FEATURES)
    dropped_set = {d[0] for d in dropped}

    log(summary_lines, f"\nDropped {len(dropped)} features:")
    for feat, kept_with, r in dropped:
        log(summary_lines, f"  {feat}  (correlated with {kept_with}, |r|={r:.3f})")

    features = [f for f in MODEL_FEATURES if f not in dropped_set]
    log(summary_lines, f"\nFeatures used: {len(features)} / {len(MODEL_FEATURES)}\n")

    # load data
    df = pd.read_csv(DATASET_ALL)
    df = shift_telemetry_features(df)

    # run walk-forward validation with selected features
    model = build_model(PRIMARY_MODEL, params)
    results = run_season_walk_forward(
        df, features, model, summary_lines,
        target='Target_Delta', min_train_races=MIN_TRAIN_RACES, print_progress=False
    )
    results = convert_deltas_to_absolute_times(results, df)

    metrics = compute_metrics(results['Actual'].values, results['Predicted'].values)
    log(summary_lines, f"\nMetrics:")
    log(summary_lines, f"  MAE:  {metrics['MAE']:.3f} s")
    log(summary_lines, f"  RMSE: {metrics['RMSE']:.3f} s")
    log(summary_lines, f"  R2:   {metrics['R2']:.4f}")
    log(summary_lines, f"  MAPE: {metrics['MAPE']:.2f} %")

    results.to_csv(os.path.join(out_dir, f"results_{PRIMARY_MODEL}.csv"), index=False)
    plot_feature_importance(df, features, model, out_dir=out_dir)

    # calculate delta
    full_path = os.path.join(RESULTS_MODEL_DIR, "model_comparison.csv")
    if os.path.exists(full_path):
        full_df = pd.read_csv(full_path)
        full_row = full_df[full_df['Model'] == PRIMARY_MODEL]
        if not full_row.empty:
            full_m = full_row.iloc[0]
            delta = pd.DataFrame([
                {'Metric': 'MAE',  'Full': round(full_m['MAE'], 4),  'NoCorrelated': round(metrics['MAE'], 4),  'Delta': round(full_m['MAE']  - metrics['MAE'], 4)},
                {'Metric': 'RMSE', 'Full': round(full_m['RMSE'], 4), 'NoCorrelated': round(metrics['RMSE'], 4), 'Delta': round(full_m['RMSE'] - metrics['RMSE'], 4)},
                {'Metric': 'R2',   'Full': round(full_m['R2'], 6),   'NoCorrelated': round(metrics['R2'], 6),   'Delta': round(metrics['R2'] - full_m['R2'], 6)},
                {'Metric': 'MAPE', 'Full': round(full_m['MAPE'], 4), 'NoCorrelated': round(metrics['MAPE'], 4), 'Delta': round(full_m['MAPE'] - metrics['MAPE'], 4)},
            ])
            delta_path = os.path.join(out_dir, "comparison_full_vs_no_correlated.csv")
            delta.to_csv(delta_path, index=False)

            log(summary_lines, f"\n{'-' * 40}")
            log(summary_lines, f"Delta vs full {PRIMARY_MODEL} (positive = no_correlated is better):")
            log(summary_lines, f"{'-' * 40}")
            log(summary_lines, delta.to_string(index=False))
        else:
            log(summary_lines, f"\n{PRIMARY_MODEL} row not found in {full_path}, skipping delta")
    else:
        log(summary_lines, f"\nNo {full_path} found, skipping delta")

    log(summary_lines, f"\nTotal time: {time.time() - start:.1f} s")
    write_summary(summary_lines, summary_path)

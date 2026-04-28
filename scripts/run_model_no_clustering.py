import os
import sys
import json
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    RESULTS_MODEL_DIR, DATASET_ALL,
    MODEL_FEATURES, MODEL_FEATURES_NO_CLUSTER, CLUSTER_PIPELINE_FEATURES,
    MIN_TRAIN_RACES, BEST_PARAMS_FILE, MODEL_DEFAULTS, MODELS_TO_COMPARE,
)
from src.utils import log, write_summary
from src.modeling.training import run_season_walk_forward, convert_deltas_to_absolute_times, shift_telemetry_features, compute_metrics

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

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


def run_ablation(name, params, df, summary_lines):
    """
    Runs walk-forward validation for a single model.
    compares with vs without clustering features
    """
    rows = []
    for label, features in [
        ("With clustering", MODEL_FEATURES),
        ("Without clustering", MODEL_FEATURES_NO_CLUSTER),
    ]:
        log(summary_lines, f"\n{'-' * 40}")
        log(summary_lines, f"{label}  ({len(features)} features)")
        log(summary_lines, f"{'-' * 40}")

        model = build_model(name, params)

        t0 = time.time()
        results = run_season_walk_forward(
            df, features, model, summary_lines,
            target='Target_Delta', min_train_races=MIN_TRAIN_RACES, print_progress=False
        )
        results = convert_deltas_to_absolute_times(results, df)
        elapsed = time.time() - t0

        metrics = compute_metrics(results['Actual'].values, results['Predicted'].values)
        log(summary_lines, f"Metrics:")
        log(summary_lines, f"  MAE:  {metrics['MAE']:.4f} s")
        log(summary_lines, f"  RMSE: {metrics['RMSE']:.4f} s")
        log(summary_lines, f"  R2:   {metrics['R2']:.6f}")
        log(summary_lines, f"  MAPE: {metrics['MAPE']:.3f} %")
        log(summary_lines, f"  Time: {elapsed:.1f} s")

        rows.append({'Features': label, **metrics, 'Time_s': elapsed})

    return rows


if __name__ == "__main__":
    start_all = time.time()

    best_params = {}
    if os.path.exists(BEST_PARAMS_FILE):
        with open(BEST_PARAMS_FILE) as f:
            best_params = json.load(f)

    df = pd.read_csv(DATASET_ALL)
    df = shift_telemetry_features(df)

    all_rows = []

    for name in MODELS_TO_COMPARE:
        params = best_params.get(name, MODEL_DEFAULTS[name])
        source = "tuned" if name in best_params else "default"

        out_dir = os.path.join(RESULTS_MODEL_DIR, name, "clustering_ablation")
        os.makedirs(out_dir, exist_ok=True)
        summary_path = os.path.join(out_dir, "summary.txt")
        summary_lines = []

        log(summary_lines, f"Clustering ablation")
        log(summary_lines, f"Model: {name} ({source} params)")
        log(summary_lines, f"Features: {len(MODEL_FEATURES_NO_CLUSTER)}")
        log(summary_lines, f"{len(CLUSTER_PIPELINE_FEATURES)} removed features: {CLUSTER_PIPELINE_FEATURES}")

        # run ablation and save per-model results
        rows = run_ablation(name, params, df, summary_lines)

        # calculate delta
        with_m, without_m = rows[0], rows[1]
        delta_row = {
            'Features': 'Delta (with - without)',
            'MAE':  with_m['MAE']  - without_m['MAE'],
            'RMSE': with_m['RMSE'] - without_m['RMSE'],
            'R2':   with_m['R2']   - without_m['R2'],
            'MAPE': with_m['MAPE'] - without_m['MAPE'],
            'Time_s': with_m['Time_s'] - without_m['Time_s'],
        }
        rows.append(delta_row)

        log(summary_lines, f"\n{'-' * 40}")
        log(summary_lines, "Delta (with clustering - without clustering)")
        log(summary_lines, f"{'-' * 40}")
        log(summary_lines, f"  MAE:  {delta_row['MAE']:+.4f} s")
        log(summary_lines, f"  RMSE: {delta_row['RMSE']:+.4f} s")
        log(summary_lines, f"  R2:   {delta_row['R2']:+.6f}")

        csv_path = os.path.join(out_dir, "ablation_clustering.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        write_summary(summary_lines, summary_path)

        for r in rows:
            all_rows.append({'Model': name, **r})

    combined_path = os.path.join(RESULTS_MODEL_DIR, "ablation_clustering_all.csv")
    combined_df = pd.DataFrame(all_rows)
    combined_df.to_csv(combined_path, index=False)
    print(f"\nCombined ablation results saved to {combined_path}")
    print(combined_df.to_string(index=False))
    print(f"\nTotal time: {time.time() - start_all:.1f} s")

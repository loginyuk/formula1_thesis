import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    RESULTS_MODEL_DIR, MODEL_FEATURES, DATASET_ALL,
    MIN_TRAIN_RACES, BEST_PARAMS_FILE, MODEL_DEFAULTS, FEATURE_BUCKETS,
)
from src.utils import log, write_summary
from src.modeling.training import shift_telemetry_features
from src.modeling.shap_plots import plot_shap_global, plot_shap_by_lag, plot_shap_over_races

TOP_N_GLOBAL = 20
SAMPLE_PER_RACE = 400


def build_model(name, params):
    if name == "LightGBM":
        return LGBMRegressor(**params, n_jobs=-1, verbose=-1)
    if name == "XGBoost":
        return XGBRegressor(**params, n_jobs=-1)
    if name == "CatBoost":
        return CatBoostRegressor(**params, silent=True)
    raise ValueError(f"Unsupported model: {name}")


def bucket_for(feature):
    """
    Determines which feature bucket a feature belongs to
    """
    for bucket, feats in FEATURE_BUCKETS.items():
        if feature in feats:
            return bucket
    return 'Static'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='LightGBM', choices=['LightGBM', 'XGBoost', 'CatBoost'])
    args = parser.parse_args()
    model_name = args.model

    start = time.time()
    summary_lines = []

    out_dir = os.path.join(RESULTS_MODEL_DIR, model_name, "shap")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.txt")

    log(summary_lines, f"SHAP analysis\n")

    # load tuned params
    best_params = {}
    if os.path.exists(BEST_PARAMS_FILE):
        with open(BEST_PARAMS_FILE) as f:
            best_params = json.load(f)
    params = best_params.get(model_name, MODEL_DEFAULTS[model_name])
    source = "tuned" if model_name in best_params else "default"

    log(summary_lines, f"Model: {model_name} ({source} params)")
    log(summary_lines, f"Features: {len(MODEL_FEATURES)}")

    df = pd.read_csv(DATASET_ALL)
    df = shift_telemetry_features(df)
    df = df.sort_values(by=['Year', 'RoundNumber']).reset_index(drop=True)
    df['RaceKey'] = df['Year'].astype(str) + '_' + df['Location']
    races = df['RaceKey'].unique()

    log(summary_lines, f"Races: {len(races)}  (initial train: {MIN_TRAIN_RACES})")

    # accumulators
    global_abs_shap = np.zeros(len(MODEL_FEATURES))
    n_total_rows = 0
    per_race_rows = []

    rng = np.random.default_rng(42)

    # run walk-forward and accumulate SHAP values
    for i in range(MIN_TRAIN_RACES, len(races)):
        train_races = races[:i]
        test_race = races[i]

        train_data = df[df['RaceKey'].isin(train_races)]
        test_data  = df[df['RaceKey'] == test_race]

        X_train = train_data[MODEL_FEATURES]
        y_train = train_data['Target_Delta']
        X_test  = test_data[MODEL_FEATURES]

        if len(X_test) > SAMPLE_PER_RACE:
            idx = rng.choice(len(X_test), size=SAMPLE_PER_RACE, replace=False)
            X_test_sample = X_test.iloc[idx]
        else:
            X_test_sample = X_test

        # fit model and compute SHAP values
        model = build_model(model_name, params)
        model.fit(X_train, y_train)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_sample)

        abs_shap = np.abs(shap_values)
        race_mean = abs_shap.mean(axis=0)

        global_abs_shap += abs_shap.sum(axis=0)
        n_total_rows += len(X_test_sample)

        # save per-race SHAP values
        for feat, v in zip(MODEL_FEATURES, race_mean):
            per_race_rows.append({'Race': test_race, 'Feature': feat, 'Mean_Abs_SHAP': v})

        print(f"[{i - MIN_TRAIN_RACES + 1}/{len(races) - MIN_TRAIN_RACES}] {test_race}  "
              f"laps={len(X_test_sample):>4}")

    df.drop(columns=['RaceKey'], inplace=True)

    # calculate global mean |SHAP| importance
    global_mean = global_abs_shap / n_total_rows
    global_df = pd.DataFrame({
        'Feature': MODEL_FEATURES,
        'Mean_Abs_SHAP': global_mean,
    }).sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)

    global_df.to_csv(os.path.join(out_dir, "shap_global.csv"), index=False)
    log(summary_lines, f"\nTop {TOP_N_GLOBAL} features by mean |SHAP|:")
    log(summary_lines, global_df.head(TOP_N_GLOBAL).to_string(index=False))

    plot_shap_global(global_df, out_dir, top_n=TOP_N_GLOBAL)

    # group by lag bucket
    global_df['Bucket'] = global_df['Feature'].apply(bucket_for)
    bucket_df = (
        global_df.groupby('Bucket')['Mean_Abs_SHAP']
        .agg(['sum', 'mean', 'count'])
        .rename(columns={'sum': 'Total_Abs_SHAP', 'mean': 'Mean_Abs_SHAP', 'count': 'N_Features'})
        .sort_values('Total_Abs_SHAP', ascending=False)
        .reset_index()
    )
    bucket_df.to_csv(os.path.join(out_dir, "shap_by_lag.csv"), index=False)
    log(summary_lines, f"\nBy lag bucket:")
    log(summary_lines, bucket_df.to_string(index=False))

    plot_shap_by_lag(bucket_df, out_dir)

    # save per-race SHAP values and plot over time
    per_race_df = pd.DataFrame(per_race_rows)
    per_race_df.to_csv(os.path.join(out_dir, "shap_over_races.csv"), index=False)

    plot_shap_over_races(per_race_df, global_df, races, out_dir, top_n=8)

    log(summary_lines, f"\nTotal time: {time.time() - start:.1f} s")
    write_summary(summary_lines, summary_path)
    print(f"\nDone. Output in {out_dir}")

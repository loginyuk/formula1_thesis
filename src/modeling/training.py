import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils import log
from src.config import TELEMETRY_FEATURES_TO_SHIFT


def compute_metrics(actual, predicted):
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2   = r2_score(actual, predicted)
    mask = np.abs(actual) > 1e-6
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100 if mask.any() else 0.0
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}


def convert_deltas_to_absolute_times(simulation_results, df_with_telemetry):
    """
    Converts deltas time back to absolute lap times
    """
    simulation_results = simulation_results.merge(
        df_with_telemetry[['Year', 'Location', 'Driver', 'LapNumber', 'Prev_LapTime']],
        on=['Year', 'Location', 'Driver', 'LapNumber'], how='left')

    simulation_results['Predicted_Time'] = simulation_results['Prev_LapTime'] + simulation_results['Predicted']
    simulation_results['Actual_Time'] = simulation_results['Prev_LapTime'] + simulation_results['Actual']

    simulation_results['Predicted'] = simulation_results['Predicted_Time']
    simulation_results['Actual'] = simulation_results['Actual_Time']

    simulation_results.drop(columns=['Predicted_Time', 'Actual_Time', 'Prev_LapTime'], inplace=True)

    return simulation_results


def shift_telemetry_features(df, features_to_shift=None):
    """
    Shifts telemetry features by 1 lap per driver/stint to remove data leakage.
    Returns the shifted dataframe with rows that have NaN in shifted columns dropped.
    """
    if features_to_shift is None:
        features_to_shift = TELEMETRY_FEATURES_TO_SHIFT

    df = df.sort_values(by=['Location', 'Driver', 'LapNumber'])
    to_shift = [f for f in features_to_shift if f in df.columns]
    for feat in to_shift:
        df[feat] = df.groupby(['Location', 'Driver', 'Stint'])[feat].shift(1)
    df = df.dropna(subset=to_shift).reset_index(drop=True)
    print(f"Shifted {len(to_shift)} telemetry features by 1 lap. Rows remaining: {len(df)}")
    return df


def run_season_walk_forward(df, features, model, summary_lines, target='LapTime_Sec', min_train_races=5, print_progress=True):
    start_time = time.time()

    df = df.sort_values(by=['Year', 'RoundNumber']).reset_index(drop=True)

    df['_RaceKey'] = df['Year'].astype(str) + '_' + df['Location']
    races = df['_RaceKey'].unique()
    predictions_log = []

    log(summary_lines, f"Total races in dataset: {len(races)}")
    log(summary_lines, f"Initial training on first {min_train_races} races\n")

    for i in range(min_train_races, len(races)):
        train_races = races[:i]
        test_race = races[i]

        train_data = df[df['_RaceKey'].isin(train_races)]
        test_data = df[df['_RaceKey'] == test_race].copy()

        X_train = train_data[features]
        y_train = train_data[target]

        X_test = test_data[features]
        y_actual = test_data[target].values

        current_model = clone(model)
        current_model.fit(X_train, y_train)

        preds = current_model.predict(X_test)

        test_data['Predicted'] = preds
        test_data['Actual'] = y_actual
        test_data['Error'] = np.abs(y_actual - preds)

        predictions_log.append(test_data[['Year', 'RoundNumber', 'Location', 'Location_Encoded', 'Driver', 'LapNumber', 'Stint', 'Compound', 'Actual', 'Predicted', 'Error']])

        if print_progress:
            race_mae = mean_absolute_error(y_actual, preds)
            log(summary_lines, f"Tested on {test_race} | Train size: {len(train_data)} laps | MAE: {race_mae:.3f} s")

    results_df = pd.concat(predictions_log, ignore_index=True)
    df.drop(columns=['_RaceKey'], inplace=True)

    mae  = mean_absolute_error(results_df['Actual'], results_df['Predicted'])
    rmse = np.sqrt(mean_squared_error(results_df['Actual'], results_df['Predicted']))

    end_time = time.time()
    if print_progress:
        log(summary_lines, f"\n{'-'*30}")
        log(summary_lines, f"Total Test Laps Predicted: {len(results_df)}")
        log(summary_lines, f"Global MAE:  {mae:.3f} s")
        log(summary_lines, f"Global RMSE: {rmse:.3f} s")
        log(summary_lines, f"Time Taken:  {end_time - start_time:.1f} seconds")

    return results_df

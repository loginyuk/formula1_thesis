import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import matplotlib.pyplot as plt
import seaborn as sns
import math

def convert_deltas_to_absolute_times(simulation_results, df_with_telemetry):
    """
    Converts deltas time back to absolute lap times
    """
    simulation_results = simulation_results.merge(
        df_with_telemetry[['Location', 'Driver', 'LapNumber', 'Prev_LapTime']], 
        on=['Location', 'Driver', 'LapNumber'], how='left')

    simulation_results['Predicted_Time'] = simulation_results['Prev_LapTime'] + simulation_results['Predicted']
    simulation_results['Actual_Time'] = simulation_results['Prev_LapTime'] + simulation_results['Actual']

    simulation_results['Predicted'] = simulation_results['Predicted_Time']
    simulation_results['Actual'] = simulation_results['Actual_Time']
    
    simulation_results.drop(columns=['Predicted_Time', 'Actual_Time', 'Prev_LapTime'], inplace=True)
    
    return simulation_results


def feature_importance_walk_forward_delta(df, features):
    audit_model = XGBRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    audit_model.fit(df[features], df['Target_Delta'])

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': audit_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=importance_df.head(20), x='Importance', y='Feature', color='teal')
    
    for i in ax.containers:
        ax.bar_label(i, fmt='%.4f', padding=5)

    plt.title("Feature Importance")
    plt.xlabel("Relative Importance Score")
    plt.ylabel("Features")
    
    plt.tight_layout()
    plt.savefig('results/feature_importance_season.png', dpi=300)

    # save results to csv
    importance_df.to_csv('results/feature_importance_season.csv', index=False)
    
    return importance_df


def analyze_slope_prediction(results_df, driver_code, stint_id):
    stint_data = results_df[
        (results_df['Driver'] == driver_code) & 
        (results_df['Stint'] == stint_id)].copy()
    
    if len(stint_data) < 5:
        print(f"Not enough laps in stint {stint_id}")
        return

    x = stint_data['LapNumber']
    y_actual = stint_data['Actual']
    y_pred = stint_data['Predicted']

    slope_actual, intercept_actual = np.polyfit(x, y_actual, 1)
    slope_pred, intercept_pred = np.polyfit(x, y_pred, 1)

    plt.figure(figsize=(12, 6))
    
    plt.scatter(x, y_actual, color='blue', alpha=0.3, label='Actual Laps')
    plt.scatter(x, y_pred, color='red', alpha=0.3, label='Predicted Laps')

    plt.plot(x, slope_actual*x + intercept_actual, color='blue', linewidth=2, linestyle='-', 
            label=f'Actual Deg: {slope_actual:.3f} s/lap')
    plt.plot(x, slope_pred*x + intercept_pred, color='red', linewidth=2, linestyle='--', 
            label=f'Predicted Deg: {slope_pred:.3f} s/lap')

    plt.title(f"Tyre Degradation Analysis for {driver_code} (Stint {stint_id})")
    plt.xlabel("Lap Number")
    plt.ylabel("Lap Time (s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()
    plt.savefig(f'results/degradation_{driver_code}_stint{stint_id}.png', dpi=300)

    error = abs(slope_actual - slope_pred)
    print(f"Degradation Analysis for {driver_code} (Stint {stint_id}):")
    print(f"Actual Degradation:    {slope_actual:.4f} s/lap")
    print(f"Predicted Degradation: {slope_pred:.4f} s/lap")
    print(f"Slope Error:           {error:.4f} s/lap")


def plot_full_season_slopes(results_df, driver_code):
    rounds = sorted(results_df['Location_Encoded'].unique())
    num_rounds = len(rounds)
    
    if num_rounds == 0:
        print(f"No data found for the {driver_code}")
        return

    cols = 2
    rows = math.ceil(num_rounds / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), constrained_layout=True)
    
    if num_rounds > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, round_num in enumerate(rounds):
        ax = axes[i]
        
        race_data = results_df[
            (results_df['Driver'] == driver_code) & 
            (results_df['Location_Encoded'] == round_num)
        ]
        
        if not race_data.empty and 'Location' in race_data.columns:
            circuit_name = race_data['Location'].iloc[0]
        else:
            circuit_name = f"Round {round_num}"
        
        stints = sorted(race_data['Stint'].unique())
        has_data = False 
        
        for stint_id in stints:
            stint_data = race_data[race_data['Stint'] == stint_id]
            
            if len(stint_data) < 5:
                continue

            x = stint_data['LapNumber'].values
            y_actual = stint_data['Actual'].values
            y_pred = stint_data['Predicted'].values
        
            mask = np.isfinite(y_actual) & np.isfinite(y_pred)
            if np.sum(mask) < 2: 
                continue
                
            x_clean = x[mask]
            y_act_clean = y_actual[mask]
            y_pred_clean = y_pred[mask]

            # calculate slopes
            slope_act, intercept_act = np.polyfit(x_clean, y_act_clean, 1)
            slope_pred, intercept_pred = np.polyfit(x_clean, y_pred_clean, 1)
            

            line, = ax.plot(x_clean, slope_act*x_clean + intercept_act, 
                            linestyle='-', linewidth=2, alpha=0.8, 
                            label=f'S{int(stint_id)} Act ({slope_act:.3f})')
            
            color = line.get_color()
            ax.scatter(x_clean, y_act_clean, color=color, alpha=0.3, s=15, marker='o')
            ax.plot(x_clean, slope_pred*x_clean + intercept_pred, 
                    color=color, linestyle='--', linewidth=2, alpha=0.8)
            ax.scatter(x_clean, y_pred_clean, color=color, alpha=0.3, s=15, marker='x')
            
            has_data = True

        ax.set_title(f"{circuit_name} (R{int(round_num)})", fontsize=12, fontweight='bold')
        ax.set_xlabel("Lap Number")
        ax.set_ylabel("Lap Time (s)")
        ax.grid(True, alpha=0.2)
        
        if has_data:
            ax.legend(fontsize=9, loc='upper right')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.suptitle(f"2023 Season Degradation Analysis: {driver_code}", fontsize=16)
    plt.savefig(f'results/season_degradation_{driver_code}.png', dpi=300)


def run_season_walk_forward(df, features, model, summary_lines, target='LapTime_Sec', min_train_races=5, print_progress=True):
    start_time = time.time()

    df = df.sort_values(by='Time').reset_index(drop=True)

    races = df['Location'].unique()
    predictions_log = []

    log(summary_lines, f"Total races in dataset: {len(races)}")
    log(summary_lines, f"Initial training on first {min_train_races} races\n")

    # train on full races and test on the next one, iterating through the season
    for i in range(min_train_races, len(races)):
        train_races = races[:i]
        test_race = races[i]

        train_data = df[df['Location'].isin(train_races)]
        test_data = df[df['Location'] == test_race].copy()

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

        predictions_log.append(test_data[['Location', 'Location_Encoded', 'Driver', 'LapNumber', 'Stint', 'Compound', 'Actual', 'Predicted', 'Error']])

        if print_progress:
            race_mae = mean_absolute_error(y_actual, preds)
            log(summary_lines, f"Tested on {test_race} | Train size: {len(train_data)} laps | MAE: {race_mae:.3f} s")

    results_df = pd.concat(predictions_log, ignore_index=True)

    global_mae = mean_absolute_error(results_df['Actual'], results_df['Predicted'])
    global_rmse = np.sqrt(mean_squared_error(results_df['Actual'], results_df['Predicted']))

    end_time = time.time()
    if print_progress:
        log(summary_lines, f"\n{'-'*30}")
        log(summary_lines, f"Total Test Laps Predicted: {len(results_df)}")
        log(summary_lines, f"Global Average MAE: {global_mae:.3f} s")
        log(summary_lines, f"Global RMSE: {global_rmse:.3f} s")
        log(summary_lines, f"Time Taken: {end_time - start_time:.1f} seconds")

    return results_df

def log(summary_lines, *args, **kwargs):
    """
    Prints to terminal and appends to summary_lines for file saving.
    """
    text = " ".join(str(a) for a in args)
    print(text, **kwargs)
    summary_lines.append(text)


if __name__ == "__main__":
    start = time.time()
    summary_lines = []
    summary_path = "results/summary_model_training.txt"

    df_with_telemetry = pd.read_csv('data/dataset_2023.csv')

    features = [
        'LapNumber', 'Stint',
        'TyreLife', 'Tyre_Compound_Interaction', 'Stint_Progress',
        'AirTemp', 'Humidity', 'Pressure',
        'TrackTemp', 'WindDirection', 'WindSpeed',
        'FuelLoad', 'Track_Evolution_Physics',
        'Traction_1_5',
        'Asphalt_Grip_1_5', 'Asphalt_Abrasion_1_5', 'Track_Evolution_1_5',
        'Tyre_Stress_1_5', 'Braking_1_5', 'Lateral_1_5', 'Downforce_1_5',
        'Min_Pressure_Front_PSI', 'Min_Pressure_Rear_PSI',
        'Wear_Severity_Index', 'Track_Flow_Type',
        'Circuit_Length_KM', 'Cumulative_Field_Dist_KM',
        'Compound_Int', 'E_lap', 'Gap_To_Car_Ahead', 'Grip_Aero_Balance',
        'Total_Min_Pressure', 'Pressure_Delta', 'LatOffset_Mean',
        'LatOffset_Std', 'Prev_LapTime', 'Lag_2', 'Rolling_Avg_3',
        'Prev_Delta', 'Driver_Encoded', 'Location_Encoded', 'Team_Encoded',
        'Position', 'Dirty_Air_Fraction', 'DRS_Fraction',
        'Tyre_Grip_Index', 'Accumulated_Tyre_Wear',
        'Mean_Apex_Speed_Ratio', 'Std_Apex_Speed_Ratio',
        'Mean_Brake_Fraction', 'Std_Brake_Fraction',
        'Mean_Brake_Point_Norm', 'Std_Brake_Point_Norm',
        'Mean_Throttle_On_Dist_Norm', 'Std_Throttle_On_Dist_Norm',
        'Mean_Throttle_Integral_Norm', 'Std_Throttle_Integral_Norm',
        'Mean_Speed_CV', 'Std_Speed_CV',
        'P_0', 'P_1', 'P_2', 'Style_Cluster_ID', 'Style_Entropy',
        'Aero_Loss', 'Lap_Damage'
    ]
    # add features to log
    log(summary_lines, f"Features used for training ({len(features)}):")

    log(summary_lines, f"{features}\n")

    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    
    simulation_results = run_season_walk_forward(df_with_telemetry, features, model, summary_lines, target='Target_Delta', min_train_races=6, print_progress=True)
    simulation_results = convert_deltas_to_absolute_times(simulation_results, df_with_telemetry)

    importance_df = feature_importance_walk_forward_delta(df_with_telemetry, features)
    plot_full_season_slopes(simulation_results, 'VER')

    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f"\nSummary saved to {summary_path}")
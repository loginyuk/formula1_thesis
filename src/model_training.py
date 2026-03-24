import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import matplotlib.pyplot as plt
import seaborn as sns


def convert_deltas_to_absolute_times(simulation_results, df_with_telemetry):
    """
    Converts deltas time back to absolute lap times
    """
    simulation_results = simulation_results.merge(
        df_with_telemetry[['Driver', 'LapNumber', 'Prev_LapTime']], 
        on=['Driver', 'LapNumber'], how='left'
    )

    simulation_results['Predicted_Time'] = simulation_results['Prev_LapTime'] + simulation_results['Predicted']
    simulation_results['Actual_Time'] = simulation_results['Prev_LapTime'] + simulation_results['Actual']

    simulation_results['Predicted'] = simulation_results['Predicted_Time']
    simulation_results['Actual'] = simulation_results['Actual_Time']
    simulation_results.drop(columns=['Predicted_Time', 'Actual_Time', 'Prev_LapTime'], inplace=True)
    return simulation_results

def run_walk_forward_validation(df, features, model, target='LapTime_Sec', initial_train_size=20, print_progress=False):
    start_time = time.time()

    df = df.sort_values(by=['Driver', 'LapNumber']).reset_index(drop=True)
    drivers = df['Driver'].unique()
    predictions_log = []
    
    for driver in drivers:
        driver_df = df[df['Driver'] == driver].reset_index(drop=True)
        
        if len(driver_df) < initial_train_size + 5:
            continue
        
        # train starting from the initial lap
        for i in range(initial_train_size, len(driver_df)):
            train_data = driver_df.iloc[:i]
            test_row = driver_df.iloc[[i]]
            
            X_train = train_data[features]
            y_train = train_data[target]
            
            X_test = test_row[features]
            y_actual = test_row[target].values[0]
            
            current_model = clone(model)

            current_model.fit(X_train, y_train)
            pred_value = current_model.predict(X_test)[0]
            
            predictions_log.append({
                'Driver': driver,
                'LapNumber': test_row['LapNumber'].values[0],
                'Stint': test_row['Stint'].values[0] if 'Stint' in test_row else 0,
                'Actual': y_actual,
                'Predicted': pred_value,
                'Error': abs(y_actual - pred_value)
            })
    
    results_df = pd.DataFrame(predictions_log)
    
    mae = mean_absolute_error(results_df['Actual'], results_df['Predicted'])
    rmse = np.sqrt(mean_squared_error(results_df['Actual'], results_df['Predicted']))
    
    end_time = time.time()
    if print_progress:
        print(f"Total Laps Predicted: {len(results_df)}")
        print(f"Average MAE: {mae:.3f} s")
        print(f"RMSE: {rmse:.3f} s")
        print(f"Time Taken: {end_time - start_time:.1f} seconds")
    
    return results_df


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
    # plt.show()
    plt.savefig('results/feature_importance.png', dpi=300)
    
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

if __name__ == "__main__":
    start = time.time()
    df_with_telemetry = pd.read_csv('data/dataset_with_telemetry.csv')
    features = [
        'LapNumber', 'Stint',
        'TyreLife', 'AirTemp', 'Humidity', 'Pressure',
        'TrackTemp', 'WindDirection', 'WindSpeed',
        'FuelLoad', 'Track_Evolution_Physics',
        'Track_Flow_Type',
        'Compound_Int', 'E_lap', 'Gap_To_Car_Ahead', 'Grip_Aero_Balance',
        'Total_Min_Pressure', 'Pressure_Delta', 'LatOffset_Mean',
        'LatOffset_Std', 'Lap_Gap', 'Prev_LapTime', 'Lag_2', 'Rolling_Avg_3',
        'Prev_Delta', 'Driver_Encoded', 'Location_Encoded', 'Team_Encoded'
    ]
    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    
    simulation_results = run_walk_forward_validation(df_with_telemetry, features, model, print_progress=True)
    simulation_results = convert_deltas_to_absolute_times(simulation_results, df_with_telemetry)

    importance_df = feature_importance_walk_forward_delta(df_with_telemetry, features)

    for driver in simulation_results['Driver'].unique():
        for stint in simulation_results[simulation_results['Driver'] == driver]['Stint'].unique():
            analyze_slope_prediction(simulation_results, driver, stint)
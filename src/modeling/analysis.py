import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

from src.config import RESULTS_MODEL_DIR


def plot_feature_importance(df, features, out_dir=RESULTS_MODEL_DIR):
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
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f'{out_dir}/feature_importance.png', dpi=300)

    importance_df.to_csv(f'{out_dir}/feature_importance.csv', index=False)

    return importance_df


def analyze_slope_prediction(results_df, driver_code, stint_id, out_dir=RESULTS_MODEL_DIR):
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
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f'{out_dir}/degradation_{driver_code}_stint{stint_id}.png', dpi=300)

    error = abs(slope_actual - slope_pred)
    print(f"Degradation Analysis for {driver_code} (Stint {stint_id}):")
    print(f"Actual Degradation:    {slope_actual:.4f} s/lap")
    print(f"Predicted Degradation: {slope_pred:.4f} s/lap")
    print(f"Slope Error:           {error:.4f} s/lap")

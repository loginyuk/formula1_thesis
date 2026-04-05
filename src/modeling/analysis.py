import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone

from src.config import RESULTS_MODEL_DIR


def plot_feature_importance(df, features, model, out_dir=RESULTS_MODEL_DIR):
    fitted = clone(model)
    fitted.fit(df[features], df['Target_Delta'])

    if hasattr(fitted, 'feature_importances_'):
        importances = fitted.feature_importances_
    elif hasattr(fitted, 'named_steps'):
        final = fitted.named_steps[list(fitted.named_steps)[-1]]
        if hasattr(final, 'feature_importances_'):
            importances = final.feature_importances_
        elif hasattr(final, 'coef_'):
            importances = np.abs(final.coef_)
        else:
            print(f"Model has no feature importances — skipping plot")
            return None
    elif hasattr(fitted, 'coef_'):
        importances = np.abs(fitted.coef_)
    else:
        print(f"Model has no feature importances — skipping plot")
        return None

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
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
    plt.close()

    importance_df.to_csv(f'{out_dir}/feature_importance.csv', index=False)

    return importance_df


def plot_model_comparison(comparison_rows, out_dir=RESULTS_MODEL_DIR):
    df = pd.DataFrame(comparison_rows).sort_values('MAE')

    has_r2   = 'R2'   in df.columns
    has_mape = 'MAPE' in df.columns

    n_metrics = 2 + has_r2
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(len(df))

    # MAE and RMSE
    ax = axes[0]
    width = 0.38
    bars_mae  = ax.bar(x - width/2, df['MAE'],  width, label='MAE',  color='steelblue', alpha=0.88)
    bars_rmse = ax.bar(x + width/2, df['RMSE'], width, label='RMSE', color='tomato',    alpha=0.88)
    ax.bar_label(bars_mae,  fmt='%.3f', padding=3, fontsize=9)
    ax.bar_label(bars_rmse, fmt='%.3f', padding=3, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Error (seconds)', fontsize=11)
    ax.set_title('MAE / RMSE', fontsize=12)
    ax.legend(frameon=False)
    ax.yaxis.grid(True, ls='--', alpha=0.3)
    ax.set_axisbelow(True)

    # R2
    if has_r2:
        ax2 = axes[1]
        bars_r2 = ax2.bar(x, df['R2'], 0.55, color='#2ecc71', alpha=0.88)
        ax2.bar_label(bars_r2, fmt='%.4f', padding=3, fontsize=9)
        ax2.set_xticks(x)
        ax2.set_xticklabels(df['Model'], rotation=20, ha='right', fontsize=10)
        ax2.set_ylabel('R2', fontsize=11)
        ax2.set_title('R²', fontsize=12)
        ax2.yaxis.grid(True, ls='--', alpha=0.3)
        ax2.set_axisbelow(True)

    # MAPE
    if has_mape:
        ax3 = axes[-1]
        bars_mape = ax3.bar(x, df['MAPE'], 0.55, color='#f39c12', alpha=0.88)
        ax3.bar_label(bars_mape, fmt='%.2f%%', padding=3, fontsize=9)
        ax3.set_xticks(x)
        ax3.set_xticklabels(df['Model'], rotation=20, ha='right', fontsize=10)
        ax3.set_ylabel('MAPE (%)', fontsize=11)
        ax3.set_title('MAPE', fontsize=12)
        ax3.yaxis.grid(True, ls='--', alpha=0.3)
        ax3.set_axisbelow(True)

    plt.suptitle('Walk-Forward Validation — Model Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f'{out_dir}/model_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()

    df.to_csv(f'{out_dir}/model_comparison.csv', index=False)
    print(f"Saved model comparison to {out_dir}/")
    return df


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

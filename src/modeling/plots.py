import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

from src.config import RESULTS_MODEL_DIR


def plot_full_season_slopes(results_df, driver_code, out_dir=RESULTS_MODEL_DIR):
    driver_df = results_df[results_df['Driver'] == driver_code]

    if 'Year' in driver_df.columns and 'Location' in driver_df.columns:
        race_keys = driver_df.groupby(['Year', 'Location']).size().reset_index()[['Year', 'Location']]
        race_keys = race_keys.sort_values(['Year', 'Location']).values.tolist()
    elif 'Location' in driver_df.columns:
        locations = sorted(driver_df['Location'].unique())
        race_keys = [(None, loc) for loc in locations]
    else:
        race_keys = [(None, enc) for enc in sorted(driver_df['Location_Encoded'].unique())]

    num_rounds = len(race_keys)

    if num_rounds == 0:
        print(f"No data found for {driver_code}")
        return

    cols = 2
    rows = math.ceil(num_rounds / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), constrained_layout=True)

    if num_rounds > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, (year, location) in enumerate(race_keys):
        ax = axes[i]

        if year is not None and 'Year' in results_df.columns:
            race_data = results_df[
                (results_df['Driver'] == driver_code) &
                (results_df['Year'] == year) &
                (results_df['Location'] == location)
            ]
            title = f"{location} {int(year)}"
        elif 'Location' in results_df.columns:
            race_data = results_df[
                (results_df['Driver'] == driver_code) &
                (results_df['Location'] == location)
            ]
            title = str(location)
        else:
            race_data = results_df[
                (results_df['Driver'] == driver_code) &
                (results_df['Location_Encoded'] == location)
            ]
            title = f"Round {location}"

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

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Lap Number")
        ax.set_ylabel("Lap Time (s)")
        ax.grid(True, alpha=0.2)

        if has_data:
            ax.legend(fontsize=9, loc='upper right')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"Season Degradation Analysis: {driver_code}", fontsize=16)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f'{out_dir}/races_degradations_{driver_code}.png', dpi=300)
    plt.close()


def plot_predicted_vs_actual(results_df, out_dir=RESULTS_MODEL_DIR):
    actual = results_df['Actual'].values
    predicted = results_df['Predicted'].values

    r2 = r2_score(actual, predicted)
    mae = mean_absolute_error(actual, predicted)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(actual, predicted, alpha=0.15, s=8, color='steelblue', rasterized=True)

    lims = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    ax.plot(lims, lims, '--', color='#e74c3c', lw=1.5, label='Perfect prediction')

    ax.set_xlabel('Actual Lap Time (s)', fontsize=11)
    ax.set_ylabel('Predicted Lap Time (s)', fontsize=11)
    ax.set_title('Predicted vs Actual Lap Times', fontsize=13)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    textstr = f'R2 = {r2:.4f}\nMAE = {mae:.3f} s\nn = {len(actual)}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f'{out_dir}/predicted_vs_actual.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_residual_analysis(results_df, out_dir=RESULTS_MODEL_DIR):
    actual = results_df['Actual'].values
    predicted = results_df['Predicted'].values
    residuals = actual - predicted

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # residuals vs predicted
    ax1.scatter(predicted, residuals, alpha=0.12, s=8, color='steelblue', rasterized=True)
    ax1.axhline(0, color='#e74c3c', lw=1.5, ls='--')
    ax1.set_xlabel('Predicted Lap Time (s)', fontsize=11)
    ax1.set_ylabel('Residual (Actual − Predicted)', fontsize=11)
    ax1.set_title('Residuals vs Predicted', fontsize=12)
    ax1.grid(True, alpha=0.2)

    # residual histogram
    ax2.hist(residuals, bins=80, color='steelblue', alpha=0.8, edgecolor='white', lw=0.3)
    ax2.axvline(0, color='#e74c3c', lw=1.5, ls='--')
    mean_r = np.mean(residuals)
    std_r = np.std(residuals)
    ax2.set_xlabel('Residual (s)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Residual Distribution', fontsize=12)
    textstr = f'μ = {mean_r:.3f} s\nσ = {std_r:.3f} s'
    ax2.text(0.95, 0.95, textstr, transform=ax2.transAxes, fontsize=11,
             ha='right', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f'{out_dir}/residual_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_per_race_mae(results_df, out_dir=RESULTS_MODEL_DIR):
    if 'Year' in results_df.columns:
        results_df = results_df.copy()
        results_df['_RaceLabel'] = results_df['Year'].astype(int).astype(str) + ' ' + results_df['Location']
    else:
        results_df = results_df.copy()
        results_df['_RaceLabel'] = results_df['Location']

    race_order = results_df.groupby('_RaceLabel')['RoundNumber'].first()
    if 'Year' in results_df.columns:
        race_year = results_df.groupby('_RaceLabel')['Year'].first()
        race_order = pd.DataFrame({'Year': race_year, 'Round': race_order}).sort_values(['Year', 'Round'])
    else:
        race_order = race_order.sort_values()

    labels = race_order.index.tolist()
    maes = [mean_absolute_error(
        results_df[results_df['_RaceLabel'] == lbl]['Actual'],
        results_df[results_df['_RaceLabel'] == lbl]['Predicted']
    ) for lbl in labels]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(range(len(labels)), maes, '-o', color='steelblue', markersize=4, lw=1.2)
    ax.axhline(np.mean(maes), color='#e74c3c', ls='--', lw=1, label=f'Mean MAE = {np.mean(maes):.3f} s')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel('MAE (s)', fontsize=11)
    ax.set_title('Per-Race MAE (chronological order)', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f'{out_dir}/per_race_mae.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_compound_breakdown(results_df, out_dir=RESULTS_MODEL_DIR):
    if 'Compound' not in results_df.columns:
        print("No 'Compound' column in results — skipping compound breakdown plot")
        return

    compounds = results_df['Compound'].dropna().unique()
    compound_order = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
    compounds = [c for c in compound_order if c in compounds]
    if not compounds:
        compounds = sorted(results_df['Compound'].dropna().unique())

    maes, rmses, r2s, mapes, counts = [], [], [], [], []
    for c in compounds:
        sub = results_df[results_df['Compound'] == c]
        actual = sub['Actual'].values
        predicted = sub['Predicted'].values
        maes.append(mean_absolute_error(actual, predicted))
        rmses.append(np.sqrt(np.mean((actual - predicted)**2)))
        r2s.append(r2_score(actual, predicted))
        mask = np.abs(actual) > 1e-6
        mapes.append(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100 if mask.any() else 0.0)
        counts.append(len(sub))

    x_labels = [f"{c}\n(n={n})" for c, n in zip(compounds, counts)]
    x = np.arange(len(compounds))

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for ax, vals, color, title, fmt in zip(
        axes,
        [maes, rmses, r2s, mapes],
        ['steelblue', 'tomato', '#2ecc71', '#f39c12'],
        ['MAE (s)', 'RMSE (s)', 'R²', 'MAPE (%)'],
        ['%.3f', '%.3f', '%.4f', '%.2f'],
    ):
        bars = ax.bar(x, vals, 0.55, color=color, alpha=0.88)
        ax.bar_label(bars, fmt=fmt, padding=3, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.yaxis.grid(True, ls='--', alpha=0.3)
        ax.set_axisbelow(True)

    plt.suptitle('Prediction Metrics by Tyre Compound', fontsize=13)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f'{out_dir}/compound_breakdown.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_driver_mae(results_df, out_dir=RESULTS_MODEL_DIR):
    drivers = results_df['Driver'].unique()
    rows = []
    for drv in drivers:
        sub = results_df[results_df['Driver'] == drv]
        rows.append({
            'Driver': drv,
            'MAE': mean_absolute_error(sub['Actual'], sub['Predicted']),
            'Laps': len(sub),
        })
    df = pd.DataFrame(rows).sort_values('MAE')

    global_mae = mean_absolute_error(results_df['Actual'], results_df['Predicted'])

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.4)))
    bars = ax.barh(df['Driver'], df['MAE'], color='steelblue', alpha=0.85)
    ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=9)
    ax.axvline(global_mae, color='#e74c3c', ls='--', lw=1.5, label=f'Global MAE = {global_mae:.3f} s')

    ax.set_xlabel('MAE (s)', fontsize=11)
    ax.set_title('Prediction Error per Driver', fontsize=13)
    ax.legend(frameon=False)
    ax.xaxis.grid(True, ls='--', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f'{out_dir}/driver_mae.png', dpi=200, bbox_inches='tight')
    plt.close()

    df.to_csv(f'{out_dir}/driver_mae.csv', index=False)


def plot_no_clustering(rows, out_dir=RESULTS_MODEL_DIR):
    """
    rows: list of dicts with keys 'Features', 'MAE', 'RMSE', 'R2', 'MAPE'
    """
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric, color, label in zip(
        axes,
        ['MAE', 'RMSE', 'R2'],
        ['steelblue', 'tomato', '#2ecc71'],
        ['MAE (s)', 'RMSE (s)', 'R2'],
    ):
        bars = ax.bar(df['Features'], df[metric], color=color, alpha=0.88, width=0.5)
        ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.yaxis.grid(True, ls='--', alpha=0.3)
        ax.set_axisbelow(True)
        # zoom y-axis to show difference clearly
        vals = df[metric].values
        spread = vals.max() - vals.min()
        margin = max(spread * 2, 0.001)
        ax.set_ylim(vals.min() - margin, vals.max() + margin * 1.5)

    plt.suptitle('With vs Without Clustering Features', fontsize=14)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f'{out_dir}/model_no_clustering.png', dpi=200, bbox_inches='tight')
    plt.close()

    df.to_csv(f'{out_dir}/model_no_clustering.csv', index=False)

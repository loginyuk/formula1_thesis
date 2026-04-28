import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

from src.config import RESULTS_MODEL_DIR, PALETTE, PRIMARY_COLOUR, ACCENT_COLOUR

# full season slopes for a driver

def plot_full_season_slopes(results_df, driver_code, out_dir=RESULTS_MODEL_DIR):
    """
    For a given driver, plot lap time vs lap number for each stint in each race,
    with linear fit lines and slope deltas annotated.
    """
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
            slope_delta = slope_act - slope_pred

            line, = ax.plot(x_clean, slope_act*x_clean + intercept_act,
                            linestyle='-', linewidth=2, alpha=0.8,
                            label=f'S{int(stint_id)} Δ ({slope_delta:+.3f})')

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

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f'{out_dir}/races_degradations_{driver_code}.png', dpi=200)
    plt.close()


# model training result plots

def plot_pred_vs_act_errors(results_df, out_dir=RESULTS_MODEL_DIR):
    """
    Left: scatter of predicted vs actual lap times with perfect prediction line
    Right: scatter of prediction errors vs predicted lap time with zero error
    """
    actual = results_df['Actual'].values
    predicted = results_df['Predicted'].values
    errors = actual - predicted

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # predicted vs actual
    lims = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    ax1.scatter(actual, predicted, alpha=0.15, s=8, color=PRIMARY_COLOUR, rasterized=True)
    ax1.plot(lims, lims, '--', color=ACCENT_COLOUR, lw=1.5, label='Perfect prediction')
    ax1.set_xlabel('Actual Lap Time (s)', fontsize=11)
    ax1.set_ylabel('Predicted Lap Time (s)', fontsize=11)
    ax1.set_title('Predicted vs Actual Lap Times', fontsize=13)
    ax1.set_aspect('equal')
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.2)

    # prediction errors vs predicted
    ax2.scatter(predicted, errors, alpha=0.12, s=8, color=PRIMARY_COLOUR, rasterized=True)
    ax2.axhline(0, color=ACCENT_COLOUR, lw=1.5, ls='--')
    ax2.set_xlabel('Predicted Lap Time (s)', fontsize=11)
    ax2.set_ylabel('Prediction Error (Actual − Predicted, s)', fontsize=11)
    ax2.set_title('Prediction Errors vs Predicted', fontsize=12)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f'{out_dir}/pred_vs_act_errors.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_per_race_mae(results_df, out_dir=RESULTS_MODEL_DIR):
    """
    For each race, calculate MAE and plot as a bar chart
    """
    if 'Year' in results_df.columns:
        results_df = results_df.copy()
        results_df['RaceKey'] = results_df['Year'].astype(int).astype(str) + ' ' + results_df['Location']
    else:
        results_df['RaceKey'] = results_df['Location']

    race_order = results_df.groupby('RaceKey')['RoundNumber'].first()
    if 'Year' in results_df.columns:
        race_year = results_df.groupby('RaceKey')['Year'].first()
        race_order = pd.DataFrame({'Year': race_year, 'Round': race_order}).sort_values(['Year', 'Round'])
    else:
        race_order = race_order.sort_values()

    labels = race_order.index.tolist()
    maes = [mean_absolute_error(
        results_df[results_df['RaceKey'] == lbl]['Actual'],
        results_df[results_df['RaceKey'] == lbl]['Predicted']
    ) for lbl in labels]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(range(len(labels)), maes, '-o', color=PRIMARY_COLOUR, markersize=4, lw=1.2)
    ax.axhline(np.mean(maes), color=ACCENT_COLOUR, ls='--', lw=1, label=f'Mean MAE = {np.mean(maes):.3f} s')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel('MAE (s)', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f'{out_dir}/per_race_mae.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_compound_breakdown(results_df, out_dir=RESULTS_MODEL_DIR):
    """
    For each tyre compound, calculate metrics and plot as 4 subplots with bar charts
    """
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
        PALETTE[:4],
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

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f'{out_dir}/compound_breakdown.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_driver_mae(results_df, out_dir=RESULTS_MODEL_DIR):
    """
    For each driver, calculate MAE and plot as a horizontal bar chart
    """
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
    bars = ax.barh(df['Driver'], df['MAE'], color=PRIMARY_COLOUR, alpha=0.85)
    ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=9)
    ax.axvline(global_mae, color=ACCENT_COLOUR, ls='--', lw=1.5, label=f'Global MAE = {global_mae:.3f} s')

    ax.set_xlabel('MAE (s)', fontsize=11)
    ax.legend(frameon=False)
    ax.xaxis.grid(True, ls='--', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f'{out_dir}/driver_mae.png', dpi=200, bbox_inches='tight')
    plt.close()


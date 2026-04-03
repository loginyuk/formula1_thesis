import os
import math
import numpy as np
import matplotlib.pyplot as plt

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

"""
clustering_plots.py
───────────────
Six plots for the F1 lap-level driving style clustering model.

Plots produced (saved to out_dir/):
  1. centroid_profiles.png       — What each cluster looks like in feature space
  2. feature_space_scatter.png   — All laps plotted in 2D feature space
  3. driver_composition.png      — Per-driver breakdown of how many laps in each mode
  4. race_style_evolution.png    — How the field's driving mode shifts over the race
  5. {driver}_enhanced_timeline.png — 3-panel plot: pace + style + entropy
  6. laptime_by_cluster.png      — Does the cluster actually correlate with lap time?
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.config import N_CLUSTERS, RESULTS_VISUALIZATIONS_DIR

C = {0: '#e74c3c', 1: '#3498db', 2: '#2ecc71'}
NAMES_SHORT = {0: 'Exit Attack', 1: 'Speed Carry', 2: 'Throttle Save'}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def plot_centroid_profiles(df, z_cols, location, out_dir=RESULTS_VISUALIZATIONS_DIR):
    """
    Grouped bar chart: one bar per (cluster x feature).
    Shows what each cluster looks like relative to the field average.
    """
    centroids = df.groupby('Style_Cluster_ID')[z_cols].mean()

    feat_labels = ['Apex Speed\nRatio', 'Throttle-On\nDistance', 'Throttle\nIntegral']
    x = np.arange(len(z_cols))
    width = 0.26
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i in range(N_CLUSTERS):
        if i not in centroids.index:
            continue
        vals = centroids.loc[i].values
        bars = ax.bar(x + offsets[i], vals, width,
                      color=C[i], alpha=0.88,
                      label=NAMES_SHORT[i],
                      edgecolor='white', linewidth=0.6)
        for bar, v in zip(bars, vals):
            ypos = v + 0.02 if v >= 0 else v - 0.06
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f'{v:+.2f}', ha='center', va='bottom', fontsize=8)

    ax.axhline(0, color='#2c3e50', linewidth=0.9, linestyle='--', alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(feat_labels, fontsize=11)
    ax.set_ylabel('Z-score (relative to field median)', fontsize=11)
    ax.set_title(
        f'Driving Style Cluster Profiles — {location}\n'
        'Each bar shows how much above/below field average that cluster sits',
        fontsize=12, pad=10
    )
    ax.legend(frameon=False, loc='upper right')
    ax.yaxis.grid(True, ls='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/1_centroid_profiles.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  1. {os.path.basename(path)}")


def plot_feature_space(df, z_cols, location, out_dir=RESULTS_VISUALIZATIONS_DIR):
    """
    2D scatter of all laps in the two most discriminating features.
    Stars mark cluster centroids.
    """
    x_col = z_cols[1]
    y_col = z_cols[2]

    fig, ax = plt.subplots(figsize=(9, 7))

    for cid in range(N_CLUSTERS):
        sub = df[df['Style_Cluster_ID'] == cid]
        ax.scatter(sub[x_col], sub[y_col],
                   c=C[cid], alpha=0.35, s=18, rasterized=True,
                   label=f"{NAMES_SHORT[cid]}  (n={len(sub)})")

    # centroids
    ctr = df.groupby('Style_Cluster_ID')[z_cols].mean()
    for cid, row in ctr.iterrows():
        ax.scatter(row[x_col], row[y_col],
                   c=C[cid], s=250, marker='*',
                   edgecolors='#2c3e50', linewidths=0.8, zorder=6)

    ax.axhline(0, color='gray', lw=0.5, alpha=0.4)
    ax.axvline(0, color='gray', lw=0.5, alpha=0.4)
    ax.set_xlabel('Z Mean Throttle-On Distance  (← more negative = earlier throttle)', fontsize=11)
    ax.set_ylabel('Z Mean Throttle Integral  (higher = more exit throttle energy)', fontsize=11)
    ax.set_title(
        f'All Race Laps in Feature Space — {location}\n'
        '★ = cluster centroid',
        fontsize=12, pad=10
    )
    ax.legend(frameon=False, markerscale=2.5, fontsize=10)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/2_feature_space_scatter.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  2. {os.path.basename(path)}")


def plot_driver_composition(df, location, out_dir=RESULTS_VISUALIZATIONS_DIR):
    """
    Horizontal stacked bar per driver.
    Sorted by cluster 2 proportion.
    """
    counts = (df.groupby(['Driver', 'Style_Cluster_ID'])
               .size()
               .unstack(fill_value=0))
    for c in range(N_CLUSTERS):
        if c not in counts.columns:
            counts[c] = 0
    counts = counts[[0, 1, 2]]
    props = counts.div(counts.sum(axis=1), axis=0)
    props = props.sort_values(2, ascending=True)

    fig, ax = plt.subplots(figsize=(8, 9))

    y = np.arange(len(props))
    left = np.zeros(len(props))

    for cid in range(N_CLUSTERS):
        vals = props[cid].values
        ax.barh(y, vals, left=left,
                color=C[cid], alpha=0.88,
                label=NAMES_SHORT[cid], height=0.72)
        for j, (v, l) in enumerate(zip(vals, left)):
            if v > 0.15:
                ax.text(l + v / 2, j, f'{v:.0%}',
                        ha='center', va='center', fontsize=8,
                        color='white', fontweight='bold')
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(props.index, fontsize=10)
    ax.set_xlabel('Proportion of race laps', fontsize=11)
    ax.set_title(
        f'Driving Mode Composition by Driver\n{location}',
        fontsize=12, pad=10
    )
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlim(0, 1)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
    ax.axvline(1 / 3, color='gray', lw=0.6, ls='--', alpha=0.35)
    ax.axvline(2 / 3, color='gray', lw=0.6, ls='--', alpha=0.35)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/3_driver_composition.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  3. {os.path.basename(path)}")


def plot_race_evolution(df, location, out_dir=RESULTS_VISUALIZATIONS_DIR):
    """
    Stacked area: field-average P_0/P_1/P_2 per lap.
    Shows how the collective driving mode shifts as tyres degrade.
    """
    p_cols = [f'P_{i}' for i in range(N_CLUSTERS)]
    lap_avg = (df.groupby('LapNumber')[p_cols]
               .mean()
               .reset_index()
               .sort_values('LapNumber'))

    laps = lap_avg['LapNumber'].values

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.stackplot(
        laps,
        *[lap_avg[f'P_{i}'].values for i in range(N_CLUSTERS)],
        labels=[NAMES_SHORT[i] for i in range(N_CLUSTERS)],
        colors=[C[i] for i in range(N_CLUSTERS)],
        alpha=0.82
    )

    ax.set_ylabel('Average cluster probability (field)', fontsize=11)
    ax.set_xlabel('Lap Number', fontsize=11)
    ax.set_title(
        f'How Driving Style Evolves Over the Race — Field Average\n{location}',
        fontsize=12, pad=10
    )
    ax.legend(loc='lower left', frameon=False)
    ax.set_xlim(laps.min(), laps.max())
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/4_race_style_evolution.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  4. {os.path.basename(path)}")


def plot_driver_enhanced_timeline(df, driver_code, location, out_dir=RESULTS_VISUALIZATIONS_DIR):
    """
    Three-panel plot for driver_code:
      Top:    lap time scatter coloured by cluster label
      Middle: stacked bar of P_0/P_1/P_2 per lap
      Bottom: style entropy
    """
    drv = df[df['Driver'] == driver_code].sort_values('LapNumber').copy()
    laps = drv['LapNumber'].values

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 10),
        gridspec_kw={'height_ratios': [3, 1.5, 1]},
        sharex=True
    )

    # Top: lap time
    ax1.plot(laps, drv['LapTime_Sec'].values, color='#bdc3c7', lw=1.0, zorder=1)
    for cid in range(N_CLUSTERS):
        sub = drv[drv['Style_Cluster_ID'] == cid]
        ax1.scatter(sub['LapNumber'], sub['LapTime_Sec'],
                    color=C[cid], s=65, edgecolor='white', lw=0.4,
                    label=NAMES_SHORT[cid], zorder=3, alpha=0.95)

    ax1.set_ylabel('Lap Time (s)', fontsize=11)
    ax1.set_title(
        f'{driver_code} — Race Pace & Driving Style\n{location}',
        fontsize=13, pad=10
    )
    ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False, fontsize=9)
    ax1.yaxis.grid(True, ls='--', alpha=0.3)
    ax1.set_axisbelow(True)

    # Middle: stacked probability bars
    bottom = np.zeros(len(drv))
    for cid in range(N_CLUSTERS):
        vals = drv[f'P_{cid}'].values
        ax2.bar(laps, vals, bottom=bottom, color=C[cid], alpha=0.85, width=0.8)
        bottom += vals

    ax2.set_ylabel('Style\nProbability', fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0, 0.5, 1.0])
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    # Bottom: entropy
    ax3.fill_between(laps, drv['Style_Entropy'].values, color='#95a5a6', alpha=0.75, step='mid')
    ax3.set_ylabel('Entropy\n(uncertainty)', fontsize=10)
    ax3.set_xlabel('Lap Number', fontsize=11)

    for ax in (ax1, ax2, ax3):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/5_{driver_code}_enhanced_timeline.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  5. {os.path.basename(path)}")


def plot_laptime_by_cluster(df, location, out_dir=RESULTS_VISUALIZATIONS_DIR):
    """
    Violin plot: lap time distribution per cluster — all drivers.
    Validates that clusters have a real relationship with pace.
    """
    data = [
        df[df['Style_Cluster_ID'] == cid]['LapTime_Sec'].dropna().values
        for cid in range(N_CLUSTERS)
    ]
    medians = [np.median(d) for d in data]
    counts  = [len(d) for d in data]

    fig, ax = plt.subplots(figsize=(9, 5))

    parts = ax.violinplot(data, positions=range(N_CLUSTERS),
                          showmedians=True, showextrema=False, widths=0.65)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(C[i])
        pc.set_alpha(0.72)
        pc.set_edgecolor('none')
    parts['cmedians'].set_colors(['#2c3e50'] * N_CLUSTERS)
    parts['cmedians'].set_linewidth(2.5)

    for i, (med, n) in enumerate(zip(medians, counts)):
        ax.text(i, med + 0.25, f'{med:.2f}s',
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                color='#2c3e50')

    x_labels = [f"{NAMES_SHORT[i]}\n(n={counts[i]})" for i in range(N_CLUSTERS)]
    ax.set_xticks(range(N_CLUSTERS))
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylabel('Lap Time (s)', fontsize=11)
    ax.set_title(
        f'Lap Time Distribution by Driving Style Cluster\n'
        f'{location} · All Drivers · Line = median',
        fontsize=12, pad=10
    )
    ax.yaxis.grid(True, ls='--', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/6_laptime_by_cluster.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  6. {os.path.basename(path)}")

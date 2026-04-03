import os
import numpy as np
import matplotlib.pyplot as plt

from src.config import CLUSTER_COLOURS, CLUSTER_NAMES, RESULTS_CLUSTERING_DIR, RESULTS_CLUSTERING_VERIFICATION_DIR


def plot_lap_clusters_scatter(df_laps, x_col, y_col, driver_code, out_dir=RESULTS_CLUSTERING_DIR):
    drv = df_laps[df_laps['Driver'] == driver_code]
    n_k = df_laps['Style_Cluster_ID'].nunique()

    plt.figure(figsize=(9, 6))
    for cid in range(n_k):
        sub = drv[drv['Style_Cluster_ID'] == cid]
        plt.scatter(sub[x_col], sub[y_col],
                    label=CLUSTER_NAMES.get(cid, f'Cluster {cid}'),
                    color=CLUSTER_COLOURS.get(cid, '#888'),
                    alpha=0.7, s=60, edgecolors='white', linewidths=0.4)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{driver_code} — Lap Clusters (GMM, k={n_k})")
    plt.legend()
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/{driver_code}_style_clusters_gmm.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_race_timeline(df_laps, driver_code="VER", out_dir=RESULTS_CLUSTERING_DIR):
    """
    Plots lap time timeline with points colored by dominant style cluster,
    and a stacked bar of style probabilities below
    """
    drv = df_laps[df_laps['Driver'] == driver_code].sort_values('LapNumber')
    p_cols = sorted([c for c in df_laps.columns
                     if c.startswith('P_') and c[2:].isdigit()])
    n_k = len(p_cols)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    ax1.plot(drv['LapNumber'], drv['LapTime_Sec'], color='gray', lw=1, alpha=0.3, zorder=1)
    for cid in range(n_k):
        sub = drv[drv['Style_Cluster_ID'] == cid]
        ax1.scatter(sub['LapNumber'], sub['LapTime_Sec'],
                    label=CLUSTER_NAMES.get(cid, f'Cluster {cid}'),
                    color=CLUSTER_COLOURS.get(cid, '#888'),
                    s=80, edgecolor='black', lw=0.4, zorder=2)

    ax1.set_ylabel("Lap Time (s)")
    ax1.set_title(f"{driver_code} — Race Pace + Driving Style (k={n_k})")
    ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    ax1.grid(True, ls='--', alpha=0.4)

    bottom = np.zeros(len(drv))
    for i, col in enumerate(p_cols):
        ax2.bar(drv['LapNumber'].values, drv[col].values, bottom=bottom, color=CLUSTER_COLOURS.get(i, '#888'), alpha=0.85, label=col)
        bottom += drv[col].values
    ax2.set_ylabel("Style probability")
    ax2.set_xlabel("Lap")
    ax2.set_ylim(0, 1)
    ax2.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/{driver_code}_race_pace_timeline_gmm.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_probability_distributions(df_laps, out_dir=RESULTS_CLUSTERING_DIR):
    """
    Histogram of lap-level cluster probabilities
    """
    p_cols = sorted([c for c in df_laps.columns
                     if c.startswith('P_') and c[2:].isdigit()])
    fig, axes = plt.subplots(1, len(p_cols), figsize=(5 * len(p_cols), 4))
    if len(p_cols) == 1:
        axes = [axes]
    for i, (ax, col) in enumerate(zip(axes, p_cols)):
        ax.hist(df_laps[col].dropna(), bins=30,
                color=CLUSTER_COLOURS.get(i, '#888'), alpha=0.8, edgecolor='white')
        ax.set_title(col)
        ax.set_xlabel("Probability")
        ax.set_ylabel("Count")
    plt.suptitle(f"Lap-Level Style Probabilities (k={len(p_cols)})")
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/proportion_distributions.png", dpi=300)
    plt.close()


def plot_cluster_verification(df_season, out_dir=RESULTS_CLUSTERING_VERIFICATION_DIR):
    """
    Quick check plots for clustering across races and drivers.
        Per-race timeline grid — 3 sampled drivers, lap time colored by cluster + probability bars
        Cross-race cluster distribution heatmap — % of laps per cluster per race
    """
    os.makedirs(out_dir, exist_ok=True)
    p_cols = sorted([c for c in df_season.columns if c.startswith('P_') and c[2:].isdigit()])
    n_k = len(p_cols)

    has_year = 'Year' in df_season.columns
    group_keys = ['Year', 'Location'] if has_year else ['Location']

    # per-race timeline grids
    for group_vals, df_race in df_season.groupby(group_keys):
        if has_year:
            year, location = group_vals
            race_label = f"{year} {location}"
            safe_name = f"{year}_{location.replace(' ', '_')}"
        else:
            location = group_vals
            race_label = location
            safe_name = location.replace(' ', '_')

        drivers = df_race['Driver'].unique()
        sample = drivers[:3]
        fig, axes = plt.subplots(len(sample), 2, figsize=(14, 4 * len(sample)),
                                 gridspec_kw={'width_ratios': [3, 1]})
        if len(sample) == 1:
            axes = [axes]

        for ax_row, drv in zip(axes, sample):
            ax_t, ax_b = ax_row
            d = df_race[df_race['Driver'] == drv].sort_values('LapNumber')

            ax_t.plot(d['LapNumber'], d['LapTime_Sec'], color='gray', lw=1, alpha=0.3, zorder=1)
            for cid in range(n_k):
                sub = d[d['Style_Cluster_ID'] == cid]
                ax_t.scatter(sub['LapNumber'], sub['LapTime_Sec'],
                             color=CLUSTER_COLOURS.get(cid, '#888'), s=50,
                             edgecolor='black', lw=0.3, zorder=2,
                             label=CLUSTER_NAMES.get(cid, f'C{cid}'))
            ax_t.set_ylabel(f"{drv}\nLapTime (s)")
            ax_t.legend(fontsize=7, loc='upper right')
            ax_t.grid(True, ls='--', alpha=0.3)

            bottom = np.zeros(len(d))
            for i, col in enumerate(p_cols):
                ax_b.bar(d['LapNumber'].values, d[col].values, bottom=bottom,
                         color=CLUSTER_COLOURS.get(i, '#888'), alpha=0.85)
                bottom += d[col].fillna(0).values
            ax_b.set_ylim(0, 1)
            ax_b.set_ylabel("P(cluster)")
            ax_b.set_xlabel("Lap")

        fig.suptitle(f"{race_label} — Cluster verification", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{safe_name}_timeline.png", dpi=150, bbox_inches='tight')
        plt.close()

    # cross-race cluster distribution heatmap
    cluster_pct = (
        df_season.groupby(group_keys + ['Style_Cluster_ID'])
        .size()
        .unstack(fill_value=0)
    )
    cluster_pct = cluster_pct.div(cluster_pct.sum(axis=1), axis=0) * 100

    if has_year:
        row_labels = [f"{y} {l}" for y, l in cluster_pct.index]
    else:
        row_labels = list(cluster_pct.index)

    fig, ax = plt.subplots(figsize=(max(6, n_k * 2), max(4, len(cluster_pct) * 0.4 + 1)))
    im = ax.imshow(cluster_pct.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
    ax.set_xticks(range(n_k))
    ax.set_xticklabels([CLUSTER_NAMES.get(i, f'C{i}') for i in range(n_k)], rotation=20, ha='right')
    ax.set_yticks(range(len(cluster_pct)))
    ax.set_yticklabels(row_labels, fontsize=7)
    for i in range(len(cluster_pct)):
        for j in range(n_k):
            ax.text(j, i, f"{cluster_pct.values[i, j]:.0f}%", ha='center', va='center', fontsize=7)
    plt.colorbar(im, ax=ax, label='% of laps')
    ax.set_title("Cluster distribution per race (%)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/cross_race_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Verification plots saved to {out_dir}/")

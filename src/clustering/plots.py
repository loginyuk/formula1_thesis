import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.config import CLUSTER_COLOURS, CLUSTER_NAMES, RESULTS_CLUSTERING_DIR, RESULTS_CLUSTERING_VERIFICATION_DIR

NAMES_K3 = {0: 'Exit Attack', 1: 'Speed Carry', 2: 'Throttle Save'}
NAMES_K2 = {0: 'Aggressive',  1: 'Save'}

def _cluster_names(cluster_ids):
    if len(cluster_ids) == 2:
        return NAMES_K2
    return NAMES_K3


plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# driver-level analysis

def plot_race_timeline(df_laps, driver_code="VER", out_dir=RESULTS_CLUSTERING_DIR):
    """
    For a given driver and race, plot lap time timeline with cluster-colored points, 
    and cluster probability stacked bars below
    """
    drv = df_laps[df_laps['Driver'] == driver_code].sort_values('LapNumber')
    p_cols = sorted([c for c in df_laps.columns if c.startswith('P_') and c[2:].isdigit()])
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
    ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    ax1.grid(True, ls='--', alpha=0.4)

    bottom = np.zeros(len(drv))
    for i, col in enumerate(p_cols):
        ax2.bar(drv['LapNumber'].values, drv[col].values, bottom=bottom,
                color=CLUSTER_COLOURS.get(i, '#888'), alpha=0.85, label=col)
        bottom += drv[col].values
    ax2.set_ylabel("Style probability")
    ax2.set_xlabel("Lap")
    ax2.set_ylim(0, 1)
    ax2.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/{driver_code}_race_pace_timeline_gmm.png", dpi=200, bbox_inches='tight')
    plt.close()


# per-cluster analysis

def plot_centroid_profiles(df, z_cols, location, out_dir=RESULTS_CLUSTERING_DIR):
    """
    Bar chart of centroid feature values per cluster
    """
    cluster_ids = sorted(df['Style_Cluster_ID'].dropna().unique().astype(int))
    names = _cluster_names(cluster_ids)
    centroids = df.groupby('Style_Cluster_ID')[z_cols].mean()

    feat_labels = ['Apex Speed\nRatio', 'Throttle-On\nDistance', 'Throttle\nIntegral']
    x = np.arange(len(z_cols))
    width = 0.26
    n = len(cluster_ids)
    offsets = [width * (i - (n - 1) / 2) for i in range(n)]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, cid in enumerate(cluster_ids):
        if cid not in centroids.index:
            continue
        vals = centroids.loc[cid].values
        bars = ax.bar(x + offsets[i], vals, width,
                      color=CLUSTER_COLOURS[cid], alpha=0.88,
                      label=names.get(cid, f'C{cid}'),
                      edgecolor='white', linewidth=0.6)
        for bar, v in zip(bars, vals):
            ypos = v + 0.02 if v >= 0 else v - 0.06
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f'{v:+.2f}', ha='center', va='bottom', fontsize=8)

    ax.axhline(0, color='#2c3e50', linewidth=0.9, linestyle='--', alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(feat_labels, fontsize=11)
    ax.set_ylabel('Z-score', fontsize=11)
    ax.legend(frameon=False, loc='upper right')
    ax.yaxis.grid(True, ls='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/centroid_profiles.png", dpi=200, bbox_inches='tight')
    plt.close()


def plot_feature_space(df, z_cols, location, out_dir=RESULTS_CLUSTERING_DIR):
    """
    Scatter of two key features with points colored by cluster and centroids highlighted
    """
    x_col = z_cols[1]
    y_col = z_cols[2]

    cluster_ids = sorted(df['Style_Cluster_ID'].dropna().unique().astype(int))
    names = _cluster_names(cluster_ids)

    fig, ax = plt.subplots(figsize=(9, 7))

    for cid in cluster_ids:
        sub = df[df['Style_Cluster_ID'] == cid]
        ax.scatter(sub[x_col], sub[y_col],
                   c=CLUSTER_COLOURS[cid], alpha=0.35, s=18, rasterized=True,
                   label=f"{names.get(cid, f'C{cid}')}  (n={len(sub)})")

    ctr = df.groupby('Style_Cluster_ID')[z_cols].mean()
    for cid, row in ctr.iterrows():
        ax.scatter(row[x_col], row[y_col],
                   c=CLUSTER_COLOURS[cid], s=250, marker='*',
                   edgecolors='#2c3e50', linewidths=0.8, zorder=6)

    ax.axhline(0, color='gray', lw=0.5, alpha=0.4)
    ax.axvline(0, color='gray', lw=0.5, alpha=0.4)
    ax.set_xlabel('Z_(Mean Throttle-On Dist Norm)', fontsize=11)
    ax.set_ylabel('Z_(Mean Throttle Integral Norm)', fontsize=11)
    ax.legend(frameon=False, markerscale=2.5, fontsize=10)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/feature_space_scatter.png", dpi=200, bbox_inches='tight')
    plt.close()


def plot_driver_composition(df, location, out_dir=RESULTS_CLUSTERING_DIR):
    """
    Horizontal stacked bar of cluster proportion per driver
    """
    cluster_ids = sorted(df['Style_Cluster_ID'].dropna().unique().astype(int))
    names = _cluster_names(cluster_ids)
    counts = df.groupby(['Driver', 'Style_Cluster_ID']).size().unstack(fill_value=0)
    for c in cluster_ids:
        if c not in counts.columns:
            counts[c] = 0
    counts = counts[cluster_ids]
    props = counts.div(counts.sum(axis=1), axis=0)
    props = props.sort_values(cluster_ids[-1], ascending=True)

    fig, ax = plt.subplots(figsize=(8, 9))
    y = np.arange(len(props))
    left = np.zeros(len(props))

    for cid in cluster_ids:
        vals = props[cid].values
        ax.barh(y, vals, left=left, color=CLUSTER_COLOURS[cid], alpha=0.88,
                label=names.get(cid, f'C{cid}'), height=0.72)
        for j, (v, l) in enumerate(zip(vals, left)):
            if v > 0.15:
                ax.text(l + v / 2, j, f'{v:.0%}',
                        ha='center', va='center', fontsize=8,
                        color='white', fontweight='bold')
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(props.index, fontsize=10)
    ax.set_xlabel('Proportion of race laps', fontsize=11)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlim(0, 1)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
    ax.axvline(1 / 3, color='gray', lw=0.6, ls='--', alpha=0.35)
    ax.axvline(2 / 3, color='gray', lw=0.6, ls='--', alpha=0.35)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/driver_composition.png", dpi=200, bbox_inches='tight')
    plt.close()


def plot_race_evolution(df, location, out_dir=RESULTS_CLUSTERING_DIR):
    """
    Stacked area plot of cluster proportions per lap across the race
    """
    cluster_ids = sorted(df['Style_Cluster_ID'].dropna().unique().astype(int))
    names = _cluster_names(cluster_ids)
    p_cols = [f'P_{i}' for i in cluster_ids]
    lap_avg = df.groupby('LapNumber')[p_cols].mean().reset_index().sort_values('LapNumber')
    laps = lap_avg['LapNumber'].values

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.stackplot(
        laps,
        *[lap_avg[f'P_{i}'].values for i in cluster_ids],
        labels=[names.get(i, f'C{i}') for i in cluster_ids],
        colors=[CLUSTER_COLOURS[i] for i in cluster_ids],
        alpha=0.82
    )
    ax.set_ylabel('Average cluster probability', fontsize=11)
    ax.set_xlabel('Lap Number', fontsize=11)
    ax.legend(loc='lower left', frameon=False)
    ax.set_xlim(laps.min(), laps.max())
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/race_style_evolution.png", dpi=200, bbox_inches='tight')
    plt.close()


def plot_laptime_by_cluster(df, location, out_dir=RESULTS_CLUSTERING_DIR):
    """
    Violin plot of lap time distribution per cluster, with median values annotated
    """
    cluster_ids = sorted(df['Style_Cluster_ID'].dropna().unique().astype(int))
    names = _cluster_names(cluster_ids)
    data = [df[df['Style_Cluster_ID'] == cid]['LapTime_Sec'].dropna().values for cid in cluster_ids]
    medians = [np.median(d) for d in data]
    counts = [len(d) for d in data]

    fig, ax = plt.subplots(figsize=(9, 5))
    parts = ax.violinplot(data, positions=range(len(cluster_ids)),
                          showmedians=True, showextrema=False, widths=0.65)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(CLUSTER_COLOURS[cluster_ids[i]])
        pc.set_alpha(0.72)
        pc.set_edgecolor('none')
    parts['cmedians'].set_colors(['#2c3e50'] * len(cluster_ids))
    parts['cmedians'].set_linewidth(2.5)

    for i, med in enumerate(medians):
        ax.text(i, med + 0.25, f'{med:.2f}s',
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2c3e50')

    x_labels = [f"{names.get(cid, f'C{cid}')}\n(n={counts[i]})" for i, cid in enumerate(cluster_ids)]
    ax.set_xticks(range(len(cluster_ids)))
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylabel('Lap Time (s)', fontsize=11)
    ax.yaxis.grid(True, ls='--', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/laptime_by_cluster.png", dpi=200, bbox_inches='tight')
    plt.close()


# per-race verification

def plot_cluster_verification(df_season, out_dir=RESULTS_CLUSTERING_VERIFICATION_DIR):
    """
    For each race, plot lap time timeline with cluster-colored points and cluster probability bars below
    """
    os.makedirs(out_dir, exist_ok=True)
    p_cols = sorted([c for c in df_season.columns if c.startswith('P_') and c[2:].isdigit()])
    n_k = len(p_cols)

    has_year = 'Year' in df_season.columns
    group_keys = ['Year', 'Location'] if has_year else ['Location']

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

        plt.tight_layout()
        plt.savefig(f"{out_dir}/{safe_name}_timeline.png", dpi=200, bbox_inches='tight')
        plt.close()

    cluster_pct = (
        df_season.groupby(group_keys + ['Style_Cluster_ID'])
        .size()
        .unstack(fill_value=0)
    )
    cluster_pct = cluster_pct.div(cluster_pct.sum(axis=1), axis=0) * 100

    row_labels = [f"{y} {l}" for y, l in cluster_pct.index] if has_year else list(cluster_pct.index)

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
    plt.tight_layout()
    plt.savefig(f"{out_dir}/cross_race_heatmap.png", dpi=200, bbox_inches='tight')
    plt.close()

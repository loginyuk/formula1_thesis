import os
import time
import fastf1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment

def add_curvature_to_telemetry(tel):
    """
    Calculates curvature using smoothed GPS coordinates.
    Replaces missing steering angle data
    """
    tel = tel.drop_duplicates(subset=['Distance']).copy()
    tel = tel[tel['Distance'].diff().fillna(1) > 0].copy()
    tel = tel.reset_index(drop=True)

    x = tel['X'].ffill().bfill().values
    y = tel['Y'].ffill().bfill().values

    x_smooth = savgol_filter(x, window_length=15, polyorder=3)
    y_smooth = savgol_filter(y, window_length=15, polyorder=3)

    dist = tel['Distance'].values
    dx, dy = np.gradient(x_smooth, dist), np.gradient(y_smooth, dist)
    ddx, ddy = np.gradient(dx, dist), np.gradient(dy, dist)

    numerator = (dx * ddy) - (dy * ddx)
    denominator = np.power(dx**2 + dy**2, 1.5) + 1e-6
    tel['Curvature'] = np.abs(numerator / denominator)
    tel.loc[tel['Speed'] < 30, 'Curvature'] = 0.0
    return tel


def extract_corner_features(corner_tel):
    """
    Extracts 6 driving-style features from a single corner
    """
    if len(corner_tel) < 5:
        return None

    dist = corner_tel['Distance'].values
    curvature = corner_tel['Curvature'].values
    brake = corner_tel['Brake'].astype(float).values   # boolean (0, 1)
    speed = corner_tel['Speed'].values
    throttle = np.clip(corner_tel['Throttle'].astype(float).values, 0, 100)

    corner_len = max(dist[-1] - dist[0], 1.0)

    apex_idx = np.argmax(curvature)
    apex_dist = dist[apex_idx]
    apex_speed = speed[apex_idx]

    # entry
    pre_apex = speed[:apex_idx] if apex_idx > 0 else speed
    entry_speed = np.max(pre_apex)
    if entry_speed < 5:
        return None

    apex_speed_ratio = apex_speed / entry_speed

    # braking behavior
    brake_fraction = np.sum(brake) / len(brake)

    if np.any(brake == 1):
        brake_start = dist[np.argmax(brake == 1)]
        brake_point_norm = (brake_start - dist[0]) / corner_len
    else:
        brake_point_norm = np.nan

    # exit
    post_apex = dist > apex_dist
    remaining_len = max(dist[-1] - apex_dist, 1.0)
    t_post = throttle[post_apex]
    d_post = dist[post_apex]

    on_mask = t_post > 20
    if np.any(on_mask):
        throttle_on_dist_norm = (d_post[on_mask][0] - apex_dist) / remaining_len
    else:
        throttle_on_dist_norm = 1.0

    if len(t_post) > 1:
        throttle_integral_norm = np.trapz(t_post, d_post) / (100.0 * remaining_len)
    else:
        throttle_integral_norm = t_post[0] / 100.0 if len(t_post) else 0.0

    # smoothness
    speed_cv = np.std(speed) / (entry_speed + 1e-6)

    return {
        'Apex_Speed_Ratio':       apex_speed_ratio,
        'Brake_Fraction':         brake_fraction,
        'Brake_Point_Norm':       brake_point_norm,
        'Throttle_On_Dist_Norm':  throttle_on_dist_norm,
        'Throttle_Integral_Norm': throttle_integral_norm,
        'Speed_CV':               speed_cv,
    }


def build_corner_zones(session):
    """
    Builds corner zones dict from session circuit info
    """
    corners_info = session.get_circuit_info().corners
    return {
        f"{c['Number']}{c['Letter']}": (c['Distance'] - 100, c['Distance'] + 100)
        for _, c in corners_info.iterrows()
    }


def build_corner_database(session, laps, all_telemetry=None, corner_zones=None):
    """
    Builds a corner-level database of driving style metrics.
    If all_telemetry dict is provided, uses it directly.
    Otherwise loads telemetry per driver from the session.
    """
    if corner_zones is None:
        corner_zones = build_corner_zones(session)

    all_corners_data = []
    for drv in laps['Driver'].unique():
        drv_laps = laps[laps['Driver'] == drv]

        try:
            if all_telemetry is not None and drv in all_telemetry:
                full_tel = all_telemetry[drv].copy()
                if 'Distance' not in full_tel.columns:
                    full_tel = full_tel.copy()
                    full_tel['Distance'] = np.nan
            else:
                full_tel = session.laps.pick_drivers(drv).get_telemetry()
                if 'Distance' not in full_tel.columns:
                    full_tel.add_distance()
            full_tel = add_curvature_to_telemetry(full_tel)
        except Exception as e:
            print(f"Telemetry load failed for {drv}: {e}")
            continue

        for _, lap in drv_laps.iterrows():
            try:
                mask = (full_tel['SessionTime'] >= lap['LapStartTime']) & (full_tel['SessionTime'] <= lap['Time'])
                lap_tel = full_tel.loc[mask].copy()
                if lap_tel.empty:
                    continue

                lap_tel['Distance'] = lap_tel['Distance'] - lap_tel['Distance'].min()

                for corner_id, (start_m, end_m) in corner_zones.items():
                    corner_tel = lap_tel[(lap_tel['Distance'] >= start_m) &
                                    (lap_tel['Distance'] <= end_m)].copy()
                    metrics = extract_corner_features(corner_tel)
                    if metrics is None:
                        continue

                    metrics['Driver'] = drv
                    metrics['LapNumber'] = lap['LapNumber']
                    metrics['LapTime_Sec'] = lap['LapTime'].total_seconds()
                    metrics['Corner_ID'] = corner_id
                    all_corners_data.append(metrics)

            except Exception as e:
                print(f"Corner extraction error on lap {lap['LapNumber']} for {drv}: {e}")

    return pd.DataFrame(all_corners_data)


def weighted_mean(g, col):
    valid = g[[col, 'corner_w']].dropna(subset=[col])
    if valid.empty:
        return np.nan
    return (valid[col] * valid['corner_w']).sum() / (valid['corner_w'].sum() + 1e-9)

def lap_summary(g):
    result = {}
    for col in FEATURE_COLS:
        result[f'Mean_{col}'] = weighted_mean(g, col)
        result[f'Std_{col}']  = g[col].std(skipna=True)
    return pd.Series(result)

def aggregate_corners_to_laps(df_corners, corner_weights):
    """
    Aggregates corner-level features to lap-level by weighted averaging across corners
    """
    df = df_corners.copy()
    df['corner_w'] = df['Corner_ID'].map(corner_weights).fillna(1.0)

    df_laps = (df.groupby(['Driver', 'LapNumber']).apply(lap_summary, include_groups=False).reset_index())

    lap_times = df_corners[['Driver', 'LapNumber', 'LapTime_Sec']].drop_duplicates()
    df_laps   = pd.merge(df_laps, lap_times, on=['Driver', 'LapNumber'])
    return df_laps

def normalize_lap_features(df_laps, feature_cols, clip_sigma=3.0):
    """
    Normalizes lap-level features using robust scaling (median + IQR) and clips outliers
    """
    df = df_laps.copy()
    z_cols = []

    for col in feature_cols:
        z = f'Z_{col}'
        median = df[col].median()
        iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
        df[z] = ((df[col] - median) / (iqr + 1e-9)).clip(-clip_sigma, clip_sigma)
        df[z] = df[z].fillna(0.0)
        z_cols.append(z)
    return df, z_cols


def normalize_dict(d):
    lo, hi = min(d.values()), max(d.values())
    if hi == lo:
        return {k: 0.5 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def calculate_circuit_weights(session, corner_zones=None):
    """
    Calculates corner weights based on their contribution to lap time and energy
    """
    if corner_zones is None:
        corner_zones = build_corner_zones(session)

    lap = session.laps.pick_quicklaps().pick_fastest()
    lap_tel = add_curvature_to_telemetry(lap.get_telemetry())

    w_time, w_energy = {}, {}
    for corner_id, (start_m, end_m) in corner_zones.items():
        c_tel = lap_tel[(lap_tel['Distance'] >= start_m) & (lap_tel['Distance'] <= end_m)]
        if c_tel.empty:
            continue

        speed_ms  = c_tel['Speed'].values / 3.6
        lateral_g = np.abs(c_tel['Curvature'].values) * speed_ms ** 2
        w_energy[corner_id] = np.sum(lateral_g * speed_ms)

        after = lap_tel[lap_tel['Distance'] > end_m]
        braking = after[after['Brake'] == 1]
        next_brake = braking.iloc[0]['Distance'] if not braking.empty else lap_tel['Distance'].max()
        straight = after[after['Distance'] < next_brake]

        if straight.empty:
            w_time[corner_id] = 0.0
        else:
            deltas = np.diff(straight['Distance'].values, prepend=straight['Distance'].iloc[0])
            w_time[corner_id] = np.sum(deltas[straight['Throttle'].astype(float).values >= 99])

    w_time = normalize_dict(w_time)
    w_energy = normalize_dict(w_energy)
    return {corner_id: 0.5 * w_time[corner_id] + 0.5 * w_energy[corner_id]
            for corner_id in corner_zones if corner_id in w_time and corner_id in w_energy}


def silhouette_scoring(X, max_k=4):
    """
    Runs silhouette scoring for different k to help validate cluster count choice
    """
    print("\nSilhouette scoring:")
    for k in range(2, max_k + 1):
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42, n_init=10)
        labels = gmm.fit_predict(X)
        if len(np.unique(labels)) < 2:
            continue
        sil = silhouette_score(X, labels, sample_size=min(2000, len(X)))
        print(f" k={k}  silhouette={sil:.4f}")


def align_labels(new_means, ref_means):
    """
    Aligns new cluster means to reference means using the Hungarian algorithm
    """
    cost = cdist(ref_means, new_means)
    row_ind, col_ind = linear_sum_assignment(cost)
    return {new: ref for ref, new in zip(row_ind, col_ind)}


def fit_gmm(X, n_clusters):
    """
    Fits GMM to the normalized data, orders clusters by aggression score,
    returns aligned probabilities and labels
    """
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42, n_init=10)
    gmm.fit(X)

    raw_proba  = gmm.predict_proba(X)
    raw_labels = np.argmax(raw_proba, axis=1)

    means = np.array([X[raw_labels == c].mean(axis=0)
                      for c in range(n_clusters)])

    # always order by aggression = low throttle-on distance + high throttle integral
    aggression = (-means[:, 1] + means[:, 2])
    order = np.argsort(aggression)[::-1]
    label_map = {old: new for new, old in enumerate(order)}
    reference_means = means[order]

    aligned = np.zeros_like(raw_proba)
    for raw_id, aligned_id in label_map.items():
        aligned[:, aligned_id] = raw_proba[:, raw_id]

    return gmm, aligned, np.argmax(aligned, axis=1), reference_means


def cluster_laps(df_laps_norm, z_features, n_clusters):
    """
    Main function to fit GMM and label laps with cluster probabilities and IDs
    """
    X = df_laps_norm[z_features].values

    _, proba, labels, _ = fit_gmm(X, n_clusters)

    df = df_laps_norm.copy()
    for i in range(proba.shape[1]):
        df[f'P_{i}'] = proba[:, i]
    df['Style_Cluster_ID'] = labels.astype(int)

    p_arr = np.clip(proba, 1e-9, 1)
    df['Style_Entropy'] = -np.sum(p_arr * np.log(p_arr), axis=1)

    return df


# Plotting

CLUSTER_COLOURS = {0: '#e74c3c', 1: '#3498db', 2: '#2ecc71', 3: '#f39c12'}
CLUSTER_NAMES = {0: 'Exit Attack', 1: 'Speed Carry', 2: 'Throttle Save', 3: 'Cluster 3',}


def plot_lap_clusters_scatter(df_laps, x_col, y_col, driver_code):
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
    plt.savefig(f"results/clustering/v4/{driver_code}_style_clusters_gmm.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_race_timeline(df_laps, driver_code="VER"):
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
    plt.savefig(f"results/clustering/v4/{driver_code}_race_pace_timeline_gmm.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_probability_distributions(df_laps):
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
    plt.savefig("results/clustering/v4/proportion_distributions.png", dpi=300)
    plt.close()


def plot_cluster_verification(df_season, out_dir="results/clustering/verification"):
    """
    Quick check plots for clustering across races and drivers.
        Per-race timeline grid — 3 sampled drivers, lap time colored by cluster + probability bars
        Cross-race cluster distribution heatmap — % of laps per cluster per race
    """
    os.makedirs(out_dir, exist_ok=True)
    p_cols = sorted([c for c in df_season.columns if c.startswith('P_') and c[2:].isdigit()])
    n_k = len(p_cols)

    # per-race timeline grids
    for location, df_race in df_season.groupby('Location'):
        drivers = df_race['Driver'].unique()
        sample = drivers[:3]  # up to 3 drivers
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

        fig.suptitle(f"{location} — Cluster verification", fontsize=12)
        plt.tight_layout()
        safe_loc = location.replace(' ', '_')
        plt.savefig(f"{out_dir}/{safe_loc}_timeline.png", dpi=150, bbox_inches='tight')
        plt.close()

    # cross-race cluster distribution heatmap
    cluster_pct = (
        df_season.groupby(['Location', 'Style_Cluster_ID'])
        .size()
        .unstack(fill_value=0)
    )
    cluster_pct = cluster_pct.div(cluster_pct.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(max(6, n_k * 2), max(4, len(cluster_pct) * 0.5 + 1)))
    im = ax.imshow(cluster_pct.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
    ax.set_xticks(range(n_k))
    ax.set_xticklabels([CLUSTER_NAMES.get(i, f'C{i}') for i in range(n_k)], rotation=20, ha='right')
    ax.set_yticks(range(len(cluster_pct)))
    ax.set_yticklabels(cluster_pct.index)
    for i in range(len(cluster_pct)):
        for j in range(n_k):
            ax.text(j, i, f"{cluster_pct.values[i, j]:.0f}%", ha='center', va='center', fontsize=8)
    plt.colorbar(im, ax=ax, label='% of laps')
    ax.set_title("Cluster distribution per race (%)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/cross_race_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Verification plots saved to {out_dir}/")


def log(summary_lines, *args, **kwargs):
    """
    Prints to terminal and appends to summary_lines for file saving.
    """
    text = " ".join(str(a) for a in args)
    print(text, **kwargs)
    summary_lines.append(text)



N_CLUSTERS = 3

FEATURE_COLS = [
    'Apex_Speed_Ratio',
    'Brake_Fraction',
    'Brake_Point_Norm',
    'Throttle_On_Dist_Norm',
    'Throttle_Integral_Norm',
    'Speed_CV',
]

CLUSTER_FEATURES = [
    'Apex_Speed_Ratio',
    'Throttle_On_Dist_Norm',
    'Throttle_Integral_Norm',
]

def run_clustering_features(session, laps, all_telemetry=None, n_clusters=N_CLUSTERS):
    """
    Full clustering pipeline: corner extraction -> lap aggregation -> normalisation -> GMM.
    Cluster labels are ordered by aggression score each race independently.
    """
    corner_zones = build_corner_zones(session)
    corner_weights = calculate_circuit_weights(session, corner_zones=corner_zones)
    df_corners = build_corner_database(session, laps, all_telemetry=all_telemetry, corner_zones=corner_zones)
    if df_corners.empty:
        return pd.DataFrame()

    df_lap_feats = aggregate_corners_to_laps(df_corners, corner_weights)

    cluster_feature_means = [f'Mean_{f}' for f in CLUSTER_FEATURES]
    df_lap_norm, z_features = normalize_lap_features(df_lap_feats, cluster_feature_means)

    df_clustered = cluster_laps(df_lap_norm, z_features, n_clusters)

    drop_cols = ['LapTime_Sec'] if 'LapTime_Sec' in df_clustered.columns else []
    df_clustered = df_clustered.drop(columns=drop_cols)

    return df_clustered


if __name__ == "__main__":
    start_time = time.time()
    os.makedirs("cache", exist_ok=True)
    os.makedirs("results/clustering/v4", exist_ok=True)
    fastf1.Cache.enable_cache('cache')

    session = fastf1.get_session(2023, 'Silverstone', 'R')
    session.load(telemetry=True, weather=False, messages=False)
    all_laps = session.laps.pick_quicklaps().pick_wo_box().reset_index(drop=True)

    summary_path = "results/clustering/v4/summary.txt"
    summary_lines = []

    log(summary_lines, f"Session: {session.event['EventName']} {session.event.year}")

    corner_zones = build_corner_zones(session)

    # Step 1 — extract corner-level features from telemetry, per driver and lap
    log(summary_lines, "\nExtracting corner features")
    df_corners = build_corner_database(session, all_laps, corner_zones=corner_zones)
    log(summary_lines, f"  {len(df_corners)} corners")

    # Step 2 — aggregate corners into laps
    log(summary_lines, "Aggregating corners to laps")
    corner_weights = calculate_circuit_weights(session, corner_zones=corner_zones)
    df_laps = aggregate_corners_to_laps(df_corners, corner_weights)
    log(summary_lines, f"  {len(df_laps)} laps, {len(df_laps['Driver'].unique())} drivers")

    # Step 3 — normalize features for clustering
    cluster_feature_means = [f'Mean_{f}' for f in CLUSTER_FEATURES]
    df_laps_norm, z_features = normalize_lap_features(df_laps, cluster_feature_means)
    log(summary_lines, "\nFeature ranges after normalisation:")
    log(summary_lines, df_laps_norm[z_features].describe().loc[['min', 'max', 'std']].round(2).to_string())

    # verify cluster amount
    silhouette_scoring(df_laps_norm[z_features].values, max_k=4)

    # Step 4 - fit GMM to cluster features
    log(summary_lines, f"\nFitting GMM with k={N_CLUSTERS} clusters")
    df_laps_clustered = cluster_laps(df_laps_norm, z_features, N_CLUSTERS)

    p_cols = sorted([c for c in df_laps_clustered.columns if c.startswith('P_') and c[2:].isdigit()])
    log(summary_lines, "\nSample output:")
    cols = ['Driver', 'LapNumber'] + p_cols + ['Style_Entropy', 'Style_Cluster_ID']
    log(summary_lines, df_laps_clustered[df_laps_clustered['Driver'] == 'VER'][cols].head(10).to_string())

    log(summary_lines, "\nCluster distribution:")
    log(summary_lines, df_laps_clustered['Style_Cluster_ID'].value_counts().to_string())

    centroid_cols = z_features + ['Style_Cluster_ID']
    log(summary_lines, "\nCluster centroids (Z-scored features):")
    log(summary_lines, df_laps_clustered[centroid_cols].groupby('Style_Cluster_ID').mean().round(3).to_string())

    raw_mean_cols = [f'Mean_{f}' for f in FEATURE_COLS]
    log(summary_lines, "\nCluster centroids (raw feature means):")
    log(summary_lines, df_laps_clustered[raw_mean_cols + ['Style_Cluster_ID']].groupby('Style_Cluster_ID').mean().round(4).to_string())

    csv_path = "results/clustering/v4/lap_clusters.csv"
    df_laps_clustered.to_csv(csv_path, index=False)
    log(summary_lines, f"\nLap-level results saved to {csv_path}")
    log(summary_lines, f"\nClustering completed in {time.time() - start_time:.2f} seconds")

    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f"\nSummary saved to {summary_path}")

    # plots
    plot_lap_clusters_scatter(df_laps_clustered, 'Z_Mean_Apex_Speed_Ratio', 'Z_Mean_Throttle_Integral_Norm', 'VER')
    for drv in df_laps_clustered['Driver'].unique():
        plot_race_timeline(df_laps_clustered, drv)
    plot_probability_distributions(df_laps_clustered)

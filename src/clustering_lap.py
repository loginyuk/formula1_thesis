from scipy.signal import savgol_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import fastf1
import os
import time

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


def calc_erraticness(signal):
    s = pd.Series(signal)
    ma = s.rolling(window=5, min_periods=1).mean()
    return np.mean(np.abs(s - ma))


def extract_all_corner_features(corner_tel):
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

    entry_speed = np.max(speed[:max(1, apex_idx)])
    if entry_speed < 5:
        return None

    # entry
    apex_speed_ratio = apex_speed / entry_speed

    min_speed = np.min(speed)
    speed_loss_frac = (entry_speed - min_speed) / entry_speed

    # braking
    brake_fraction = np.sum(brake) / len(brake)

    if np.any(brake == 1):
        brake_start_idx  = np.argmax(brake == 1)
        brake_start_dist = dist[brake_start_idx]
        brake_point_norm = (brake_start_dist - dist[0]) / corner_len
    else:
        brake_point_norm = 1.0

    # exit
    post_apex_mask = dist > apex_dist
    remaining_len  = max(dist[-1] - apex_dist, 1.0)

    throttle_post  = throttle[post_apex_mask]
    dist_post      = dist[post_apex_mask]

    on_mask = throttle_post > 20
    if np.any(on_mask):
        throttle_on_dist_norm = (dist_post[on_mask][0] - apex_dist) / remaining_len
    else:
        throttle_on_dist_norm = 1.0

    if len(throttle_post) > 1:
        throttle_integral_norm = (
            np.trapz(throttle_post, dist_post) / (100.0 * remaining_len)
        )
    else:
        throttle_integral_norm = throttle_post[0] / 100.0 if len(throttle_post) else 0.0

    # smoothness
    speed_cv = np.std(speed) / (entry_speed + 1e-6)

    return {
        # entry commitment
        'Apex_Speed_Ratio':           apex_speed_ratio,
        'Speed_Loss_Frac':            speed_loss_frac,
        # braking character
        'Brake_Fraction':             brake_fraction,
        'Brake_Point_Norm':           brake_point_norm,
        # exit aggression
        'Throttle_On_Dist_Norm':      throttle_on_dist_norm,
        'Throttle_Integral_Norm':     throttle_integral_norm,
        # smoothness
        'Speed_CV':                   speed_cv,
    }


def build_mjrt_corner_database(session, laps):
    """
    Builds a corner-level database of driving style metrics.
    Loads continuous telemetry per driver
    """
    corners_info = session.get_circuit_info().corners

    corner_zones = {}
    for _, corner in corners_info.iterrows():
        corner_id = f"{corner['Number']}{corner['Letter']}"
        corner_zones[corner_id] = (corner['Distance'] - 100, corner['Distance'] + 100)

    all_corners_data = []
    drivers = laps['Driver'].unique()

    for drv in drivers:
        drv_laps = laps[laps['Driver'] == drv]
        if drv_laps.empty:
            continue

        try:
            full_tel = session.laps.pick_drivers(drv).get_telemetry()
            if 'Distance' not in full_tel.columns:
                full_tel.add_distance()
            full_tel = add_curvature_to_telemetry(full_tel)
        except Exception as e:
            print(f"Failed to load full telemetry for {drv}: {e}")
            continue

        for _, lap in drv_laps.iterrows():
            try:
                lap_start_time = lap['LapStartTime']
                lap_end_time = lap['Time']

                mask = (full_tel['SessionTime'] >= lap_start_time) & (full_tel['SessionTime'] <= lap_end_time)
                lap_tel = full_tel.loc[mask].copy()

                if lap_tel.empty:
                    continue

                lap_dist = lap_tel['Distance'] - lap_tel['Distance'].min()
                lap_tel['Distance'] = lap_dist

                for corner_id, (start_m, end_m) in corner_zones.items():
                    corner_tel = lap_tel[
                        (lap_tel['Distance'] >= start_m) &
                        (lap_tel['Distance'] <= end_m)
                    ].copy()

                    metrics = extract_all_corner_features(corner_tel)
                    if metrics is None:
                        continue

                    metrics['Driver'] = drv
                    metrics['LapNumber'] = lap['LapNumber']
                    metrics['LapTime_Sec'] = lap['LapTime'].total_seconds() if hasattr(lap['LapTime'], 'total_seconds') else lap['LapTime']
                    metrics['Corner_ID'] = corner_id
                    all_corners_data.append(metrics)

            except Exception as e:
                print(f"Error on lap {lap['LapNumber']} for {drv}: {e}")
                continue

    return pd.DataFrame(all_corners_data)


def normalize_by_corner(df, feature_cols, clip_sigma=3.0):
    """
    Normalizes features per corner using median and IQR, then clips to clip_sigma
    """
    df_norm = df.copy()
    z_cols = []

    for col in feature_cols:
        z_col = f'Z_{col}'
        z_cols.append(z_col)
        df_norm[z_col] = df_norm.groupby('Corner_ID')[col].transform(
            lambda x: (x - x.median()) / (x.quantile(0.75) - x.quantile(0.25) + 1e-9)
        )

    df_norm[z_cols] = df_norm[z_cols].fillna(0).clip(-clip_sigma, clip_sigma)
    return df_norm, z_cols


def normalize_dict(dictionary):
    min_val = min(dictionary.values())
    max_val = max(dictionary.values())
    if max_val == min_val:
        return {k: 0.5 for k in dictionary}
    return {k: (v - min_val) / (max_val - min_val) for k, v in dictionary.items()}


def calculate_corner_weights(lap_telemetry, corners_dict, alpha=0.5, beta=0.5):
    w_time_raw, w_energy_raw = {}, {}

    for corner_id, bounds in corners_dict.items():
        corner_tel = lap_telemetry[
            (lap_telemetry['Distance'] >= bounds['start']) &
            (lap_telemetry['Distance'] <= bounds['end'])
        ]
        if len(corner_tel) == 0:
            continue
        
        # thermal weight
        speed_ms = corner_tel['Speed'].values / 3.6
        curvature = corner_tel['Curvature'].values
        lateral_g = np.abs(curvature) * (speed_ms ** 2)
        w_energy_raw[corner_id] = np.sum(lateral_g * speed_ms)

        # traction weight
        post_corner_tel = lap_telemetry[lap_telemetry['Distance'] > bounds['end']]
        brake_zones = post_corner_tel[post_corner_tel['Brake'] == 1]
        next_brake_dist = brake_zones.iloc[0]['Distance'] if not brake_zones.empty else lap_telemetry['Distance'].max()
        straight_tel = post_corner_tel[post_corner_tel['Distance'] < next_brake_dist]

        if straight_tel.empty:
            ft_distance = 0.0
        else:
            dist_deltas = np.diff(straight_tel['Distance'].values, prepend=straight_tel['Distance'].iloc[0])
            is_full_throttle = straight_tel['Throttle'].astype(float).values >= 99
            ft_distance = np.sum(dist_deltas[is_full_throttle])
        w_time_raw[corner_id] = ft_distance

    w_time_norm = normalize_dict(w_time_raw)
    w_energy_norm = normalize_dict(w_energy_raw)

    final_weights = {}
    for corner_id in corners_dict.keys():
        if corner_id in w_time_norm and corner_id in w_energy_norm:
            final_weights[corner_id] = (alpha * w_time_norm[corner_id]) + (beta * w_energy_norm[corner_id])

    return final_weights


def calculate_circuit_weights(session):
    lap = session.laps.pick_quicklaps().pick_fastest()
    lap_tel = lap.get_telemetry()
    corners_info = session.get_circuit_info().corners

    corners_dict = {
        f"{c['Number']}{c['Letter']}": {'start': c['Distance'] - 100, 'end': c['Distance'] + 100}
        for _, c in corners_info.iterrows()
    }
    lap_tel = add_curvature_to_telemetry(lap_tel)
    return calculate_corner_weights(lap_tel, corners_dict)


def select_n_clusters_bic(X, max_k=6):
    """
    Finds the optimal number of GMM clusters using BIC
    """
    bics = {}
    for k in range(2, max_k + 1):
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42, n_init=5)
        gmm.fit(X)
        bics[k] = gmm.bic(X)
        print(f"k={k}  BIC={bics[k]:.1f}")
    best_k = min(bics, key=bics.get)
    print(f"Best k by BIC: {best_k}")
    return best_k

def run_silhouette_analysis(X_for_bic, best_k):
    bic_scores = {}
    for k in range(2, min(best_k, 4) + 1):
        g = GaussianMixture(n_components=k, covariance_type='full', random_state=42, n_init=10)
        g.fit(RobustScaler().fit_transform(X_for_bic))
        bic_scores[k] = g.bic(RobustScaler().fit_transform(X_for_bic))

    # pick k with best silhouette
    best_sil, best_k_sil = -1, 3
    for _k in range(2, min(best_k, 4) + 1):
        g = GaussianMixture(n_components=k, covariance_type='full',
                             random_state=42, n_init=10)
        Xs = RobustScaler().fit_transform(X_for_bic)
        g.fit(Xs)
        lbl = g.predict(Xs)
        if len(np.unique(lbl)) < 2:
            continue
        try:
            s = silhouette_score(Xs, lbl, sample_size=min(5000, len(Xs)))
        except Exception:
            s = -1
        if s > best_sil:
            best_sil, best_k_sil = s, k
    
    print(f"Best silhouette score: {best_sil:.4f} at k={best_k_sil}")
    return best_k_sil, bic_scores

def align_labels_to_reference(new_means, ref_means):
    """
    Aligns new cluster means to reference means using the Hungarian algorithm
    """
    cost = cdist(ref_means, new_means)
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {new: ref for ref, new in zip(row_ind, col_ind)}
    return mapping


def fit_gmm_clustering(X, n_clusters=3, reference_means=None):
    """
    Fits GMM to the data, returns aligned probabilities and labels
    """
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42, n_init=10)
    gmm.fit(X_scaled)

    raw_proba = gmm.predict_proba(X_scaled)   # (N, k)
    raw_labels = np.argmax(raw_proba, axis=1)

    # align cluster means to reference if provided, otherwise sort by first feature
    if reference_means is None:
        cluster_means = np.array([X_scaled[raw_labels == c].mean(axis=0)
                                  for c in range(n_clusters)])
        order = np.argsort(cluster_means[:, 0])[::-1]
        label_map = {old: new for new, old in enumerate(order)}
        reference_means = cluster_means[order]
    else:
        current_means = np.array([X_scaled[raw_labels == c].mean(axis=0)
                                  for c in range(n_clusters)])
        label_map = align_labels_to_reference(current_means, reference_means)

    aligned_proba = np.zeros_like(raw_proba)
    for raw_id, aligned_id in label_map.items():
        aligned_proba[:, aligned_id] = raw_proba[:, raw_id]

    aligned_labels = np.argmax(aligned_proba, axis=1)
    return gmm, scaler, aligned_proba, aligned_labels, reference_means

def weighted_avg(group, col):
    w = group['corner_w']
    return (group[col] * w).sum() / (w.sum() + 1e-9)

def lap_stats(group, prob_cols):
    mid = len(group) // 2
    first_half  = group.iloc[:mid]
    second_half = group.iloc[mid:]
    result = {}
    for col in prob_cols:
        result[col] = weighted_avg(group, col)
        result[f'FH_{col}'] = weighted_avg(first_half,  col) if len(first_half)  > 0 else result[col]
        result[f'SH_{col}'] = weighted_avg(second_half, col) if len(second_half) > 0 else result[col]
    p_arr = np.clip([result[c] for c in prob_cols], 1e-9, 1)
    result['Style_Entropy'] = -np.sum(p_arr * np.log(p_arr))
    return pd.Series(result)

def aggregate_corner_probs_to_lap(df_corners, proba, corner_weights):
    """
    Aggregates corner-level probabilities to lap-level by weighted averaging,
    then computes lap-level features like entropy and dominant style
    """
    n_k = proba.shape[1]
    prob_cols = []
    if n_k == 3:
        prob_cols = ['P_Push', 'P_Balanced', 'P_Save']
    else:
        prob_cols = [f'P_{i}' for i in range(n_k)]

    df = df_corners.copy()
    for i, col in enumerate(prob_cols):
        df[col] = proba[:, i]
    df['Corner_Style_ID'] = np.argmax(proba, axis=1)
    df['corner_w'] = df['Corner_ID'].map(corner_weights).fillna(1.0)

    lap_probs = (df.groupby(['Driver', 'LapNumber'])
                    .apply(lambda g: lap_stats(g, prob_cols), include_groups=False).reset_index())

    lap_probs['Dominant_Style'] = lap_probs[prob_cols].values.argmax(axis=1)
    lap_probs['Style_Cluster_ID'] = lap_probs['Dominant_Style']

    # attach lap times
    lap_times = df_corners[['Driver', 'LapNumber', 'LapTime_Sec']].drop_duplicates()
    df_laps = pd.merge(lap_probs, lap_times, on=['Driver', 'LapNumber'])

    return df_laps


def cluster_mjrt_driving_style(df_corners, needed_features, corner_weights, n_clusters=3, reference_means=None, run_bic_check=False):
    """
    Main clustering function: fits GMM, aligns labels, and aggregates to lap-level.
    If run_bic_check=True, it will first run BIC to suggest optimal k before clustering
    """
    X = df_corners[needed_features].values

    if run_bic_check:
        print("Running BIC model selection…")
        select_n_clusters_bic(X)

    gmm, scaler, proba, hard_labels, ref_means = fit_gmm_clustering(X, n_clusters=n_clusters, reference_means=reference_means)

    df_corners = df_corners.copy()

    # add aligned probabilities and hard labels to corners dataframe
    for i in range(proba.shape[1]):
        df_corners[f'P_{i}'] = proba[:, i]
    df_corners['Corner_Style_ID'] = hard_labels

    df_laps = aggregate_corner_probs_to_lap(df_corners, proba, corner_weights)

    return df_corners, df_laps, ref_means


# Plotting

STYLE_LABELS = {0: 'Push / Attack', 1: 'Balanced / Race Pace', 2: 'Save / Traffic'}
STYLE_PALETTE = {0: '#e74c3c', 1: '#f39c12', 2: '#2ecc71'}


def plot_style_clusters_xy(df_corners_norm, x_col, y_col, driver_code, label_col='Corner_Style_ID'):
    drv = df_corners_norm[df_corners_norm['Driver'] == driver_code]
    plt.figure(figsize=(10, 7))
    for cid, label in STYLE_LABELS.items():
        subset = drv[drv[label_col] == cid]
        plt.scatter(subset[x_col], subset[y_col],
                    label=label, alpha=0.5, s=20, color=STYLE_PALETTE[cid])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{driver_code} — Corner Clusters (GMM)")
    plt.legend()
    plt.tight_layout()
    os.makedirs("results/clustering/v2", exist_ok=True)
    plt.savefig(f"results/clustering/v2/{driver_code}_style_clusters_gmm.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_race_pace_timeline(df_laps, driver_code="VER"):
    """
    Plots lap time timeline with points colored by dominant style cluster,
    and a stacked bar of style probabilities below
    """
    driver_data = df_laps[df_laps['Driver'] == driver_code].sort_values('LapNumber')

    # lap time line plot with cluster-colored points
    named = [c for c in df_laps.columns if c in ('P_Push', 'P_Balanced', 'P_Save')]
    numeric = sorted([c for c in df_laps.columns if c.startswith('P_') and c[2:].isdigit()])
    prob_cols = named if named else numeric

    base_labels  = {0: 'Push / Attack', 1: 'Balanced / Race Pace', 2: 'Save / Traffic'}
    base_colours = {0: '#e74c3c', 1: '#f39c12', 2: '#2ecc71'}
    extra_colours = ['#3498db', '#9b59b6', '#1abc9c']
    n_k = len(prob_cols)
    labels  = {i: base_labels.get(i,  f'Cluster {i}') for i in range(n_k)}
    palette = {i: base_colours.get(i, extra_colours[i - 3]) for i in range(n_k)}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    ax1.plot(driver_data['LapNumber'], driver_data['LapTime_Sec'],
             color='gray', linestyle='-', alpha=0.3, zorder=1)
    for cid, label in labels.items():
        subset = driver_data[driver_data['Style_Cluster_ID'] == cid]
        ax1.scatter(subset['LapNumber'], subset['LapTime_Sec'],
                    label=label, color=palette[cid],
                    s=90, edgecolor='black', linewidth=0.5, zorder=2)
    ax1.set_ylabel("Lap Time (s)")
    ax1.set_title(f"{driver_code} — Race Pace + Driving Style (GMM proportions, k={n_k})")
    ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # stacked probability bars
    laps = driver_data['LapNumber'].values
    bottom = np.zeros(len(driver_data))
    for i, col in enumerate(prob_cols):
        vals = driver_data[col].values
        ax2.bar(laps, vals, bottom=bottom, color=palette[i], label=col, alpha=0.85)
        bottom += vals
    ax2.set_ylabel("Style Probability")
    ax2.set_xlabel("Lap Number")
    ax2.set_ylim(0, 1)
    ax2.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    os.makedirs("results/clustering/v2", exist_ok=True)
    plt.savefig(f"results/clustering/v2/{driver_code}_race_pace_timeline_gmm.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_proportion_distribution(df_laps):
    """
    Histogram of lap-level cluster probabilities
    """
    named   = [c for c in df_laps.columns if c in ('P_Push', 'P_Balanced', 'P_Save')]
    numeric = sorted([c for c in df_laps.columns if c.startswith('P_') and c[2:].isdigit()])
    prob_cols = named if named else numeric

    all_colours = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6', '#1abc9c']

    fig, axes = plt.subplots(1, len(prob_cols), figsize=(5 * len(prob_cols), 4))
    
    if len(prob_cols) == 1:
        axes = [axes]
        
    for i, (ax, col) in enumerate(zip(axes, prob_cols)):
        color = all_colours[i % len(all_colours)]
        
        ax.hist(df_laps[col].dropna(), bins=30, color=color, alpha=0.8, edgecolor='white')
        ax.set_title(f'Distribution: {col}')
        ax.set_xlabel("Probability")
        ax.set_ylabel("Count")
        
    plt.suptitle(f"Lap-Level Style Probabilities (GMM, k={len(prob_cols)})")
    plt.tight_layout()
    os.makedirs("results/clustering/v2", exist_ok=True)
    plt.savefig("results/clustering/v2/proportion_distributions.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    start_time = time.time()
    os.makedirs("cache", exist_ok=True)
    fastf1.Cache.enable_cache('cache')

    session = fastf1.get_session(2023, 'Silverstone', 'R')
    session.load(telemetry=True, weather=False, messages=False)
    all_laps = session.laps.pick_quicklaps().pick_wo_box().reset_index(drop=True)

    df_corners = build_mjrt_corner_database(session, all_laps)

    
    all_extracted_features = ['Apex_Speed_Ratio', 'Speed_Loss_Frac', 'Brake_Fraction', 'Brake_Point_Norm', 
                            'Throttle_On_Dist_Norm', 'Throttle_Integral_Norm', 'Speed_CV']

    df_corners_norm, z_features = normalize_by_corner(df_corners, all_extracted_features)

    corner_weights = calculate_circuit_weights(session)
    
    # features to be run on
    features = ['Z_Apex_Speed_Ratio', 'Z_Speed_Loss_Frac', 'Z_Brake_Fraction', 
                'Z_Brake_Point_Norm', 'Z_Throttle_On_Dist_Norm', 'Z_Throttle_Integral_Norm']

    print(df_corners_norm[features].describe().loc[['min', 'max', 'std']].round(2))


    # run BIC to find amount of clusters
    X_for_bic = df_corners_norm[features].values
    print("\nRunning BIC")
    best_k = select_n_clusters_bic(X_for_bic)

    # calculate silhouette scores to check cluster quality
    best_k_sil, bic_scores = run_silhouette_analysis(X_for_bic, best_k)

    n_clusters_final = best_k_sil

    df_corners_norm, df_laps, reference_means = cluster_mjrt_driving_style(
        df_corners_norm, features, corner_weights,
        n_clusters=n_clusters_final,
        run_bic_check=False
    )

    # demonstrate results
    p_display = sorted([c for c in df_laps.columns if c.startswith('P_') and c[2:].isdigit()])
    print("\nSample lap-level features:")
    print(df_laps[['Driver', 'LapNumber'] + p_display + ['Style_Entropy', 'Style_Cluster_ID']].head(10))

    print("\nCluster distribution:")
    print(df_laps['Style_Cluster_ID'].value_counts())

    plot_style_clusters_xy(df_corners_norm, 'Z_Speed_Loss_Frac', 'Z_Throttle_On_Dist_Norm', 'VER')
    plot_race_pace_timeline(df_laps, 'VER')
    plot_proportion_distribution(df_laps)

    print(f"\nClustering completed in {time.time() - start_time:.2f} seconds")
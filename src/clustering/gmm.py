import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from src.config import FEATURE_COLS
from .corners import build_corner_zones
from .corners import add_curvature_to_telemetry


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


def calculate_circuit_weights(session, corner_zones=None, all_telemetry=None):
    """
    Calculates corner weights based on their contribution to lap time and energy
    """
    if corner_zones is None:
        corner_zones = build_corner_zones(session)

    lap = session.laps.pick_quicklaps().pick_fastest()

    drv_abbr = lap['Driver'] if 'Driver' in lap.index else None
    if all_telemetry and drv_abbr in all_telemetry:
        lap_tel = add_curvature_to_telemetry(all_telemetry[drv_abbr].copy())
    else:
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

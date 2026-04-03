import pandas as pd

from src.config import N_CLUSTERS, CLUSTER_FEATURES
from .corners import build_corner_zones, build_corner_database
from .gmm import calculate_circuit_weights, aggregate_corners_to_laps, normalize_lap_features, cluster_laps


def run_clustering_features(session, laps, all_telemetry=None, n_clusters=N_CLUSTERS, circuit_info=None):
    """
    Full clustering pipeline: corner extraction -> lap aggregation -> normalisation -> GMM.
    Cluster labels are ordered by aggression score each race independently.
    """
    corner_zones = build_corner_zones(circuit_info=circuit_info) if circuit_info else build_corner_zones(session=session)
    corner_weights = calculate_circuit_weights(session, corner_zones=corner_zones, all_telemetry=all_telemetry)
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

import numpy as np
from sklearn.mixture import GaussianMixture

from src.config import RANDOM_STATE


def fit_gmm(X, n_clusters):
    """
    Fits Gaussian Mixture Model to normalized data
    """
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=RANDOM_STATE, n_init=10)
    gmm.fit(X)

    raw_proba = gmm.predict_proba(X)
    raw_labels = np.argmax(raw_proba, axis=1)

    means = np.array([X[raw_labels == c].mean(axis=0) for c in range(n_clusters)])

    # order by aggression: −mean_throttle_on_dist + mean_throttle_integral
    aggression = (-means[:, 1] + means[:, 2])
    order = np.argsort(aggression)[::-1]
    label_map = {old: new for new, old in enumerate(order)}
    reference_means = means[order]

    # align probabilities to the new order
    aligned = np.zeros_like(raw_proba)
    for raw_id, aligned_id in label_map.items():
        aligned[:, aligned_id] = raw_proba[:, raw_id]

    return gmm, aligned, np.argmax(aligned, axis=1), reference_means


def cluster_laps(df_laps_norm, z_features, n_clusters):
    """
    Pipeline to cluster laps based on normalized features.
    assign probabilities and entropy
    """
    X = df_laps_norm[z_features].values
    _, proba, labels, _ = fit_gmm(X, n_clusters)

    df = df_laps_norm.copy()
    for i in range(proba.shape[1]):
        df[f'P_{i}'] = proba[:, i]
    df['Style_Cluster_ID'] = labels.astype(int)

    # calculate entropy
    with np.errstate(divide='ignore'):
        log_p = np.where(proba > 0, np.log(proba), 0.0)
    df['Style_Entropy'] = -np.sum(proba * log_p, axis=1)

    return df

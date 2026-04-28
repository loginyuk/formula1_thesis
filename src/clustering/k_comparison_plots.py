import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.config import PALETTE, RANDOM_STATE


def plot_centroid_radar(fitted, k_range, feature_labels, out_dir):
    """Radar/spider charts of cluster centroids per K."""
    n_feat = len(feature_labels)
    angles = np.linspace(0, 2 * np.pi, n_feat, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, axes = plt.subplots(1, len(k_range), figsize=(5 * len(k_range), 5),
                             subplot_kw=dict(polar=True))

    for ax, k in zip(axes, k_range):
        centroids = fitted[k]['centroids']
        for cid in range(k):
            vals = centroids[cid].tolist() + [centroids[cid][0]]
            ax.plot(angles, vals, 'o-', color=PALETTE[cid], lw=2, label=f'C{cid}', ms=5)
            ax.fill(angles, vals, color=PALETTE[cid], alpha=0.08)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_labels, fontsize=8)
        ax.set_title(f'K = {k}', fontsize=13, pad=15)
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=8)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f'{out_dir}/centroid_radar.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_cluster_sizes(fitted, k_range, out_dir):
    """Cluster size distribution bar chart per K."""
    fig, axes = plt.subplots(1, len(k_range), figsize=(4 * len(k_range), 4))

    for ax, k in zip(axes, k_range):
        labels = fitted[k]['labels']
        sizes = np.bincount(labels, minlength=k)
        pcts = sizes / sizes.sum() * 100
        bars = ax.bar(range(k), pcts, color=[PALETTE[i] for i in range(k)],
                      edgecolor='white', linewidth=1.2)
        for bar, pct, cnt in zip(bars, pcts, sizes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{pct:.1f}%\n({cnt})', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(range(k))
        ax.set_xticklabels([f'C{i}' for i in range(k)])
        ax.set_ylabel('% of laps')
        ax.set_title(f'K = {k}')
        ax.set_ylim(0, max(pcts) + 12)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f'{out_dir}/cluster_sizes.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_pca_scatter(X, fitted, k_range, out_dir, random_state=RANDOM_STATE):
    """PCA projection of clusters per K."""
    pca = PCA(n_components=2, random_state=random_state)
    X2d = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, len(k_range), figsize=(5 * len(k_range), 4.5))

    for ax, k in zip(axes, k_range):
        labels = fitted[k]['labels']
        for cid in range(k):
            mask = labels == cid
            ax.scatter(X2d[mask, 0], X2d[mask, 1], c=PALETTE[cid], s=8,
                       alpha=0.3, rasterized=True, label=f'C{cid}')
        ax.set_xlabel(f'PC1 — {var_explained[0]:.1%} variance')
        ax.set_ylabel(f'PC2 — {var_explained[1]:.1%} variance')
        ax.set_title(f'K = {k}')
        ax.legend(fontsize=8, markerscale=3)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f'{out_dir}/pca_scatter.png', dpi=200, bbox_inches='tight')
    plt.close()

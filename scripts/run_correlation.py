"""
run_correlation.py
──────────────────
Computes and saves the feature correlation matrix and flags highly correlated pairs.

Run from project root:
    python scripts/run_correlation.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MODEL_FEATURES, RESULTS_CORRELATION_DIR, DATASET_ALL

OUT_DIR = RESULTS_CORRELATION_DIR
os.makedirs(OUT_DIR, exist_ok=True)

if __name__ == "__main__":
    df = pd.read_csv(DATASET_ALL)

    available = [f for f in MODEL_FEATURES if f in df.columns]
    missing = [f for f in MODEL_FEATURES if f not in df.columns]
    if missing:
        print(f"Missing columns (skipped): {missing}")

    df_feat = df[available].copy()

    # full correlation matrix
    corr = df_feat.corr()
    corr.to_csv(f'{OUT_DIR}/correlation_matrix.csv')
    print(f"Saved correlation matrix: {len(available)} x {len(available)}")

    fig, ax = plt.subplots(figsize=(28, 24))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(
        corr, mask=mask, ax=ax, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
        linewidths=0.3, annot=False, square=True, cbar_kws={'shrink': 0.6, 'label': 'Pearson r'}
    )
    ax.set_title('Feature Correlation Matrix (all training features)', fontsize=16, pad=12)
    ax.tick_params(axis='x', rotation=90, labelsize=7)
    ax.tick_params(axis='y', rotation=0, labelsize=7)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/correlation_matrix_full.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUT_DIR}/correlation_matrix_full.png")

    # high-correlation pairs (|r| > 0.75)
    corr_abs = corr.abs()
    upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
    high_corr = (
        upper.stack()
        .reset_index()
        .rename(columns={'level_0': 'Feature_A', 'level_1': 'Feature_B', 0: 'abs_r'})
        .sort_values('abs_r', ascending=False)
    )
    high_corr['r'] = [corr.loc[a, b] for a, b in zip(high_corr['Feature_A'], high_corr['Feature_B'])]
    high_corr_filtered = high_corr[high_corr['abs_r'] > 0.75]

    print(f"\nHigh-correlation pairs (|r| > 0.75): {len(high_corr_filtered)}")
    print(high_corr_filtered.to_string(index=False))
    high_corr_filtered.to_csv(f'{OUT_DIR}/high_correlation_pairs.csv', index=False)

    # focused heatmap
    flagged = sorted(set(high_corr_filtered['Feature_A']) | set(high_corr_filtered['Feature_B']))
    if flagged:
        corr_flagged = corr.loc[flagged, flagged]
        fig2, ax2 = plt.subplots(figsize=(max(10, len(flagged) * 0.55), max(8, len(flagged) * 0.5)))
        mask2 = np.zeros_like(corr_flagged, dtype=bool)
        mask2[np.triu_indices_from(mask2)] = True
        sns.heatmap(
            corr_flagged, mask=mask2, ax=ax2, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            annot=True, fmt='.2f', linewidths=0.4, annot_kws={'size': 7}, square=True,
            cbar_kws={'shrink': 0.7, 'label': 'Pearson r'}
        )
        ax2.set_title(f'Highly Correlated Features (|r| > 0.75) — {len(flagged)} features', fontsize=13, pad=10)
        ax2.tick_params(axis='x', rotation=90, labelsize=8)
        ax2.tick_params(axis='y', rotation=0, labelsize=8)
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/correlation_high_only.png', dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {OUT_DIR}/correlation_high_only.png")

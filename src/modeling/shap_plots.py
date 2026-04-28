import os
import numpy as np
import matplotlib.pyplot as plt

from src.config import PRIMARY_COLOUR, PALETTE

def plot_shap_global(global_df, out_dir, top_n=20):
    """
    Plots the global SHAP feature importance as a horizontal bar chart
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    top = global_df.head(top_n).iloc[::-1]
    ax.barh(top['Feature'], top['Mean_Abs_SHAP'], color=PRIMARY_COLOUR, alpha=0.88)
    for y, v in enumerate(top['Mean_Abs_SHAP']):
        ax.text(v, y, f' {v:.4f}', va='center', fontsize=9)
    ax.set_xlabel('Mean |SHAP|  (s)')
    ax.set_xlim(0, top['Mean_Abs_SHAP'].max() * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_global.png"), dpi=200, bbox_inches='tight')
    plt.close()


def plot_shap_by_lag(bucket_df, out_dir):
    """
    Plots SHAP values by lag as a bar chart
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colours = [PALETTE[i % len(PALETTE)] for i in range(len(bucket_df))]
    bars = ax.bar(bucket_df['Bucket'], bucket_df['Total_Abs_SHAP'], color=colours, alpha=0.88)
    for bar, n, m in zip(bars, bucket_df['N_Features'], bucket_df['Mean_Abs_SHAP']):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f' n={n}\n μ={m:.4f}',
                ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Sum of mean |SHAP| (s)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, ls='--', alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_by_lag.png"), dpi=200, bbox_inches='tight')
    plt.close()


def plot_shap_over_races(per_race_df, global_df, races, out_dir, top_n=8):
    """
    Plots SHAP values over races for the top features
    """
    top_features = global_df.head(top_n)['Feature'].tolist()
    pivot = per_race_df[per_race_df['Feature'].isin(top_features)].pivot(
        index='Race', columns='Feature', values='Mean_Abs_SHAP'
    )
    race_order = [r for r in races if r in pivot.index]
    pivot = pivot.loc[race_order, top_features]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(pivot))
    for j, feat in enumerate(top_features):
        ax.plot(x, pivot[feat].values,
                label=feat, color=PALETTE[j % len(PALETTE)],
                linewidth=1.4, alpha=0.9)
    ax.set_xticks(x[::6])
    ax.set_xticklabels([pivot.index[i] for i in x[::6]], rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('Mean |SHAP| per race (s)')
    ax.set_xlabel('Test race (chronological)')
    ax.legend(frameon=False, fontsize=8, ncol=2, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, ls='--', alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_over_races.png"), dpi=200, bbox_inches='tight')
    plt.close()
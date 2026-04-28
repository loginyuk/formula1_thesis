import pandas as pd

from src.config import FEATURE_COLS

def weighted_mean(g, col):
    """
    Calculate importance-weighted mean of a column for a group.
    """
    valid = g[[col, 'corner_w']].dropna(subset=[col])
    total_w = valid['corner_w'].sum()
    if valid.empty or total_w == 0:
        return float('nan')
    return (valid[col] * valid['corner_w']).sum() / total_w


def lap_std_mean(g):
    """
    Calculate mean and std of lap-level features
    """
    result = {}
    for col in FEATURE_COLS:
        result[f'Mean_{col}'] = weighted_mean(g, col)
        result[f'Std_{col}'] = g[col].std(skipna=True)
    return pd.Series(result)


def aggregate_corners_to_laps(df_corners, corner_weights):
    """
    Aggregates corner-level features to lap-level by weighted averaging
    """
    df = df_corners.copy()
    df['corner_w'] = df['Corner_ID'].map(corner_weights).fillna(1.0)

    df_laps = (df.groupby(['Driver', 'LapNumber']).apply(lap_std_mean, include_groups=False).reset_index())

    lap_times = df_corners[['Driver', 'LapNumber', 'LapTime_Sec']].drop_duplicates()
    df_laps = pd.merge(df_laps, lap_times, on=['Driver', 'LapNumber'])
    return df_laps

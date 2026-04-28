def normalize_lap_features(df_laps, feature_cols, clip_sigma=3.0):
    """
    Normalizes lap-level features using robust scaling (median + IQR)
    """
    df = df_laps.copy()
    z_cols = []

    for col in feature_cols:
        z = f'Z_{col}'
        median = df[col].median()
        iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
        if iqr > 0:
            df[z] = ((df[col] - median) / iqr).clip(-clip_sigma, clip_sigma).fillna(0.0)
        else:
            df[z] = 0.0
        z_cols.append(z)
    return df, z_cols

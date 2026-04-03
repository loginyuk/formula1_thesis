from sklearn.preprocessing import LabelEncoder
from src.utils import log


def remove_wet_laps(df, summary_lines):
    """
    Cleans wet laps from a dataframe
    """
    df_dry = df.copy()

    # remove all laps during or after rainfall starts
    rainy_laps = df_dry[df_dry['Rainfall'] == True]

    if not rainy_laps.empty:
        first_rain_time = rainy_laps['Time'].min()
        df_dry = df_dry[df_dry['Time'] < first_rain_time]
        log(summary_lines, f"Rain detected at {first_rain_time}. Dropped all subsequent laps.")

    log(summary_lines, f"Total Laps input: {len(df)}")
    log(summary_lines, f"Valid Dry Laps saved: {len(df_dry)}\n")

    return df_dry


def clean_laps(df, summary_lines):
    """
    Deletes in/out, SC/VSC laps and keeps only needed columns
    """
    # remove rain
    df_dry = remove_wet_laps(df, summary_lines)

    # filter green flag laps with real timing and no pit stops
    mask = (
        (df_dry['TrackStatus'] == '1') &       # green flag only
        (df_dry['FastF1Generated'] == False) & # real timing only
        (df_dry['PitInTime'].isna()) &         # no pit in laps
        (df_dry['PitOutTime'].isna()) &        # no pit out laps
        (df_dry['Deleted'] == False)           # not deleted laps
    )
    df_clean = df_dry.loc[mask].copy()

    # clean up columns
    cols_to_keep = ['Time', 'Year', 'RoundNumber', 'Driver', 'LapTime', 'LapNumber', 'Stint', 'Compound',
       'TyreLife', 'Team', 'Location', 'Position', 'AirTemp', 'Humidity', 'Pressure',
       'Rainfall', 'TrackTemp', 'WindDirection', 'WindSpeed', 'LapTime_Sec',
       'FuelLoad', 'Track_Evolution_Physics', 'Traction_1_5',
       'Asphalt_Grip_1_5', 'Asphalt_Abrasion_1_5', 'Track_Evolution_1_5',
       'Tyre_Stress_1_5', 'Braking_1_5', 'Lateral_1_5', 'Downforce_1_5',
       'Min_Pressure_Front_PSI', 'Min_Pressure_Rear_PSI', 'Compound_Hard',
       'Compound_Medium', 'Compound_Soft', 'Circuit_Length_KM', 'Total_Laps',
       'Cumulative_Field_Dist_KM', 'Track_Evolution_Physics.1',
       'Compound_Hard_Hardness', 'Compound_Medium_Hardness', 'Compound_Soft_Hardness',
       'Compound_Hard_Int', 'Compound_Medium_Int', 'Compound_Soft_Int',
       'Wear_Severity_Index', 'Track_Flow_Type', 'Compound_Int',
       'Tyre_Compound_Interaction', 'E_lap',
       'Gap_To_Car_Ahead', 'Dirty_Air_Fraction', 'DRS_Fraction',
       'Tyre_Grip_Index', 'Aero_Loss',
       'P_surface', 'M_aero', 'Lap_Damage',
       'Accumulated_Tyre_Wear', 'Micro_Stint_ID', 'Prev_LapTime',
       'Lag_2', 'Rolling_Avg_3',
       'Grip_Aero_Balance', 'Total_Min_Pressure', 'Pressure_Delta',
       'LatOffset_Mean', 'LatOffset_Std',
       'Mean_Apex_Speed_Ratio', 'Std_Apex_Speed_Ratio',
       'Mean_Brake_Fraction', 'Std_Brake_Fraction',
       'Mean_Brake_Point_Norm', 'Std_Brake_Point_Norm',
       'Mean_Throttle_On_Dist_Norm', 'Std_Throttle_On_Dist_Norm',
       'Mean_Throttle_Integral_Norm', 'Std_Throttle_Integral_Norm',
       'Mean_Speed_CV', 'Std_Speed_CV',
       'P_0', 'P_1', 'P_2', 'Style_Cluster_ID', 'Style_Entropy',
       'Target_Delta', 'Prev_Delta']

    valid_cols = [c for c in cols_to_keep if c in df_clean.columns]

    log(summary_lines, f"Clean Green Laps: {len(df_clean)}\n")

    return df_clean[valid_cols].sort_values(by=['Driver', 'LapNumber']).reset_index(drop=True)


def add_lag_features(df_clean, summary_lines):
    """
    Add to dataframe lag features of previous times
    """
    df_ts = df_clean.copy()

    df_ts = df_ts.sort_values(by=['Driver', 'LapNumber'])
    df_ts['Lap_Gap'] = df_ts.groupby('Driver')['LapNumber'].diff()

    condition = (df_ts['Lap_Gap'] > 1) | (df_ts['Lap_Gap'].isna())

    df_ts['Micro_Stint_ID'] = condition.groupby(df_ts['Driver']).cumsum()
    stint_group = df_ts.groupby(['Driver', 'Micro_Stint_ID'])

    df_ts['Prev_LapTime'] = stint_group['LapTime_Sec'].shift(1)
    df_ts['Lag_2'] = stint_group['LapTime_Sec'].shift(2)
    df_ts['Rolling_Avg_3'] = stint_group['LapTime_Sec'].transform(
        lambda x: x.rolling(window=3).mean().shift(1)
    )

    df_ts['Target_Delta'] = df_ts['LapTime_Sec'] - df_ts['Prev_LapTime']
    df_ts['Prev_Delta'] = df_ts['Prev_LapTime'] - df_ts['Lag_2']

    df_ts = df_ts.dropna(subset=['Prev_LapTime', 'Rolling_Avg_3']).reset_index(drop=True)
    df_ts = df_ts.drop(columns=['Micro_Stint_ID'])

    log(summary_lines, f"Final Laps with History: {len(df_ts)}")
    return df_ts


def encode_categorical_features(df):
    """
    Label encodes 'Driver', 'Location', and 'Team' columns.
    """
    df_encoded = df.copy()
    encoders = {}

    cols_to_encode = ['Driver', 'Location', 'Team']

    for col in cols_to_encode:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[f"{col}_Encoded"] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
        else:
            print(f"Column '{col}' not found in dataframe")

    return df_encoded, encoders

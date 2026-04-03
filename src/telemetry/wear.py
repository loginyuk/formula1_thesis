import numpy as np


def build_accumulated_wear(df_laps):
    df = df_laps.copy()
    df = df.sort_values(by=['Driver', 'LapNumber']).reset_index(drop=True)

    df['P_surface'] = df['Wear_Severity_Index'] * df['Compound_Int']
    df['M_aero'] = np.where(df['Gap_To_Car_Ahead'] < 2.0, 1.15, 1.0)

    df['E_lap'] = df.groupby(['Driver', 'Stint'])['E_lap'].transform(lambda x: x.fillna(x.median()))
    df['E_lap'] = df['E_lap'].fillna(0)

    df['Lap_Damage'] = df['E_lap'] * df['P_surface'] * df['M_aero']
    df['Accumulated_Tyre_Wear'] = df.groupby(['Driver', 'Stint'])['Lap_Damage'].cumsum()

    df['Aero_Loss'] = np.exp(-df['Gap_To_Car_Ahead'] / 1.5)

    wear_norm = df.groupby(['Driver', 'Stint'])['Accumulated_Tyre_Wear'].transform(lambda x: x / (x.max() + 1e-9))
    compound_mu = np.where(df['Compound'] == 'SOFT', 1.2,
                  np.where(df['Compound'] == 'MEDIUM', 1.0, 0.85))
    track_temp_factor = 1.0 + 0.002 * (df['TrackTemp'] - 30.0)
    wear_factor = 1.0 - 0.15 * wear_norm
    df['Tyre_Grip_Index'] = compound_mu * track_temp_factor * wear_factor

    return df

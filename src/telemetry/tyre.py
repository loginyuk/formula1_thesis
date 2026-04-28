import numpy as np

from .curvature import compute_curvature

def calculate_energy(telemetry):
    """
    Calculate an energy proxy for tyre wear for a lap
    """
    # calculate longitudinal acceleration and speed
    dt = telemetry['Time'].dt.total_seconds().diff().fillna(0.05).clip(0.001, 0.5)
    v = telemetry['Speed'] / 3.6
    a_long = (v.diff() / dt).fillna(0)

    # calculate lateral acceleration from curvature and speed
    try:
        x = telemetry['X'].values
        y = telemetry['Y'].values
        a_lat = np.clip((v ** 2) * compute_curvature(x, y), 0, 60.0)
    except Exception:
        a_lat = np.zeros(len(telemetry))

    combined_accel = np.sqrt(a_long ** 2 + a_lat ** 2)
    return (combined_accel * v * dt) / 1000.0


def build_tyre_features(df_laps):
    """
    Calculate tyre wear features for each lap
    """
    df = df_laps.copy()
    df = df.sort_values(by=['Driver', 'LapNumber']).reset_index(drop=True)

    # calculate lap damage and accumulated tyre wear
    P_surface = df['Wear_Severity_Index'] * (df['Compound_Int'] + 1)
    M_aero = np.where(df['Gap_To_Car_Ahead'] < 2.0, 1.15, 1.0)
    df['Lap_Damage'] = df['Energy_Lap'] * P_surface * M_aero
    df['Accumulated_Tyre_Wear'] = df.groupby(['Driver', 'Stint'])['Lap_Damage'].cumsum()

    return df
import numpy as np

def calculate_physics_fuel_load(session, df_merged):
    """
    Calculate physics-based fuel load
    """
    MAX_FUEL_KG = 110.0
    MIN_FUEL_KG = 1.0

    total_laps = session.total_laps
    base_burn_rate = (MAX_FUEL_KG - MIN_FUEL_KG) / total_laps

    MULTIPLIER_SC   = 0.35
    MULTIPLIER_PIT  = 1.20

    df = df_merged.sort_values(by=['Driver', 'LapNumber']).copy()

    # determine burn rate multipliers
    cond_sc = df['TrackStatus'].isin(['4', '6', '7'])
    cond_pit = df['PitInTime'].notna() | df['PitOutTime'].notna()
    cond_green = df['TrackStatus'] == '1'

    df['Lap_Burn'] = np.select(
        [cond_sc, cond_pit, cond_green],
        [base_burn_rate * MULTIPLIER_SC,
            base_burn_rate * MULTIPLIER_PIT,
            base_burn_rate],
        default=base_burn_rate
    )

    df['Fuel_Consumed'] = df.groupby('Driver')['Lap_Burn'].cumsum()
    df['FuelLoad'] = (MAX_FUEL_KG - df['Fuel_Consumed']).clip(lower=MIN_FUEL_KG)
    return df


def calculate_physics_track_evolution(df_laps, df_tracks):
    """
    Calculate physics-based track evolution by merging with Pirelli data.
    """
    df_merged = df_laps.merge(df_tracks, on=['Location', 'Year'], how='left')
    df_merged = df_merged.sort_values(by='Time')

    laps_in_race = df_merged.groupby(['Year', 'Location']).cumcount()
    df_merged['Cumulative_Field_Dist_KM'] = laps_in_race * df_merged['Circuit_Length_KM']
    df_merged['Track_Evolution'] = df_merged['Cumulative_Field_Dist_KM'] * df_merged['Track_Evolution_1_5']

    return df_merged


def build_static_features(session, df_weather, df_pirelli):
    df = calculate_physics_fuel_load(session, df_weather)
    df = calculate_physics_track_evolution(df, df_pirelli)
    return df
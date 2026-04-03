import numpy as np


def calculate_physics_fuel_load(session, df_merged):
    """
    Calculate fuel load based on physics-based burn rate and track status
    - for SC/VSC laps, assume a 65% reduction in fuel burn.
    - for in/out laps, assume a 20% increase in fuel burn.
    """
    MAX_FUEL_KG = 110.0
    MIN_FUEL_KG = 1.0

    total_laps = session.total_laps
    base_burn_rate = (MAX_FUEL_KG - MIN_FUEL_KG) / total_laps

    MULTIPLIER_RACE = 1.00
    MULTIPLIER_SC   = 0.35   # Safety Car / VSC
    MULTIPLIER_SLOW = 1.20   # In-laps / Out-laps

    df = df_merged.sort_values(by=['Driver', 'LapNumber']).copy()

    cond_sc = (df['TrackStatus'].isin(['4', '6', '7']))
    cond_pit = (df['PitInTime'].notna()) | (df['PitOutTime'].notna())
    cond_green = (df['TrackStatus'] == '1')

    df['Instant_Burn'] = np.select(
        [cond_sc, cond_pit, cond_green],
        [
            base_burn_rate * MULTIPLIER_SC,
            base_burn_rate * MULTIPLIER_SLOW,
            base_burn_rate * MULTIPLIER_RACE
        ],
        default=base_burn_rate * MULTIPLIER_RACE
    )

    df['Fuel_Consumed'] = df.groupby('Driver')['Instant_Burn'].cumsum()
    df['FuelLoad'] = MAX_FUEL_KG - df['Fuel_Consumed']
    df['FuelLoad'] = df['FuelLoad'].clip(lower=MIN_FUEL_KG)

    return df


def add_physics_track_evolution(df_laps, df_tracks):
    """
    Add physics-based track evolution features based on cumulative distance driven and Pirelli parameters
    """
    df_merged = df_laps.merge(df_tracks, on=['Location', 'Year'], how='left')

    if df_merged['Track_Evolution_1_5'].isna().any():
        print(f"Missing track evolution data: {df_merged[df_merged['Track_Evolution_1_5'].isna()]['Location'].unique()}")

    df_merged = df_merged.sort_values(by='Time')
    global_lap_count = np.arange(len(df_merged))

    # calculate distance driven
    df_merged['Cumulative_Field_Dist_KM'] = global_lap_count * df_merged['Circuit_Length_KM']

    # grip coefficient
    df_merged['Track_Evolution_Physics'] = (df_merged['Cumulative_Field_Dist_KM'] * df_merged['Track_Evolution_1_5'])

    return df_merged

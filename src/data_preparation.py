import fastf1
import os
import numpy as np
import pandas as pd

from telemetry import run_telemetry_feature_generation

def prepare_race(session):
    """
    Get laps, drivers and stints dataframes for the race
    """
    laps = session.laps
    laps['Location'] = session.event['Location']

    drivers = session.drivers
    drivers = [session.get_driver(driver)["Abbreviation"] for driver in drivers]

    stints = laps[["Driver", "Stint", "Compound", "LapNumber"]]
    stints = stints.groupby(["Driver", "Stint", "Compound"])
    stints = stints.count().reset_index()
    stints = stints.rename(columns={"LapNumber": "StintLength"})

    return laps, drivers, stints


def merge_weather(session, laps):
    """"
    Merge weather data with laps data
    """
    df = laps.copy()

    weather = session.weather_data.copy()

    df = df.sort_values('Time')
    weather = weather.sort_values('Time')

    df_weather = pd.merge_asof(df, weather, on='Time', direction='backward')

    df_weather['LapTime_Sec'] = df_weather['LapTime'].dt.total_seconds()
    return df_weather


def calculate_physics_fuel_load(session, df_merged):
    """"
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
    df_merged = df_laps.merge(df_tracks, on='Location', how='left')

    if df_merged['Track_Evolution_1_5'].isna().any():
        print(f"Missing track evolution data: {df_merged[df_merged['Track_Evolution_1_5'].isna()]['Location'].unique()}")

    df_merged = df_merged.sort_values(by='Time')
    global_lap_count = np.arange(len(df_merged))

    # calculate distance driven
    df_merged['Cumulative_Field_Dist_KM'] = global_lap_count * df_merged['Circuit_Length_KM']

    # grip coefficient
    df_merged['Track_Evolution_Physics'] = (df_merged['Cumulative_Field_Dist_KM'] * df_merged['Track_Evolution_1_5'])

    return df_merged


def add_lag_features(df_clean):
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
    
    df_ts = df_ts.dropna(subset=['Prev_LapTime', 'Rolling_Avg_3']).reset_index(drop=True)
    df_ts = df_ts.drop(columns=['Micro_Stint_ID'])
    
    print(f"Final Laps with History: {len(df_ts)}")
    return df_ts


def remove_wet_laps(df):
    """
    Cleans wet laps from a dataframe
    """
    df_dry = df.copy()
    
    # remove all laps during or after rainfall starts
    rainy_laps = df_dry[df_dry['Rainfall'] == True]
    
    if not rainy_laps.empty:
        first_rain_time = rainy_laps['Time'].min()
        df_dry = df_dry[df_dry['Time'] < first_rain_time]
        print(f"Rain detected at {first_rain_time}. Dropped all subsequent laps.")
    
    # filter wet compounds
    # final_dry_laps = df_dry[~df_dry['Compound'].isin(['INTERMEDIATE', 'WET'])].copy()
    
    print(f"Total Laps Input: {len(df)}")
    print(f"Valid Dry Laps Retained: {len(df_dry)}\n")
    
    return df_dry



def clean_laps(df):
    """
    Deletes in/out, SC/VSC laps and keeps only needed columns
    """
    # remove rain
    df_dry = remove_wet_laps(df)

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
    cols_to_keep = ['Time', 'Driver', 'LapTime', 'LapNumber', 'Stint', 'Compound',
       'TyreLife', 'Team', 'Location', 'AirTemp', 'Humidity', 'Pressure',
       'Rainfall', 'TrackTemp', 'WindDirection', 'WindSpeed', 'LapTime_Sec',
       'FuelLoad', 'Track_Evolution_Physics', 'Traction_1_5',
       'Asphalt_Grip_1_5', 'Asphalt_Abrasion_1_5', 'Track_Evolution_1_5',
       'Tyre_Stress_1_5', 'Braking_1_5', 'Lateral_1_5', 'Downforce_1_5',
       'Min_Pressure_Front_PSI', 'Min_Pressure_Rear_PSI', 'Compound_Hard',
       'Compound_Medium', 'Compound_Soft', 'Circuit_Length_KM', 'Total_Laps',
       'Cumulative_Field_Dist_KM', 'Track_Evolution_Physics.1',
       'Compound_Hard_Int', 'Compound_Medium_Int', 'Compound_Soft_Int',
       'Wear_Severity_Index', 'Track_Flow_Type', 'Compound_Int', 'E_lap',
       'Gap_To_Car_Ahead', 'P_surface', 'M_aero', 'Lap_Damage',
       'Accumulated_Tyre_Wear', 'Micro_Stint_ID', 'Prev_LapTime',
       'Lag_2', 'Rolling_Avg_3',
       'Grip_Aero_Balance', 'Total_Min_Pressure', 'Pressure_Delta']

    valid_cols = [c for c in cols_to_keep if c in df_clean.columns]

    print(f"Clean Green Laps: {len(df_clean)}\n")

    return df_clean[valid_cols].sort_values(by=['Driver', 'LapNumber']).reset_index(drop=True)


def get_pirelli_press_data(file_path):
    """
    Downloads additional Pirelli press dataset
    """
    df = pd.read_csv(file_path)
    return df


if __name__ == "__main__":
    os.makedirs("cache", exist_ok=True)
    fastf1.Cache.enable_cache('cache')

    session = fastf1.get_session(2023, "Silverstone", 'R')
    session.load()

    laps, drivers, stints = prepare_race(session)
    df_weather = merge_weather(session, laps)
    df_fuel = calculate_physics_fuel_load(session, df_weather)

    df_pirelli = get_pirelli_press_data('track_parameters.csv')
    df_track = add_physics_track_evolution(df_fuel, df_pirelli)

    # telemetry features
    df_telemetry = run_telemetry_feature_generation(session, df_track)

    df_clean = clean_laps(df_telemetry)

    df_lag = add_lag_features(df_clean)

    os.makedirs("data", exist_ok=True)
    df_lag.to_csv('data/dataset_with_telemetry.csv', index=False)

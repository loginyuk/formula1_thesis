import fastf1
import os
import logging
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder

from telemetry import run_telemetry_feature_generation
from clustering_lap import plot_cluster_verification

logger = logging.getLogger('data_preparation')

def prepare_race(session):
    """
    Get laps, drivers and stints dataframes for the race
    """
    laps = session.laps
    laps['Location'] = session.event['Location']
    laps['Year'] = session.event.year
    laps['RoundNumber'] = session.event['RoundNumber']

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


def log(summary_lines, *args, **kwargs):
    """
    Prints to terminal and appends to summary_lines for file saving
    """
    text = " ".join(str(a) for a in args)
    print(text, **kwargs)
    summary_lines.append(text)


def get_pirelli_press_data(file_path):
    """
    Downloads additional Pirelli press dataset
    """
    df = pd.read_csv(file_path)
    return df


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


if __name__ == "__main__":
    start_time = time.time()
    os.makedirs("cache", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    fastf1.Cache.enable_cache('cache')

    # file logger for errors across all pipeline modules
    class RaceContextFilter(logging.Filter):
        def __init__(self):
            super().__init__()
            self.race = 'N/A'
        def filter(self, record):
            record.race = self.race
            return True

    race_filter = RaceContextFilter()

    file_handler = logging.FileHandler('logs/errors.log', mode='w')
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(race)s | %(name)s | %(levelname)s | %(message)s'))
    file_handler.addFilter(race_filter)
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().setLevel(logging.WARNING)

    summary_lines = []
    summary_path = "logs/summary_data_preparation.txt"

    df_pirelli_all = get_pirelli_press_data('data/track_parameters.csv')

    YEARS = [2022, 2023, 2024, 2025]
    full_dataset = []

    for YEAR in YEARS:
        df_pirelli = df_pirelli_all[df_pirelli_all['Year'] == YEAR].copy()
        locations = df_pirelli['Location'].unique()

        for location in locations:
            race_filter.race = f"{YEAR} {location}"
            log(summary_lines, f"\n{'-'*55}")
            log(summary_lines, f"Processing: {YEAR} - {location}\n")

            try:
                session = fastf1.get_session(YEAR, location, 'R')
                for attempt in range(3):
                    try:
                        session.load()
                        break
                    except Exception as e:
                        if attempt == 2:
                            raise
                        logger.warning(f"{YEAR} {location}: session.load() attempt {attempt+1} failed: {e}, retrying...")
                        time.sleep(5)

                circuit_info = session.get_circuit_info()

                laps, drivers, stints = prepare_race(session)
                df_weather = merge_weather(session, laps)
                df_fuel = calculate_physics_fuel_load(session, df_weather)
                df_track = add_physics_track_evolution(df_fuel, df_pirelli)

                df_telemetry = run_telemetry_feature_generation(session, df_track, circuit_info=circuit_info)

                df_clean = clean_laps(df_telemetry, summary_lines)
                df_lag = add_lag_features(df_clean, summary_lines)

                full_dataset.append(df_lag)
                log(summary_lines, f"{YEAR} {location} processed {len(df_lag)} rows")

            except Exception as e:
                logger.error(f"{YEAR} {location}: {e}", exc_info=True)
                log(summary_lines, f"Could not process {YEAR} {location}. Error: {e}")
                continue

    if full_dataset:
        df_all = pd.concat(full_dataset, ignore_index=True)

        # encode once across all years
        df_all, label_encoders = encode_categorical_features(df_all)
        df_all.to_csv('data/dataset_all.csv', index=False)

        if 'Style_Cluster_ID' in df_all.columns:
            plot_cluster_verification(df_all)
            log(summary_lines, "Verification plots saved to results/clustering/verification/")
        else:
            log(summary_lines, "Skipping cluster verification — clustering columns missing")

        total_time = (time.time() - start_time) / 60
        log(summary_lines, f"Total time taken: {total_time:.2f} minutes")
        log(summary_lines, f"Total laps: {len(df_all)}")
    else:
        log(summary_lines, "No races were successfully processed")

    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f"Summary saved to {summary_path}")

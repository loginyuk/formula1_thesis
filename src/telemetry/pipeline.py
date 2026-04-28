import logging
import time
import numpy as np

from src.clustering.pipeline import run_clustering_features
from .tyre import calculate_energy, build_tyre_features
from .aero import calculate_dirty_air, build_aero_features
from .line import get_reference_lap, build_racing_line_features

logger = logging.getLogger('telemetry')

def generate_telemetry_features_dataset(session, df_laps, circuit_info=None):
    """
    Calculate all telemetry-derived features.
    tyre model, aerodynamics model, racing line deviation and clustering features call
    """
    df_laps = df_laps[df_laps['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])].reset_index(drop=True)

    df_laps['Energy_Lap'] = np.nan
    df_laps['Gap_To_Car_Ahead'] = 5.0
    df_laps['Dirty_Air_Fraction'] = 0.0
    df_laps['DRS_Fraction'] = 0.0
    df_laps['LatOffset_Mean'] = np.nan
    df_laps['LatOffset_Std'] = np.nan

    active_drivers = df_laps['Driver'].unique()
    ref_nd, ix, iy, window = get_reference_lap(session)
    has_reference_lap = ref_nd is not None

    # load session telemetry once
    all_telemetry = {}
    for drv in active_drivers:
        try:
            d_laps = session.laps.pick_drivers(drv)
            if len(d_laps) > 0:
                tel = d_laps.get_telemetry()
                if len(tel) >= 10:
                    # tyre energy
                    tel['Energy_Tick'] = calculate_energy(tel)
                    all_telemetry[drv] = tel
        except Exception as e:
            logger.warning(f"Telemetry load failed for driver {drv}: {e}")

    # main per-lap feature calculation loop
    for idx, row in df_laps.iterrows():
        driver = row['Driver']
        if driver not in all_telemetry:
            continue

        try:
            lap_obj = session.laps.pick_drivers(driver).pick_laps(row['LapNumber']).iloc[0]
            lap_start = lap_obj['LapStartTime']
            lap_end = lap_obj['Time']

            drv_tel = all_telemetry[driver]
            mask = (drv_tel['SessionTime'] >= lap_start) & (drv_tel['SessionTime'] <= lap_end)
            lap_tel = drv_tel.loc[mask].copy()

            if lap_tel.empty or len(lap_tel) < 10:
                continue

            df_laps.loc[idx, 'Energy_Lap'] = lap_tel['Energy_Tick'].sum()

            # aero
            mean_gap, dirty_frac = calculate_dirty_air(lap_tel)
            df_laps.loc[idx, 'Gap_To_Car_Ahead']  = mean_gap
            df_laps.loc[idx, 'Dirty_Air_Fraction'] = dirty_frac

            # DRS usage
            if 'DRS' in lap_tel.columns:
                df_laps.loc[idx, 'DRS_Fraction'] = lap_tel['DRS'].isin([10, 12, 14]).sum() / len(lap_tel)

            # racing line deviation
            if has_reference_lap:
                build_racing_line_features(df_laps, idx, lap_tel, ref_nd, ix, iy, window)

        except Exception as e:
            logger.warning(f"Driver {driver} lap {row['LapNumber']}: {e}")
            continue

    # driving style clustering
    clustering_laps = df_laps[
        (df_laps['TrackStatus'] == '1') &
        (df_laps['PitInTime'].isna()) &
        (df_laps['PitOutTime'].isna()) &
        (df_laps.get('FastF1Generated', False) == False)
    ].reset_index(drop=True)

    try:
        df_clustered = run_clustering_features(session, clustering_laps, all_telemetry=all_telemetry, circuit_info=circuit_info)
        if not df_clustered.empty:
            df_laps = df_laps.merge(df_clustered, on=['Driver', 'LapNumber'], how='left')
    except Exception as e:
        logger.error(f"Clustering features failed: {e}", exc_info=True)

    df_laps = build_tyre_features(df_laps)
    df_laps = build_aero_features(df_laps)

    return df_laps


def run_telemetry_feature_generation(session, df, circuit_info=None):
    start = time.time()
    df_race = generate_telemetry_features_dataset(session, df, circuit_info=circuit_info)
    print(f"\nTelemetry features time taken: {time.time() - start:.4f} seconds\n")
    return df_race

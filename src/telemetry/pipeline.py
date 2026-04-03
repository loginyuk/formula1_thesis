import logging
import time
import numpy as np

from src.clustering.pipeline import run_clustering_features
from .features import add_derived_features, calculate_energy, calculate_dirty_air, calculate_lateral_offset, get_reference_lap
from .wear import build_accumulated_wear

logger = logging.getLogger('telemetry')


def generate_telemetry_features_dataset(session, df_laps, circuit_info=None):
    """
    Main pipeline function to generate telemetry features dataset
    """
    # filter out wet compounds
    df_laps = df_laps[df_laps['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])].reset_index(drop=True)
    df_laps = add_derived_features(df_laps)

    df_laps['Compound_Int'] = np.where(df_laps['Compound'] == 'SOFT', df_laps['Compound_Soft_Int'],
                            np.where(df_laps['Compound'] == 'MEDIUM', df_laps['Compound_Medium_Int'],
                            df_laps['Compound_Hard_Int']))
    df_laps['Tyre_Compound_Interaction'] = df_laps['TyreLife'] * df_laps['Compound_Int']

    df_laps['E_lap'] = np.nan
    df_laps['Gap_To_Car_Ahead'] = 5.0
    df_laps['Dirty_Air_Fraction'] = 0.0
    df_laps['DRS_Fraction'] = 0.0
    df_laps['LatOffset_Mean'] = np.nan
    df_laps['LatOffset_Std'] = np.nan

    active_drivers = df_laps['Driver'].unique()

    ref_nd, ix, iy, window = get_reference_lap(session)
    has_reference_lap = ref_nd is not None

    # get telemetry
    all_telemetry = {}
    for drv in active_drivers:
        try:
            d_laps = session.laps.pick_drivers(drv)
            if len(d_laps) > 0:
                tel = d_laps.get_telemetry()
                if len(tel) >= 10:
                    tel['Energy_Tick'] = calculate_energy(tel)
                    all_telemetry[drv] = tel
        except Exception as e:
            logger.warning(f"Telemetry load failed for driver {drv}: {e}")

    # map telemetry back to laps
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

            if not lap_tel.empty and len(lap_tel) >= 10:
                df_laps.loc[idx, 'E_lap'] = lap_tel['Energy_Tick'].sum()

                # calculate dirty air metrics
                mean_gap, dirty_frac = calculate_dirty_air(lap_tel, dirty_air_threshold=2.0)
                df_laps.loc[idx, 'Gap_To_Car_Ahead'] = mean_gap
                df_laps.loc[idx, 'Dirty_Air_Fraction'] = dirty_frac

                # DRS %
                if 'DRS' in lap_tel.columns:
                    drs_active = lap_tel['DRS'].isin([10, 12, 14])
                    df_laps.loc[idx, 'DRS_Fraction'] = drs_active.sum() / len(lap_tel)

                # calculate lateral offset metrics
                if has_reference_lap:
                    lap_dist = lap_tel['Distance'] - lap_tel['Distance'].min()
                    lap_tel['NormDist'] = lap_dist / lap_dist.max()
                    lap_tel = lap_tel.sort_values('NormDist').drop_duplicates('NormDist').reset_index(drop=True)

                    lap_offset_tel = calculate_lateral_offset(lap_tel, ref_nd, ix, iy, window)
                    df_laps.loc[idx, 'LatOffset_Mean'] = lap_offset_tel['LateralOffset_m'].mean()
                    df_laps.loc[idx, 'LatOffset_Std'] = lap_offset_tel['LateralOffset_m'].std()

        except Exception as e:
            logger.warning(f"Driver {driver} lap {row['LapNumber']}: {e}")
            continue

    # run clustering
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

    df_laps = build_accumulated_wear(df_laps)

    return df_laps


def run_telemetry_feature_generation(session, df, circuit_info=None):
    start = time.time()
    df_race_wear = generate_telemetry_features_dataset(session, df, circuit_info=circuit_info)

    end = time.time()
    print(f"\nTelemetry features time taken: {end - start:.4f} seconds\n")

    return df_race_wear

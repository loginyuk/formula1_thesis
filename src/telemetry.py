import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import time
from clustering_lap import run_clustering_features

def add_derived_features(df):
    """"
    Add derived features from Pirelli press data to the laps dataframe
    """
    compound_map = {'C0': 0, 'C1': 1, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5}
    df['Compound_Hard_Int'] = df['Compound_Hard'].map(compound_map)
    df['Compound_Medium_Int'] = df['Compound_Medium'].map(compound_map)
    df['Compound_Soft_Int'] = df['Compound_Soft'].map(compound_map)

    df['Wear_Severity_Index'] = df['Asphalt_Abrasion_1_5'] * df['Tyre_Stress_1_5']
    df['Track_Flow_Type'] = df['Lateral_1_5'] / (df['Traction_1_5'] + df['Braking_1_5'] + 1)

    df['Grip_Aero_Balance'] = df['Asphalt_Grip_1_5'] / (df['Downforce_1_5'] + 1)
    df['Total_Min_Pressure'] = df['Min_Pressure_Front_PSI'] + df['Min_Pressure_Rear_PSI']
    df['Pressure_Delta'] = df['Min_Pressure_Front_PSI'] - df['Min_Pressure_Rear_PSI']
    return df

def calculate_dirty_air(target_driver, all_telemetry):
    """
    Calculate the average distance to the closest car ahead for each telemetry point of the target driver
    """
    target_tel = all_telemetry[target_driver]
    ref_time = target_tel['SessionTime'].dt.total_seconds().values
    ref_dist = target_tel['Distance'].values 

    rival_distances = pd.DataFrame(index=ref_time)
    
    for drv, d_tel in all_telemetry.items():
        if drv == target_driver: 
            continue
        try:
            d_time = d_tel['SessionTime'].dt.total_seconds().values
            d_dist = d_tel['Distance'].values

            _, unique_indices = np.unique(d_time, return_index=True)
            d_time = d_time[unique_indices]
            d_dist = d_dist[unique_indices]
            
            f_dist = interp1d(d_time, d_dist, kind='linear', bounds_error=False, fill_value=np.nan)
            rival_distances[drv] = f_dist(ref_time)
        except Exception:
            continue

    current_dist_series = pd.Series(ref_dist, index=ref_time)
    diff_matrix = rival_distances.sub(current_dist_series, axis=0)
    
    mask_ahead = (diff_matrix > 0) & (diff_matrix < 250)
    closest_dist_m = diff_matrix.where(mask_ahead).min(axis=1)
    
    my_speed_ms = target_tel['Speed'] / 3.6
    my_speed_ms = my_speed_ms.replace(0, 10.0)
    
    time_gaps = closest_dist_m.values / my_speed_ms.values
    time_gaps = pd.Series(time_gaps).fillna(5.0)
    time_gaps[time_gaps > 5.0] = 5.0
    
    return time_gaps.values

def calculate_energy(telemetry):
    """
    Calculates tyre energy for a single telemetry dataframe
    """
    dt = telemetry['Time'].dt.total_seconds().diff().fillna(0.05)
    dt = np.clip(dt, 0.0, 0.1) 
    v = telemetry['Speed'] / 3.6 
    
    a_long = v.diff() / dt
    a_long = a_long.fillna(0)
    
    try:
        x = telemetry['X'].values
        y = telemetry['Y'].values

        wl = min(15, len(x) if len(x) % 2 != 0 else len(x) - 1)
        if wl > 3:
            x_smooth = savgol_filter(x, window_length=wl, polyorder=3)
            y_smooth = savgol_filter(y, window_length=wl, polyorder=3)
        else:
            x_smooth, y_smooth = x, y

        dx = np.gradient(x_smooth)
        dy = np.gradient(y_smooth)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        curvature = (dx * ddy - dy * ddx) / (np.power(dx**2 + dy**2, 1.5) + 1e-6)
        a_lat = (v**2) * np.abs(curvature)
        a_lat = np.clip(a_lat, 0, 60.0) 
    except:
        a_lat = np.zeros(len(telemetry))

    combined_accel = np.sqrt(a_long**2 + a_lat**2)
    instantaneous_energy = combined_accel * v * dt
    
    return instantaneous_energy / 1000.0 

def build_accumulated_wear(df_laps):
    df = df_laps.copy()
    df = df.sort_values(by=['Driver', 'LapNumber']).reset_index(drop=True)
    
    df['P_surface'] = df['Wear_Severity_Index'] * df['Compound_Int']
    df['M_aero'] = np.where(df['Gap_To_Car_Ahead'] < 2.0, 1.15, 1.0)
    
    df['E_lap'] = df.groupby(['Driver', 'Stint'])['E_lap'].transform(lambda x: x.fillna(x.median()))
    df['E_lap'] = df['E_lap'].fillna(0)
    
    df['Lap_Damage'] = df['E_lap'] * df['P_surface'] * df['M_aero']
    df['Accumulated_Tyre_Wear'] = df.groupby(['Driver', 'Stint'])['Lap_Damage'].cumsum()
    
    return df

def calculate_lateral_offset(tgt_tel, ref_nd, ix, iy, window):
    """
    Calculate the lateral offset of the target telemetry from the reference racing line using vectorized operations
    """
    tgt_tel = tgt_tel.copy()
    
    # smooth the target lap coordinates
    tgt_tel['X'] = savgol_filter(tgt_tel['X'].values, window, polyorder=3)
    tgt_tel['Y'] = savgol_filter(tgt_tel['Y'].values, window, polyorder=3)
    
    # calculation of the reference point on the line for each target point
    nd = np.clip(tgt_tel['NormDist'].values, ref_nd.min(), ref_nd.max())
    rx, ry = ix(nd), iy(nd)
    vx, vy = tgt_tel['X'].values - rx, tgt_tel['Y'].values - ry
    
    eps = 0.002
    nd_lo = np.clip(nd - eps, ref_nd.min(), ref_nd.max())
    nd_hi = np.clip(nd + eps, ref_nd.min(), ref_nd.max())
    dx = ix(nd_hi) - ix(nd_lo)
    dy = iy(nd_hi) - iy(nd_lo)
    
    length = np.sqrt(dx**2 + dy**2)
    length = np.where(length < 1e-9, 1.0, length)
    tx, ty = dx / length, dy / length
    nx, ny = ty, -tx
    
    tgt_tel['LateralOffset_m'] = np.abs((vx * nx + vy * ny) / 10.0)
    return tgt_tel

def get_reference_lap(session):
    """
    Get the reference lap telemetry and prepare interpolation functions for the racing line
    """
    q_laps = session.laps.pick_quicklaps().pick_wo_box()
    ref_lap = q_laps.pick_fastest()
    ref_tel = ref_lap.get_telemetry().add_distance().dropna(subset=['X', 'Y', 'Distance'])
    ref_tel['NormDist'] = ref_tel['Distance'] / ref_tel['Distance'].max()
    ref_tel = ref_tel.sort_values('NormDist').drop_duplicates('NormDist').reset_index(drop=True)

    window = min(15, len(ref_tel) if len(ref_tel) % 2 == 1 else len(ref_tel) - 1)
    ref_x_smooth = savgol_filter(ref_tel['X'].values, window, polyorder=3)
    ref_y_smooth = savgol_filter(ref_tel['Y'].values, window, polyorder=3)
    ref_nd = ref_tel['NormDist'].values

    ix = interp1d(ref_nd, ref_x_smooth, kind='linear', fill_value="extrapolate")
    iy = interp1d(ref_nd, ref_y_smooth, kind='linear', fill_value="extrapolate")

    return ref_nd, ix, iy, window


def generate_telemetry_features_dataset(session, df_laps):
    """
    Main pipeline function to generate telemetry features dataset
    """
    # filter out wet compounds
    df_laps = df_laps[df_laps['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])].reset_index(drop=True)
    df_laps = add_derived_features(df_laps)
    
    df_laps['Compound_Int'] = np.where(df_laps['Compound'] == 'SOFT', df_laps['Compound_Soft_Int'],
                            np.where(df_laps['Compound'] == 'MEDIUM', df_laps['Compound_Medium_Int'],
                            df_laps['Compound_Hard_Int']))

    df_laps['E_lap'] = np.nan
    df_laps['Gap_To_Car_Ahead'] = 5.0
    df_laps['LatOffset_Mean'] = np.nan
    df_laps['LatOffset_Std'] = np.nan
    
    active_drivers = df_laps['Driver'].unique()

    ref_nd, ix, iy, window = get_reference_lap(session)

    # get telemetry
    all_telemetry = {}
    for drv in active_drivers:
        try:
            d_laps = session.laps.pick_drivers(drv)
            if len(d_laps) > 0:
                tel  = d_laps.get_telemetry()
                if len(tel) >= 10:
                    if 'Distance' not in tel.columns:
                        tel.add_distance()
                    all_telemetry[drv] = tel
        except Exception:
            pass

    # apply dirty air and tyre energy calculations to telemetry
    for drv, tel in all_telemetry.items():
        tel['Gap_To_Car_Ahead'] = calculate_dirty_air(drv, all_telemetry)
        tel['Energy_Tick'] = calculate_energy(tel)
        all_telemetry[drv] = tel

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
                df_laps.loc[idx, 'Gap_To_Car_Ahead'] = lap_tel['Gap_To_Car_Ahead'].mean()

                # start lateral offset calculation
                lap_dist = lap_tel['Distance'] - lap_tel['Distance'].min()
                lap_tel['NormDist'] = lap_dist / lap_dist.max()				
                lap_tel = lap_tel.sort_values('NormDist').drop_duplicates('NormDist').reset_index(drop=True)

                lap_offset_tel = calculate_lateral_offset(lap_tel, ref_nd, ix, iy, window)
                
                df_laps.loc[idx, 'LatOffset_Mean'] = lap_offset_tel['LateralOffset_m'].mean()
                df_laps.loc[idx, 'LatOffset_Std'] = lap_offset_tel['LateralOffset_m'].std()
                
        except Exception as e:
            print(f"Error processing telemetry for driver {driver} lap {row['LapNumber']}: {e}")
            continue
    
    # run clustering
    clustering_laps = df_laps[
        (df_laps['TrackStatus'] == '1') &
        (df_laps['PitInTime'].isna()) &
        (df_laps['PitOutTime'].isna()) &
        (df_laps.get('FastF1Generated', False) == False)
    ].reset_index(drop=True)

    try:
        df_clustered = run_clustering_features(session, clustering_laps, all_telemetry=all_telemetry)
        if not df_clustered.empty:
            df_laps = df_laps.merge(df_clustered, on=['Driver', 'LapNumber'], how='left')
    except Exception as e:
        print(f"Clustering features failed: {e}")

    return df_laps


def run_telemetry_feature_generation(session, df):
    start = time.time()
    df_race_wear = generate_telemetry_features_dataset(session, df)

    end = time.time()
    print(f"\nTelemetry features time taken: {end - start:.4f} seconds\n")

    return df_race_wear

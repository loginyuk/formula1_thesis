from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import fastf1
import os
import time


def add_curvature_to_telemetry(tel):
    """
    Calculates a curvature using smoothed GPS coordinates
    This is instead or missing Steering Angle
    """
    tel = tel.drop_duplicates(subset=['Distance']).copy()
    tel = tel[tel['Distance'].diff().fillna(1) > 0].copy()
    tel = tel.reset_index(drop=True)

    x = tel['X'].ffill().bfill().values
    y = tel['Y'].ffill().bfill().values

    # reduce noise
    x_smooth = savgol_filter(x, window_length=15, polyorder=3)
    y_smooth = savgol_filter(y, window_length=15, polyorder=3)

    # calculate derivatives
    dist = tel['Distance'].values
    dx, dy = np.gradient(x_smooth, dist), np.gradient(y_smooth, dist)
    ddx, ddy = np.gradient(dx, dist), np.gradient(dy, dist)

    numerator = (dx * ddy) - (dy * ddx)
    denominator = np.power(dx**2 + dy**2, 1.5) + 1e-6
    tel['Curvature'] = np.abs(numerator / denominator)

    tel.loc[tel['Speed'] < 30, 'Curvature'] = 0.0
    return tel



def calc_erraticness(signal):
    s = pd.Series(signal)
    ma = s.rolling(window=5, min_periods=1).mean()
    return np.mean(np.abs(s - ma))

def extract_all_corner_features(corner_tel):
    if len(corner_tel) < 5:
        return None

    dist = corner_tel['Distance'].values
    curvature = corner_tel['Curvature'].values
    brake = corner_tel['Brake'].astype(float).values
    speed = corner_tel['Speed'].values
    throttle = np.clip(corner_tel['Throttle'].astype(float).values, 0, 100)

    curv_grad = np.gradient(curvature, dist)
    brake_grad = np.gradient(brake, dist)
    speed_grad = np.gradient(speed, dist)
    throttle_grad = np.gradient(throttle, dist)

    apex_idx = np.argmax(curvature)
    apex_dist = dist[apex_idx]

    # entry
    brake_aggression = np.abs(np.min(speed_grad[brake == 1])) if np.any(brake == 1) else 0.0

    if np.any(brake == 1):
        brake_start_dist = dist[brake == 1][0]
        dist_to_apex = (apex_dist - brake_start_dist) if (brake_start_dist < apex_dist) else 0.0
    else:
        dist_to_apex = 0.0

    brake_release_slope = 0.0
    if np.any(brake == 1):
        brake_on = np.where(brake == 1)[0]
        brake_release_region = brake_grad[brake_on]
        releasing = brake_release_region[brake_release_region < 0]
        brake_release_slope = np.mean(np.abs(releasing)) if len(releasing) > 0 else 0.0

    # apex
    is_trail_braking = (brake == 1) & (curvature > 0.005)
    trail_braking_pct = np.sum(is_trail_braking) / len(corner_tel)

    max_entry_speed = np.max(speed[:max(1, apex_idx)])
    min_speed = np.min(speed)
    speed_drop_pct = (max_entry_speed - min_speed) / max_entry_speed if max_entry_speed > 0 else 0.0

    apex_speed = speed[apex_idx]
    apex_speed_ratio = apex_speed / max_entry_speed if max_entry_speed > 0 else 1.0

    # exit
    throttle_aggression = np.max(throttle_grad[throttle > 0]) if np.any(throttle > 0) else 0.0

    post_apex_mask = dist > apex_dist
    throttle_on_mask = (throttle > 20) & post_apex_mask
    if np.any(throttle_on_mask):
        throttle_on_dist = dist[throttle_on_mask][0] - apex_dist
    else:
        throttle_on_dist = dist[-1] - apex_dist

    full_throttle_pct = np.sum(throttle >= 99) / len(corner_tel)

    # smoothness
    steer_speed = np.mean(np.abs(curv_grad))
    brake_speed = np.mean(np.abs(brake_grad))
    throttle_speed = np.mean(np.abs(throttle_grad))

    curvature_erraticness = calc_erraticness(curvature) * 1000
    throttle_erraticness = calc_erraticness(throttle)
    brake_erraticness = calc_erraticness(brake)

    longitudinal_jerk = np.mean(np.abs(np.gradient(speed_grad, dist)))

    # strategy
    is_coasting = (throttle < 5) & (brake == 0)
    rolling_phase_pct = np.sum(is_coasting) / len(corner_tel)

    brake_throttle_overlap = np.sum((brake == 1) & (throttle > 10)) / len(corner_tel)

    return {
        # entry
        'Brake_Aggression':          brake_aggression,
        'Braking_Distance_To_Apex':  dist_to_apex,
        'Brake_Release_Slope':       brake_release_slope,
        # apex
        'Trail_Braking_Pct':         trail_braking_pct,
        'Speed_Drop_Pct':            speed_drop_pct,
        'Apex_Speed_Ratio':          apex_speed_ratio,
        'Min_Speed':                 min_speed,
        # exit
        'Throttle_Aggression':       throttle_aggression,
        'Throttle_On_Dist':          throttle_on_dist,
        'Full_Throttle_Pct':         full_throttle_pct,
        # smoothness
        'Steering_Speed':            steer_speed,
        'Brake_Speed':               brake_speed,
        'Throttle_Speed':            throttle_speed,
        'Curvature_Erraticness':     curvature_erraticness,
        'Throttle_Erraticness':      throttle_erraticness,
        'Brake_Erraticness':         brake_erraticness,
        'Longitudinal_Jerk':         longitudinal_jerk,
        # strategy
        'Rolling_Phase_Pct':         rolling_phase_pct,
        'Brake_Throttle_Overlap':    brake_throttle_overlap,
    }



import pandas as pd

def build_mjrt_corner_database(session, laps):
    """
    Builds a corner-level database of driving style metrics.
    Optimized to load continuous telemetry per driver
    """
    corners_info = session.get_circuit_info().corners

    # corner zones: +-100 m around each apex
    corner_zones = {}
    for _, corner in corners_info.iterrows():
        corner_id = f"{corner['Number']}{corner['Letter']}"
        corner_zones[corner_id] = (corner['Distance'] - 100, corner['Distance'] + 100)

    all_corners_data = []

    drivers = laps['Driver'].unique()

    for drv in drivers:
        drv_laps = laps[laps['Driver'] == drv]

        if drv_laps.empty:
            continue

        try:
            # load full race telemetry once per driver
            full_tel = session.laps.pick_drivers(drv).get_telemetry()

            if 'Distance' not in full_tel.columns:
                full_tel.add_distance()

            # compute curvature for the whole race at once
            full_tel = add_curvature_to_telemetry(full_tel)

        except Exception as e:
            print(f"Failed to load full telemetry for {drv}: {e}")
            continue

        for _, lap in drv_laps.iterrows():
            try:
                lap_start_time = lap['LapStartTime']
                lap_end_time = lap['Time']

                mask = (full_tel['SessionTime'] >= lap_start_time) & (full_tel['SessionTime'] <= lap_end_time)
                lap_tel = full_tel.loc[mask].copy()

                if lap_tel.empty:
                    continue

                # reset distance to 0 for this lap so it aligns with corner_zones
                lap_dist = lap_tel['Distance'] - lap_tel['Distance'].min()
                lap_tel['Distance'] = lap_dist

                for corner_id, (start_m, end_m) in corner_zones.items():
                    
                    corner_tel = lap_tel[(lap_tel['Distance'] >= start_m) & (lap_tel['Distance'] <= end_m)]
                    metrics = extract_all_corner_features(corner_tel)
                    
                    if metrics:
                        metrics['Driver'] = drv
                        metrics['LapNumber'] = lap['LapNumber']
                        metrics['LapTime_Sec'] = lap['LapTime'].total_seconds()
                        metrics['Corner_ID'] = corner_id
                        all_corners_data.append(metrics)
                        
            except Exception as e:
                print(f"Skipping {drv} Lap {lap['LapNumber']} due to error: {e}") 
                continue
            
    return pd.DataFrame(all_corners_data)



def plot_style_clusters_xy(df_features, x_col, y_col, driver_code, label_col='Style_Cluster_ID'):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_features[df_features['Driver'] == driver_code], 
        x=x_col, 
        y=y_col, 
        hue=label_col, 
        palette='viridis', 
        s=100, 
        edgecolor='black'
    )
    plt.title(f"Driver {driver_code} Style Clusters", fontsize=14)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title=label_col, loc='upper right')
    os.makedirs("results/clustering/v1", exist_ok=True)
    plt.savefig(f"results/clustering/v1/{driver_code}_style_clusters_opt.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_race_pace_timeline(df_laps, driver_code="VER"):
    plt.figure(figsize=(14, 6))

    driver_data = df_laps[df_laps['Driver'] == driver_code]

    plt.plot(driver_data['LapNumber'], driver_data['LapTime_Sec'], 
                color='gray', linestyle='-', alpha=0.4, zorder=1)

    sns.scatterplot(
        data=driver_data,
        x='LapNumber',
        y='LapTime_Sec',
        hue='Style_Cluster_ID',
        palette='viridis',
        s=130,
        edgecolor='black',
        linewidth=1,
        zorder=2
    )

    cluster_amounts = len(driver_data['Style_Cluster_ID'].value_counts())

    plt.title(f"{driver_code} - Race Pace Timeline (Standardized Clusters)", fontsize=16)
    plt.xlabel("Lap Number", fontsize=12)
    plt.ylabel("Lap Time (Seconds)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    handles, _ = plt.gca().get_legend_handles_labels()
    if cluster_amounts == 3:
        plt.legend(handles, ['0: Push/Attack', '1: Balanced/Race Pace', '2: Save/Traffic'], 
                title='Driving Style', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend(title='Driving Style', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    os.makedirs("results/clustering/v1", exist_ok=True)
    plt.savefig(f"results/clustering/v1/{driver_code}_race_pace_timeline_opt.png", dpi=300, bbox_inches='tight')
    plt.close()



def normalize_dict(dictionary):
    min_val = min(dictionary.values())
    max_val = max(dictionary.values())
    if max_val == min_val:
        return {k: 0.5 for k, v in dictionary.items()}
    return {k: (v - min_val) / (max_val - min_val) for k, v in dictionary.items()}

def calculate_corner_weights(lap_telemetry, corners_dict, alpha=0.5, beta=0.5):
    """
    Calculates corner weights traction and thermal load
    """
    w_time_raw = {}
    w_energy_raw = {}

    for corner_id, bounds in corners_dict.items():
        start_dist = bounds['start']
        end_dist = bounds['end']
        
        corner_tel = lap_telemetry[(lap_telemetry['Distance'] >= start_dist) 
                                        & (lap_telemetry['Distance'] <= end_dist)]
        if len(corner_tel) == 0:
            continue

        # thermal weight
        speed_ms = corner_tel['Speed'].values / 3.6  
        curvature = corner_tel['Curvature'].values

        lateral_g = np.abs(curvature) * (speed_ms ** 2)  
        energy_proxy = np.sum(lateral_g * speed_ms)
        w_energy_raw[corner_id] = energy_proxy


        # traction weight
        post_corner_tel = lap_telemetry[lap_telemetry['Distance'] > end_dist]
        brake_zones = post_corner_tel[post_corner_tel['Brake'] == 1]
        if not brake_zones.empty:
            next_brake_dist = brake_zones.iloc[0]['Distance']
        else:
            next_brake_dist = lap_telemetry['Distance'].max()
            
        straight_tel = post_corner_tel[post_corner_tel['Distance'] < next_brake_dist]

        if straight_tel.empty:
            ft_distance = 0.0
        else:
            dist_deltas = np.diff(straight_tel['Distance'].values, prepend=straight_tel['Distance'].iloc[0])
            is_full_throttle = straight_tel['Throttle'].astype(float).values >= 99
            ft_distance = np.sum(dist_deltas[is_full_throttle])
            
        w_time_raw[corner_id] = ft_distance
        

    w_time_norm = normalize_dict(w_time_raw)
    w_energy_norm = normalize_dict(w_energy_raw)

    final_weights = {}
    for corner_id in corners_dict.keys():
        if corner_id in w_time_norm and corner_id in w_energy_norm:
            final_weights[corner_id] = (alpha * w_time_norm[corner_id]) + (beta * w_energy_norm[corner_id])

    return final_weights


def calculate_circuit_weights(session):
    lap = session.laps.pick_quicklaps().pick_fastest()
    lap_tel = lap.get_telemetry()
    corners_info = session.get_circuit_info().corners

    corners_dict = {}
    for _, corner in corners_info.iterrows():
        corner_id = f"{corner['Number']}{corner['Letter']}"
        corner_start = corner['Distance'] - 100
        corner_end = corner['Distance'] + 100
        corners_dict[corner_id] = {'start': corner_start, 'end': corner_end}

    lap_tel = add_curvature_to_telemetry(lap_tel)
    corner_weights = calculate_corner_weights(lap_tel, corners_dict, alpha=0.5, beta=0.5)
    return corner_weights

def normalize_by_corner(df, feature_cols):
    df_norm = df.copy()
    z_cols = []
    
    for col in feature_cols:
        z_col = f'Z_{col}'
        z_cols.append(z_col)
        df_norm[z_col] = df_norm.groupby('Corner_ID')[col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
    df_norm[z_cols] = df_norm[z_cols].fillna(0)
    return df_norm, z_cols


def apply_weighted_vote(lap_df, weights_dict):
    unique_classes = sorted(lap_df['Corner_Style_ID'].unique())
    class_scores = {c: 0.0 for c in unique_classes}
    
    for _, row in lap_df.iterrows():
        corner_id = str(row['Corner_ID']) 
        predicted_class = row['Corner_Style_ID'] 
        
        weight = weights_dict.get(corner_id)
        if weight is not None:
            class_scores[predicted_class] += weight
        else:
            print(f"Warning: No weight found for corner {corner_id}")
            continue
    
    if sum(class_scores.values()) == 0:
        return lap_df['Corner_Style_ID'].mode()[0]
    
    return max(class_scores, key=class_scores.get)



def cluster_mjrt_driving_style(df_corners, needed_features, corner_weights, group_column, n_clusters=3):
    X_scaled = df_corners[needed_features]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_corners['Raw_Corner_Style_ID'] = kmeans.fit_predict(X_scaled) 

    # Universal Standardizer 
    cluster_coasting = df_corners.groupby('Raw_Corner_Style_ID')[group_column].mean()
    sorted_clusters = cluster_coasting.sort_values().index.tolist()
    label_map = {old_id: new_id for new_id, old_id in enumerate(sorted_clusters)}
    df_corners['Corner_Style_ID'] = df_corners['Raw_Corner_Style_ID'].map(label_map)

    # Weighted Majority Vote
    lap_votes = df_corners.groupby(['Driver', 'LapNumber']).apply(
        lambda lap_df: apply_weighted_vote(lap_df, corner_weights),
        include_groups=False
    ).reset_index(name='Style_Cluster_ID')

    lap_times = df_corners[['Driver', 'LapNumber', 'LapTime_Sec']].drop_duplicates()
    df_laps = pd.merge(lap_votes, lap_times, on=['Driver', 'LapNumber'])

    return df_corners, df_laps

if __name__ == "__main__":
    start_time = time.time()
    os.makedirs("cache", exist_ok=True)
    fastf1.Cache.enable_cache('cache')

    session_2023_silv = fastf1.get_session(2023, 'Silverstone', 'R')
    session_2023_silv.load(telemetry=True, weather=False, messages=False)
    all_laps = session_2023_silv.laps.pick_quicklaps().reset_index(drop=True)
    
    df_corners = build_mjrt_corner_database(session_2023_silv, all_laps)

    all_extracted_features = ['Brake_Aggression', 'Braking_Distance_To_Apex', 'Brake_Release_Slope',
                            'Trail_Braking_Pct', 'Speed_Drop_Pct', 'Apex_Speed_Ratio',
                            'Throttle_Aggression', 'Throttle_On_Dist', 'Full_Throttle_Pct',
                            'Steering_Speed', 'Brake_Speed', 'Throttle_Speed',
                            'Longitudinal_Jerk', 'Rolling_Phase_Pct', 'Brake_Throttle_Overlap']

    df_corners_norm, z_features = normalize_by_corner(df_corners, all_extracted_features)

    corner_weights = calculate_circuit_weights(session_2023_silv)
    features = ['Z_Throttle_On_Dist', 'Z_Throttle_Speed', 'Z_Apex_Speed_Ratio']

    df_corners_norm, df_laps = cluster_mjrt_driving_style(df_corners_norm, features, corner_weights, 
                            group_column='Z_Throttle_On_Dist', n_clusters=3)

    plot_style_clusters_xy(df_corners_norm, 'Z_Throttle_Speed', 'Z_Throttle_On_Dist', 'VER', label_col='Corner_Style_ID')
    plot_race_pace_timeline(df_laps, 'VER')

    print(f"Clustering completed in {time.time() - start_time:.2f} seconds")
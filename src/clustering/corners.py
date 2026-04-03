import logging
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

logger = logging.getLogger('clustering')


def add_curvature_to_telemetry(tel):
    """
    Calculates curvature using smoothed GPS coordinates.
    Replaces missing steering angle data
    """
    tel = tel.drop_duplicates(subset=['Distance']).copy()
    tel = tel[tel['Distance'].diff().fillna(1) > 0].copy()
    tel = tel.reset_index(drop=True)

    x = tel['X'].ffill().bfill().values
    y = tel['Y'].ffill().bfill().values

    if len(x) < 5:
        tel['Curvature'] = 0.0
        return tel

    # window length should be odd and less than data length
    wl = min(15, len(x))
    if wl % 2 == 0:
        wl -= 1
    wl = max(wl, 5)

    x_smooth = savgol_filter(x, window_length=wl, polyorder=min(3, wl - 1))
    y_smooth = savgol_filter(y, window_length=wl, polyorder=min(3, wl - 1))

    dist = tel['Distance'].values
    dx, dy = np.gradient(x_smooth, dist), np.gradient(y_smooth, dist)
    ddx, ddy = np.gradient(dx, dist), np.gradient(dy, dist)

    with np.errstate(over='ignore', invalid='ignore'):
        numerator = (dx * ddy) - (dy * ddx)
        denominator = np.power(dx**2 + dy**2, 1.5) + 1e-6
        curvature = np.abs(numerator / denominator)

    curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
    tel['Curvature'] = curvature
    tel.loc[tel['Speed'] < 30, 'Curvature'] = 0.0
    return tel


def extract_corner_features(corner_tel):
    """
    Extracts 6 driving-style features from a single corner
    """
    if len(corner_tel) < 5:
        return None

    dist = corner_tel['Distance'].values
    curvature = corner_tel['Curvature'].values
    brake = corner_tel['Brake'].astype(float).values   # boolean (0, 1)
    speed = corner_tel['Speed'].values
    throttle = np.clip(corner_tel['Throttle'].astype(float).values, 0, 100)

    corner_len = max(dist[-1] - dist[0], 1.0)

    apex_idx = np.argmax(curvature)
    apex_dist = dist[apex_idx]
    apex_speed = speed[apex_idx]

    # entry
    pre_apex = speed[:apex_idx] if apex_idx > 0 else speed
    entry_speed = np.max(pre_apex)
    if entry_speed < 5:
        return None

    apex_speed_ratio = apex_speed / entry_speed

    # braking behavior
    brake_fraction = np.sum(brake) / len(brake)

    if np.any(brake == 1):
        brake_start = dist[np.argmax(brake == 1)]
        brake_point_norm = (brake_start - dist[0]) / corner_len
    else:
        brake_point_norm = np.nan

    # exit
    post_apex = dist > apex_dist
    remaining_len = max(dist[-1] - apex_dist, 1.0)
    t_post = throttle[post_apex]
    d_post = dist[post_apex]

    on_mask = t_post > 20
    if np.any(on_mask):
        throttle_on_dist_norm = (d_post[on_mask][0] - apex_dist) / remaining_len
    else:
        throttle_on_dist_norm = 1.0

    if len(t_post) > 1:
        throttle_integral_norm = np.trapz(t_post, d_post) / (100.0 * remaining_len)
    else:
        throttle_integral_norm = t_post[0] / 100.0 if len(t_post) else 0.0

    # smoothness
    speed_cv = np.std(speed) / (entry_speed + 1e-6)

    return {
        'Apex_Speed_Ratio':       apex_speed_ratio,
        'Brake_Fraction':         brake_fraction,
        'Brake_Point_Norm':       brake_point_norm,
        'Throttle_On_Dist_Norm':  throttle_on_dist_norm,
        'Throttle_Integral_Norm': throttle_integral_norm,
        'Speed_CV':               speed_cv,
    }


def build_corner_zones(session=None, circuit_info=None):
    """
    Builds corner zones dict from session circuit info.
    """
    if circuit_info is None:
        circuit_info = session.get_circuit_info()
    corners_info = circuit_info.corners
    return {
        f"{c['Number']}{c['Letter']}": (c['Distance'] - 100, c['Distance'] + 100)
        for _, c in corners_info.iterrows()
    }


def build_corner_database(session, laps, all_telemetry=None, corner_zones=None):
    """
    Builds a corner-level database of driving style metrics.
    If all_telemetry dict is provided, uses it directly.
    Otherwise loads telemetry per driver from the session.
    """
    if corner_zones is None:
        corner_zones = build_corner_zones(session)

    all_corners_data = []
    for drv in laps['Driver'].unique():
        drv_laps = laps[laps['Driver'] == drv]

        try:
            if all_telemetry is not None and drv in all_telemetry:
                full_tel = all_telemetry[drv].copy()
                if 'Distance' not in full_tel.columns:
                    full_tel = full_tel.copy()
                    full_tel['Distance'] = np.nan
            else:
                full_tel = session.laps.pick_drivers(drv).get_telemetry()
                if 'Distance' not in full_tel.columns:
                    full_tel.add_distance()
            full_tel = add_curvature_to_telemetry(full_tel)
        except Exception as e:
            logger.warning(f"Telemetry load failed for {drv}: {e}")
            continue

        for _, lap in drv_laps.iterrows():
            try:
                mask = (full_tel['SessionTime'] >= lap['LapStartTime']) & (full_tel['SessionTime'] <= lap['Time'])
                lap_tel = full_tel.loc[mask].copy()
                if lap_tel.empty:
                    continue

                lap_tel['Distance'] = lap_tel['Distance'] - lap_tel['Distance'].min()

                for corner_id, (start_m, end_m) in corner_zones.items():
                    corner_tel = lap_tel[(lap_tel['Distance'] >= start_m) &
                                    (lap_tel['Distance'] <= end_m)].copy()
                    metrics = extract_corner_features(corner_tel)
                    if metrics is None:
                        continue

                    metrics['Driver'] = drv
                    metrics['LapNumber'] = lap['LapNumber']
                    metrics['LapTime_Sec'] = lap['LapTime'].total_seconds()
                    metrics['Corner_ID'] = corner_id
                    all_corners_data.append(metrics)

            except Exception as e:
                logger.warning(f"Corner extraction error on lap {lap['LapNumber']} for {drv}: {e}")
    return pd.DataFrame(all_corners_data)

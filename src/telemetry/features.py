import logging
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

logger = logging.getLogger('telemetry')


def add_derived_features(df):
    """
    Add derived features from Pirelli press data to the laps dataframe
    """
    df['Compound_Hard_Int'] = df['Compound_Hard_Hardness']
    df['Compound_Medium_Int'] = df['Compound_Medium_Hardness']
    df['Compound_Soft_Int'] = df['Compound_Soft_Hardness']

    df['Wear_Severity_Index'] = df['Asphalt_Abrasion_1_5'] * df['Tyre_Stress_1_5']
    df['Track_Flow_Type'] = df['Lateral_1_5'] / (df['Traction_1_5'] + df['Braking_1_5'] + 1)

    df['Grip_Aero_Balance'] = df['Asphalt_Grip_1_5'] / (df['Downforce_1_5'] + 1)
    df['Total_Min_Pressure'] = df['Min_Pressure_Front_PSI'] + df['Min_Pressure_Rear_PSI']
    df['Pressure_Delta'] = df['Min_Pressure_Front_PSI'] - df['Min_Pressure_Rear_PSI']
    return df


def calculate_dirty_air(lap_tel, dirty_air_threshold=2.0):
    """
    Calculate dirty air metrics for a single lap using FastF1's per-lap add_driver_ahead()
    """
    try:
        lap_tel_with_ahead = lap_tel.add_driver_ahead()
        dist_ahead = lap_tel_with_ahead['DistanceToDriverAhead']

        my_speed_ms = lap_tel_with_ahead['Speed'] / 3.6
        my_speed_ms = my_speed_ms.replace(0, 10.0)

        time_gaps = (dist_ahead / my_speed_ms).clip(upper=5.0)
        valid = time_gaps.dropna()

        if valid.empty:
            return 5.0, 0.0

        mean_gap = valid.mean()
        dirty_air_fraction = (valid < dirty_air_threshold).sum() / len(valid)
        return mean_gap, dirty_air_fraction
    except Exception as e:
        logger.warning(f"Dirty air calculation failed: {e}")
        return 5.0, 0.0


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

        x_m = x_smooth / 10.0
        y_m = y_smooth / 10.0

        dist = np.cumsum(np.asarray(v) * np.asarray(dt)) + np.arange(len(x_m)) * 1e-9

        # handle numerical issues in curvature calculation
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            dx = np.gradient(x_m, dist)
            dy = np.gradient(y_m, dist)
            ddx = np.gradient(dx, dist)
            ddy = np.gradient(dy, dist)

            curvature = (dx * ddy - dy * ddx) / (np.power(dx**2 + dy**2, 1.5) + 1e-6)
        curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
        a_lat = (v**2) * np.abs(curvature)
        a_lat = np.clip(a_lat, 0, 60.0)
    except Exception as e:
        logger.warning(f"Lateral acceleration calculation failed: {e}")
        a_lat = np.zeros(len(telemetry))

    combined_accel = np.sqrt(a_long**2 + a_lat**2)
    instantaneous_energy = combined_accel * v * dt

    return instantaneous_energy / 1000.0


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
    if ref_lap is None or (hasattr(ref_lap, 'empty') and ref_lap.empty):
        return None, None, None, None
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

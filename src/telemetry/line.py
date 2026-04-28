import logging
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

logger = logging.getLogger('telemetry')

def get_reference_lap(session):
    """
    Pick the fastest lap of the session as reference and 
    build interpolation functions for the racing line
    """
    # get the fastest lap with valid telemetry
    q_laps = session.laps.pick_quicklaps().pick_wo_box()
    ref_lap = q_laps.pick_fastest()
    if ref_lap is None or (hasattr(ref_lap, 'empty') and ref_lap.empty):
        return None, None, None, None

    ref_tel = ref_lap.get_telemetry().add_distance().dropna(subset=['X', 'Y', 'Distance'])
    ref_tel['NormDist'] = ref_tel['Distance'] / ref_tel['Distance'].max()
    ref_tel = ref_tel.sort_values('NormDist').drop_duplicates('NormDist').reset_index(drop=True)

    # smooth reference line coordinates
    window = min(15, len(ref_tel) if len(ref_tel) % 2 == 1 else len(ref_tel) - 1)
    ref_x_smooth = savgol_filter(ref_tel['X'].values, window, polyorder=3)
    ref_y_smooth = savgol_filter(ref_tel['Y'].values, window, polyorder=3)
    ref_nd = ref_tel['NormDist'].values

    # build interpolation functions for the reference line
    ix = interp1d(ref_nd, ref_x_smooth, kind='linear', fill_value="extrapolate")
    iy = interp1d(ref_nd, ref_y_smooth, kind='linear', fill_value="extrapolate")

    return ref_nd, ix, iy, window


def calculate_lateral_offset(tgt_tel, ref_nd, ix, iy, window):
    """
    Calculate lateral deviation from the reference racing line
    """
    tgt_tel = tgt_tel.copy()

    # smooth target lap coordinates
    tgt_tel['X'] = savgol_filter(tgt_tel['X'].values, window, polyorder=3)
    tgt_tel['Y'] = savgol_filter(tgt_tel['Y'].values, window, polyorder=3)

    # project target telemetry points onto the reference line
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


def build_racing_line_features(df_laps, idx, lap_tel, ref_nd, ix, iy, window):
    """
    Calculate mean and std features of lateral offset
    """
    lap_dist = lap_tel['Distance'] - lap_tel['Distance'].min()
    lap_tel['NormDist'] = lap_dist / lap_dist.max()
    lap_tel = lap_tel.sort_values('NormDist').drop_duplicates('NormDist').reset_index(drop=True)

    lap_offset_tel = calculate_lateral_offset(lap_tel, ref_nd, ix, iy, window)
    df_laps.loc[idx, 'LatOffset_Mean'] = lap_offset_tel['LateralOffset_m'].mean()
    df_laps.loc[idx, 'LatOffset_Std']  = lap_offset_tel['LateralOffset_m'].std()

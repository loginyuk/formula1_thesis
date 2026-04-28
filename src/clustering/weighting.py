import numpy as np

from .corners import add_curvature_to_telemetry, build_corner_zones

def normalize_dict(d):
    """
    Normalize a dict of values to [0, 1] range
    """
    lo, hi = min(d.values()), max(d.values())
    if hi == lo:
        return {k: 0.5 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def calculate_circuit_weights(session, corner_zones=None, all_telemetry=None):
    """
    Calculates corner importance weights based on traction and thermal load
    """
    if corner_zones is None:
        corner_zones = build_corner_zones(session)

    lap = session.laps.pick_quicklaps().pick_fastest()
    
    # get telemetry for the lap, preloaded if exists
    if 'Driver' in lap.index:
        drv_abbr = lap['Driver'] 
    else:
        None

    if all_telemetry and drv_abbr in all_telemetry:
        lap_tel = add_curvature_to_telemetry(all_telemetry[drv_abbr].copy())
    else:
        lap_tel = add_curvature_to_telemetry(lap.get_telemetry())

    w_time, w_energy = {}, {}
    for corner_id, (start_m, end_m) in corner_zones.items():
        c_tel = lap_tel[(lap_tel['Distance'] >= start_m) & (lap_tel['Distance'] <= end_m)]
        if c_tel.empty:
            continue
        
        # calculate energy load as sum of lateral g * speed over the corner
        speed_ms = c_tel['Speed'].values / 3.6
        lateral_g = np.abs(c_tel['Curvature'].values) * speed_ms ** 2
        w_energy[corner_id] = np.sum(lateral_g * speed_ms)

        # calculate time importance as throttle time on the straight after the corner
        after = lap_tel[lap_tel['Distance'] > end_m]
        braking = after[after['Brake'] == 1]
        next_brake = braking.iloc[0]['Distance'] if not braking.empty else lap_tel['Distance'].max()
        straight = after[after['Distance'] < next_brake]

        if straight.empty:
            w_time[corner_id] = 0.0
        else:
            deltas = np.diff(straight['Distance'].values, prepend=straight['Distance'].iloc[0])
            w_time[corner_id] = np.sum(deltas[straight['Throttle'].astype(float).values >= 99])

    # get normalized weights
    w_time = normalize_dict(w_time)
    w_energy = normalize_dict(w_energy)
    return {
        corner_id: 0.5 * w_time[corner_id] + 0.5 * w_energy[corner_id]
        for corner_id in corner_zones if corner_id in w_time and corner_id in w_energy
    }
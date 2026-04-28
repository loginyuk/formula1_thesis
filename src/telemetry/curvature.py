import numpy as np
from scipy.signal import savgol_filter

def compute_curvature(x, y):
    """
    Track curvature from GPS coordinates, in 1/m.
    """
    if len(x) < 5:
        return np.zeros(len(x))

    x_m = np.asarray(x, dtype=float) / 10.0
    y_m = np.asarray(y, dtype=float) / 10.0

    # smooth coordinates to remove gps noise
    wl = min(15, len(x_m))
    if wl % 2 == 0:
        wl -= 1
    x_s = savgol_filter(x_m, wl, polyorder=3)
    y_s = savgol_filter(y_m, wl, polyorder=3)

    # calculate curvature using finite differences
    seg = np.sqrt(np.diff(x_s, prepend=x_s[0]) ** 2 + np.diff(y_s, prepend=y_s[0]) ** 2)
    seg = np.maximum(seg, 1e-6)
    dist = np.cumsum(seg)

    dx, dy = np.gradient(x_s, dist), np.gradient(y_s, dist)
    ddx, ddy = np.gradient(dx, dist), np.gradient(dy, dist)

    denom = (dx ** 2 + dy ** 2) ** 1.5
    return np.divide(np.abs(dx * ddy - dy * ddx), denom, out=np.zeros_like(denom), where=denom > 0)
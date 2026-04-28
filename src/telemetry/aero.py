import logging
import numpy as np

from src.config import DIRTY_AIR_THRESHOLD_SEC

logger = logging.getLogger('telemetry')

def calculate_dirty_air(lap_tel, dirty_air_threshold=DIRTY_AIR_THRESHOLD_SEC):
    """
    Calculate mean gap to car ahead and dirty air fraction for a lap
    """
    try:
        # Add driver ahead info to telemetry
        lap_tel_with_ahead = lap_tel.add_driver_ahead()
        dist_ahead = lap_tel_with_ahead['DistanceToDriverAhead']

        my_speed_ms = lap_tel_with_ahead['Speed'] / 3.6
        my_speed_ms = my_speed_ms.where(my_speed_ms > 1.0, np.nan)

        # calculate time gaps
        time_gaps = (dist_ahead / my_speed_ms).clip(upper=5.0)
        valid = time_gaps.dropna()

        if valid.empty:
            return 5.0, 0.0

        mean_gap = valid.mean()
        dirty_air_fraction = (valid < dirty_air_threshold).sum() / len(valid)
        return mean_gap, dirty_air_fraction
    except Exception as e:
        logger.warning(f"Error calculating dirty air: {e}")
        return 5.0, 0.0


def build_aero_features(df_laps):
    """
    Calculate aero loss
    """
    df = df_laps.copy()
    df['Aero_Loss'] = np.exp(-df['Gap_To_Car_Ahead'] / 2.0)
    return df
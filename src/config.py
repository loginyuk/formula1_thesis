import os

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directory paths
CACHE_DIR          = os.path.join(PROJECT_ROOT, "cache")
DATA_RAW_DIR       = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR        = os.path.join(PROJECT_ROOT, "results")
LOGS_DIR           = os.path.join(PROJECT_ROOT, "logs")
SUMMARIES_DIR      = os.path.join(LOGS_DIR, "summaries")

# Results subdirectories
RESULTS_CLUSTERING_DIR              = os.path.join(RESULTS_DIR, "clustering")
RESULTS_CLUSTERING_VERIFICATION_DIR = os.path.join(RESULTS_CLUSTERING_DIR, "verification")
RESULTS_MODEL_DIR                   = os.path.join(RESULTS_DIR, "model")
RESULTS_CORRELATION_DIR             = os.path.join(RESULTS_DIR, "correlation")
RESULTS_VISUALIZATIONS_DIR          = os.path.join(RESULTS_DIR, "visualizations")

# File paths
ERRORS_LOG            = os.path.join(LOGS_DIR, "errors.log")
TRACK_PARAMETERS_FILE = os.path.join(DATA_RAW_DIR, "track_parameters.csv")
DATASET_ALL           = os.path.join(DATA_PROCESSED_DIR, "dataset_all.csv")
DATASET_2023          = os.path.join(DATA_PROCESSED_DIR, "dataset_2023.csv")

# Clustering constants
N_CLUSTERS = 3

FEATURE_COLS = [
    'Apex_Speed_Ratio',
    'Brake_Fraction',
    'Brake_Point_Norm',
    'Throttle_On_Dist_Norm',
    'Throttle_Integral_Norm',
    'Speed_CV',
]

CLUSTER_FEATURES = [
    'Apex_Speed_Ratio',
    'Throttle_On_Dist_Norm',
    'Throttle_Integral_Norm',
]

CLUSTER_COLOURS = {0: '#e74c3c', 1: '#3498db', 2: '#2ecc71', 3: '#f39c12'}
CLUSTER_NAMES   = {0: 'Exit Attack', 1: 'Speed Carry', 2: 'Throttle Save', 3: 'Cluster 3'}

# Model feature lists
MODEL_FEATURES = [
    'Year', 'LapNumber', 'Stint',
    'TyreLife', 'Tyre_Compound_Interaction',
    'AirTemp', 'Humidity', 'Pressure',
    'TrackTemp', 'WindDirection', 'WindSpeed',
    'FuelLoad', 'Track_Evolution_Physics',
    'Traction_1_5',
    'Asphalt_Grip_1_5', 'Asphalt_Abrasion_1_5', 'Track_Evolution_1_5',
    'Tyre_Stress_1_5', 'Braking_1_5', 'Lateral_1_5', 'Downforce_1_5',
    'Min_Pressure_Front_PSI', 'Min_Pressure_Rear_PSI',
    'Wear_Severity_Index', 'Track_Flow_Type',
    'Circuit_Length_KM', 'Cumulative_Field_Dist_KM',
    'Compound_Int', 'E_lap', 'Gap_To_Car_Ahead', 'Grip_Aero_Balance',
    'Total_Min_Pressure', 'Pressure_Delta', 'LatOffset_Mean',
    'LatOffset_Std', 'Prev_LapTime', 'Lag_2', 'Rolling_Avg_3',
    'Prev_Delta', 'Driver_Encoded', 'Location_Encoded', 'Team_Encoded',
    'Position', 'Dirty_Air_Fraction', 'DRS_Fraction',
    'Tyre_Grip_Index', 'Accumulated_Tyre_Wear',
    'Mean_Apex_Speed_Ratio', 'Std_Apex_Speed_Ratio',
    'Mean_Brake_Fraction', 'Std_Brake_Fraction',
    'Mean_Brake_Point_Norm', 'Std_Brake_Point_Norm',
    'Mean_Throttle_On_Dist_Norm', 'Std_Throttle_On_Dist_Norm',
    'Mean_Throttle_Integral_Norm', 'Std_Throttle_Integral_Norm',
    'Mean_Speed_CV', 'Std_Speed_CV',
    'P_0', 'P_1', 'P_2', 'Style_Cluster_ID', 'Style_Entropy',
    'Aero_Loss', 'Lap_Damage',
]

# Features shifted by 1 lap to remove data leakage (forecasting model)
TELEMETRY_FEATURES_TO_SHIFT = [
    'E_lap', 'Gap_To_Car_Ahead', 'Dirty_Air_Fraction', 'DRS_Fraction',
    'LatOffset_Mean', 'LatOffset_Std', 'Aero_Loss', 'Lap_Damage',
    'Accumulated_Tyre_Wear', 'Tyre_Grip_Index',
    'Mean_Apex_Speed_Ratio', 'Std_Apex_Speed_Ratio',
    'Mean_Brake_Fraction', 'Std_Brake_Fraction',
    'Mean_Brake_Point_Norm', 'Std_Brake_Point_Norm',
    'Mean_Throttle_On_Dist_Norm', 'Std_Throttle_On_Dist_Norm',
    'Mean_Throttle_Integral_Norm', 'Std_Throttle_Integral_Norm',
    'Mean_Speed_CV', 'Std_Speed_CV',
    'P_0', 'P_1', 'P_2', 'Style_Cluster_ID', 'Style_Entropy',
]

# Walk-forward validation constants
MIN_TRAIN_RACES = 20   # minimum races before first prediction
CV_N_SPLITS     = 5    # TimeSeriesSplit folds for hyperparameter tuning
CV_N_JOBS       = 1    # parallelism for CV
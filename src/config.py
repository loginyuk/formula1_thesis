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
RESULTS_CLUSTERING_K_COMPARISON_DIR = os.path.join(RESULTS_CLUSTERING_DIR, "k_comparison")
RESULTS_MODEL_DIR                   = os.path.join(RESULTS_DIR, "model")
RESULTS_CORRELATION_DIR             = os.path.join(RESULTS_DIR, "correlation")

# File paths
ERRORS_LOG            = os.path.join(LOGS_DIR, "errors.log")
TRACK_PARAMETERS_FILE = os.path.join(DATA_RAW_DIR, "track_parameters.csv")
DATASET_ALL           = os.path.join(DATA_PROCESSED_DIR, "dataset_all.csv")

# Clustering constants
N_CLUSTERS = 3

FEATURE_COLS = [
    'Apex_Speed_Ratio',
    'Brake_Fraction',
    'Brake_Point_Norm',
    'Throttle_On_Dist_Norm',
    'Throttle_Integral_Norm',
    'Speed_Variability',
]

CLUSTER_FEATURES = [
    'Apex_Speed_Ratio',
    'Throttle_On_Dist_Norm',
    'Throttle_Integral_Norm',
]

# Shared categorical palette (Flat UI) — used for clusters and any other
# categorical grouping (metrics, compounds, comparison bars, etc.)
PALETTE = [
    '#e74c3c',  # red
    '#3498db',  # blue
    '#2ecc71',  # green
    '#f39c12',  # orange
    '#9b59b6',  # purple
]

# Single-color and accent colors for generic plots
PRIMARY_COLOUR = PALETTE[1]   # blue — default bar/line fill
ACCENT_COLOUR  = PALETTE[0]   # red  — reference lines, "perfect prediction", mean markers

CLUSTER_COLOURS = {i: PALETTE[i] for i in range(len(PALETTE))}
CLUSTER_NAMES   = {0: 'Exit Attack', 1: 'Speed Carry', 2: 'Throttle Save', 3: 'Cluster 3'}

# Model feature lists
MODEL_FEATURES = [
    'LapNumber', 'Stint', 'TyreLife', 'Position', 
    
    'AirTemp', 'Humidity', 'Pressure', 'TrackTemp', 'WindDirection', 'WindSpeed',

    'FuelLoad', 'Track_Evolution', 'Cumulative_Field_Dist_KM',

    'Lap_Damage', 'LatOffset_Mean', 'Accumulated_Tyre_Wear', 'LatOffset_Std', 
    'Energy_Lap', 'Gap_To_Car_Ahead', 'Dirty_Air_Fraction', 'Aero_Loss', 'DRS_Fraction', 
    
    'Traction_1_5', 'Asphalt_Grip_1_5', 'Asphalt_Abrasion_1_5', 'Track_Evolution_1_5',
    'Tyre_Stress_1_5', 'Braking_1_5', 'Lateral_1_5', 'Downforce_1_5',
    'Min_Pressure_Front_PSI', 'Min_Pressure_Rear_PSI', 'Circuit_Length_KM',
    
    'Compound_Int',
    'Wear_Severity_Index', 'Track_Flow_Type', 'Grip_Aero_Balance', 'Total_Min_Pressure', 'Pressure_Delta',
    
    'Prev_LapTime', 'Lag_2', 'Rolling_Avg_3', 'Prev_Delta',

    'Driver_Encoded', 'Location_Encoded', 'Team_Encoded',

    'Mean_Apex_Speed_Ratio', 'Std_Apex_Speed_Ratio',
    'Mean_Brake_Fraction', 'Std_Brake_Fraction',
    'Mean_Brake_Point_Norm', 'Std_Brake_Point_Norm',
    'Mean_Throttle_On_Dist_Norm', 'Std_Throttle_On_Dist_Norm',
    'Mean_Throttle_Integral_Norm', 'Std_Throttle_Integral_Norm',
    'Mean_Speed_Variability', 'Std_Speed_Variability',

    'P_0', 'P_1', 'P_2', 'Style_Cluster_ID', 'Style_Entropy',

    'Year', 'Tyre_Compound_Interaction',
]

# Features shifted by 1 lap to remove data leakage (forecasting model)
TELEMETRY_FEATURES_TO_SHIFT = [
    'Energy_Lap', 'Gap_To_Car_Ahead', 'Dirty_Air_Fraction', 'DRS_Fraction',
    'LatOffset_Mean', 'LatOffset_Std', 'Aero_Loss', 'Lap_Damage',
    'Accumulated_Tyre_Wear',
    'Mean_Apex_Speed_Ratio', 'Std_Apex_Speed_Ratio',
    'Mean_Brake_Fraction', 'Std_Brake_Fraction',
    'Mean_Brake_Point_Norm', 'Std_Brake_Point_Norm',
    'Mean_Throttle_On_Dist_Norm', 'Std_Throttle_On_Dist_Norm',
    'Mean_Throttle_Integral_Norm', 'Std_Throttle_Integral_Norm',
    'Mean_Speed_Variability', 'Std_Speed_Variability',
    'P_0', 'P_1', 'P_2', 'Style_Cluster_ID', 'Style_Entropy',
    'P_0_k2', 'P_1_k2', 'Style_Cluster_ID_k2', 'Style_Entropy_k2',
]

# Cluster features for clustering ablation
CLUSTER_PIPELINE_FEATURES = [
    'Mean_Apex_Speed_Ratio', 'Std_Apex_Speed_Ratio',
    'Mean_Brake_Fraction', 'Std_Brake_Fraction',
    'Mean_Brake_Point_Norm', 'Std_Brake_Point_Norm',
    'Mean_Throttle_On_Dist_Norm', 'Std_Throttle_On_Dist_Norm',
    'Mean_Throttle_Integral_Norm', 'Std_Throttle_Integral_Norm',
    'Mean_Speed_Variability', 'Std_Speed_Variability',
    'P_0', 'P_1', 'P_2', 'Style_Cluster_ID', 'Style_Entropy',
]

MODEL_FEATURES_NO_CLUSTER = [f for f in MODEL_FEATURES if f not in CLUSTER_PIPELINE_FEATURES]

# K=2 variant: replace K=3 cluster columns (P_0, P_1, P_2, Style_Cluster_ID, Style_Entropy)
# with their K=2 counterparts (P_0_k2, P_1_k2, Style_Cluster_ID_k2, Style_Entropy_k2)
_K3_CLUSTER_COLS = {'P_0', 'P_1', 'P_2', 'Style_Cluster_ID', 'Style_Entropy'}
MODEL_FEATURES_K2 = [
    f for f in MODEL_FEATURES if f not in _K3_CLUSTER_COLS
] + ['P_0_k2', 'P_1_k2', 'Style_Cluster_ID_k2', 'Style_Entropy_k2']

# Global reproducibility seed
RANDOM_STATE = 42

# Walk-forward validation constants
MIN_TRAIN_RACES = 5   # minimum races before first prediction
CV_N_SPLITS     = 5    # TimeSeriesSplit folds for hyperparameter tuning
CV_N_JOBS       = 1    # parallelism for CV

# Correlation analysis
HIGH_CORR_THRESHOLD = 0.75  # |r| above this flags a feature pair as highly correlated

# Aerodynamics
DIRTY_AIR_THRESHOLD_SEC = 2.0  # gap (seconds) below which a lap is considered to be in dirty air

# Model choice
# choices: "XGBoost", "LightGBM", "CatBoost", "RandomForest", "Ridge"
PRIMARY_MODEL    = "XGBoost"
BEST_PARAMS_FILE = os.path.join(PROJECT_ROOT, "best_params.json")

MODELS_TO_COMPARE = ["RandomForest", "XGBoost", "LightGBM", "CatBoost", "Ridge"]

MODEL_DEFAULTS = {
    "XGBoost":      dict(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE, verbosity=0),
    "LightGBM":     dict(n_estimators=100, max_depth=5, num_leaves=31, learning_rate=0.1, random_state=RANDOM_STATE, verbose=-1),
    "CatBoost":     dict(iterations=100, depth=5, learning_rate=0.1, random_state=RANDOM_STATE, silent=True),
    "RandomForest": dict(n_estimators=100, max_features='sqrt', max_depth=15, random_state=RANDOM_STATE),
    "Ridge":        dict(),
}

# SHAP analysis: feature category buckets for grouped attribution
FEATURE_BUCKETS = {
    'Lag-1':     ['Prev_LapTime', 'Prev_Delta'],
    'Lag-2':     ['Lag_2'],
    'Rolling-3': ['Rolling_Avg_3'],
    'Cluster':   CLUSTER_PIPELINE_FEATURES,
    'Telemetry': ['Energy_Lap', 'Gap_To_Car_Ahead', 'Dirty_Air_Fraction',
                  'DRS_Fraction', 'LatOffset_Mean', 'LatOffset_Std',
                  'Aero_Loss', 'Lap_Damage', 'Accumulated_Tyre_Wear'],
}
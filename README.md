# Formula 1 Race Pace Prediction Using Machine Learning with Integrated Vehicle Dynamics and Driving Styles

A machine learning pipeline that predicts Formula 1 lap times by combining telemetry-derived vehicle dynamics, physics-based models, and GMM-based driving style clustering. The system uses walk-forward temporal validation across 4 seasons (2022-2025), achieving a best MAE of **0.333 seconds** (R² = 0.9973) on absolute lap times with LightGBM.

Telemetry features are shifted by 1 lap, making this a forecasting model: it uses data available up to lap N-1 to predict lap N time.

## Project Structure

```
formula1_thesis/
├── src/
│   ├── config.py                 # all paths, constants, feature lists, model defaults
│   ├── utils.py                  # shared logging utility
│   ├── logging_setup.py          # logger configuration
│   ├── clustering/               # Driving style clustering
│   │   ├── corners.py            # corner extraction, zone building
│   │   ├── weighting.py          # per-corner time/energy weights
│   │   ├── aggregation.py        # corner -> lap aggregation
│   │   ├── normalization.py      # robust scaling (median + IQR)
│   │   ├── gmm.py                # GMM fitting, label alignment
│   │   ├── pipeline.py           # clustering orchestration
│   │   ├── plots.py              # cluster timelines and verification
│   │   └── k_comparison_plots.py # K-comparison (radar, sizes, PCA)
│   ├── telemetry/                # Telemetry-derived feature engineering
│   │   ├── curvature.py          # GPS curvature with arc-length parameterisation
│   │   ├── tyre.py               # tyre energy, lap damage and accumulated wear
│   │   ├── aero.py               # dirty air, gap, aero loss
│   │   ├── line.py               # reference lap, lateral offset
│   │   └── pipeline.py           # telemetry orchestration
│   ├── data_preparation/         # Static feature construction and cleaning
│   │   ├── loading.py            # FastF1 race loading, weather merge, Pirelli press
│   │   ├── physics.py            # fuel load model, track evolution
│   │   ├── pirelli.py            # Pirelli circuit indices, compound encoding
│   │   ├── cleaning.py           # wet lap removal, lag features, encoding
│   │   └── combine_tracks.py     # per-year Pirelli track parameter merging
│   └── modeling/                 # Model training and analysis
│       ├── training.py           # walk-forward CV, feature shifting, delta conversion
│       ├── analysis_plots.py     # feature importance, model comparison
│       ├── plots.py              # diagnostic plots (per-race MAE, compound, driver)
│       └── shap_plots.py         # SHAP global / by-bucket / over-races plots
│
├── scripts/                          # Runnable entry points
│   ├── run_data_pipeline.py          # build full 2022-2025 dataset
│   ├── run_combine_tracks.py         # merge per-year Pirelli track parameter CSVs
│   ├── run_correlation.py            # correlation matrix
│   ├── run_clustering_single.py      # GMM clustering for a single race
│   ├── run_clustering_k2.py          # K=2 clustering across all races
│   ├── run_cluster_k_comparison.py   # BIC / silhouette / PCA across K = 2..5
│   ├── run_hyperparameter_tuning.py  # Optuna Bayesian tuning (TimeSeriesSplit CV)
│   ├── run_model_training.py         # train PRIMARY_MODEL with walk-forward
│   ├── run_model_comparison.py       # compare all 5 models with diagnostic plots
│   ├── run_model_comparison_k2.py    # model comparison using K=2 clustering features
│   ├── run_model_no_clustering.py    # ablation: with vs without clustering features
│   ├── run_model_no_correlated.py    # ablation: drop one feature from each high-correlated pair
│   └── run_shap_analysis.py          # Tree SHAP analysis (global / by bucket / over time)
│
├── data/
│   ├── raw/                      # Pirelli track parameters (per-year CSVs)
│   └── processed/                # generated dataset_all.csv
├── results/
│   ├── model/                    # per-model results, plots, ablation outputs
│   ├── clustering/               # cluster timelines, verification, K comparison
│   └── correlation/              # correlation matrix, high-correlated pairs
├── logs/                         # error logs and training summaries
├── cache/                        # FastF1 telemetry cache
├── notebooks/                    # Jupyter experiments
├── best_params.json              # Optuna-tuned hyperparameters per model
└── archive/                      # old experimental files
```

## Methodology

### Data Collection and Feature Engineering

Race data is fetched via the FastF1 API for seasons 2022-2025 (84 races, ~57,000 clean laps). The pipeline engineers 65 features across several domains:

#### 1. Data Cleaning

Raw laps are filtered to keep only green-flag, dry conditions:
- Wet laps are removed by detecting the first rainfall timestamp and dropping all subsequent laps (track evolution becomes unpredictable)
- Safety car / VSC laps (`TrackStatus != '1'`) are excluded
- Pit in/out laps, deleted laps, and FastF1-generated (interpolated) laps are removed

#### 2. Physics-Based Features

**Fuel load model** estimates remaining fuel mass per lap. The model starts at 110 kg (with 1 kg expected to remain at the end of the race) and applies a linear burn rate adjusted by track status:
- Green flag: 100% burn rate
- Safety car / VSC: 35% burn rate (reduced engine load)
- Pit in/out laps: 120% burn rate (limiter + acceleration)

**Track evolution** models rubber build-up as the cumulative distance driven by all cars multiplied by the Pirelli `Track_Evolution` rating. This captures the grip improvement over a race as more rubber is laid down.

#### 3. Pirelli-Derived Circuit Features

Pirelli publishes per-circuit ratings (1–5 scale) for: Traction, Asphalt Grip, Asphalt Abrasion, Track Evolution, Tyre Stress, Braking, Lateral, and Downforce. These are merged per year along with minimum tyre pressures and circuit length. Compound hardness integers are also sourced from Pirelli data.

Derived indices:
- `Wear_Severity_Index = Asphalt_Abrasion × Tyre_Stress`
- `Track_Flow_Type = Lateral / (Traction + Braking)` — cornering vs straight-line character
- `Grip_Aero_Balance = Asphalt_Grip / Downforce`
- `Tyre_Compound_Interaction = TyreLife × Compound_Int` — age effect scaled by compound hardness

All of these are static per race weekend (known before the race starts).

#### 4. Tyre Model (`telemetry/tyre.py`)

Per-lap tyre energy is computed from raw telemetry by integrating combined longitudinal and lateral acceleration over the lap:

- Longitudinal acceleration: speed gradient over time
- Lateral acceleration: $a_{lat} = v^2 \cdot |\kappa|$ where curvature $\kappa$ is derived from smoothed GPS coordinates

$$E_{lap} = \frac{1}{1000} \sum \sqrt{a_{long}^2 + a_{lat}^2} \cdot v \cdot dt$$

A physics-inspired degradation model then accumulates wear per stint:

$$\text{Lap Damage} = E_{lap} \times (W_{sev} \times (C_{int} + 1)) \times M_{aero}$$

where $M_{aero} = 1.15$ if gap to car ahead $< 2\,\text{s}$ (dirty air increases tyre sliding), else $1.0$.

`Accumulated_Tyre_Wear` is the cumulative sum of `Lap_Damage` within each stint.

#### 5. Aero Model (`telemetry/aero.py`)

Track position relative to the car ahead is computed via FastF1's `add_driver_ahead()`:

- `Gap_To_Car_Ahead`: mean time gap to the car ahead (seconds)
- `Dirty_Air_Fraction`: fraction of the lap spent within `DIRTY_AIR_THRESHOLD_SEC` (default 2 s) of the car ahead
- `Aero_Loss = exp(-\text{gap} / 2.0)` — exponential downforce penalty from turbulent air
- `DRS_Fraction`: fraction of lap distance with DRS active

#### 6. Racing Line Model (`telemetry/line.py`)

Lateral deviation from the optimal racing line is computed per lap. The reference line is built from the session's fastest lap by interpolating smoothed (X, Y) GPS coordinates against normalised distance. Each target lap's offset is projected onto the normal vector of the reference line, yielding `LatOffset_Mean` and `LatOffset_Std`.

#### 7. Lag Features

Temporal features are computed within micro-stints (consecutive green-flag laps per driver):
- `Prev_LapTime`: previous lap time
- `Lag_2`: 2-lap lag
- `Rolling_Avg_3`: 3-lap rolling average
- `Prev_Delta`: previous lap-to-lap time change
- `Target_Delta = LapTime - Prev_LapTime` (the prediction target)

Laps without history (first 3 laps of each micro-stint) are dropped.

---

### Driving Style Clustering

Each lap is classified into one of 3 driving styles using Gaussian Mixture Models (GMM):

#### 1. Corner Extraction

Corners are identified from the circuit info provided by FastF1. For each corner, a +/-100 m zone around the corner distance is defined. Curvature is computed from smoothed GPS coordinates using Savitzky-Golay filtering, with arc-length parameterisation handled inside `telemetry/curvature.py`.

#### 2. Per-Corner Feature Extraction

For each corner zone on each lap, 6 metrics are extracted:
- **Apex Speed Ratio**: `min_speed / entry_speed` — how much speed the driver carries through the apex
- **Brake Fraction**: fraction of corner distance spent braking
- **Brake Point Norm**: normalised distance of first braking point within the corner
- **Throttle On Dist Norm**: normalised distance from apex to first throttle application (> 20%)
- **Throttle Integral Norm**: integrated throttle area post-apex, normalised by distance and max throttle
- **Speed Variability**: coefficient of variation of speed through the corner (smoothness)

#### 3. Lap Aggregation

Corner features are aggregated to lap level using weighted means and standard deviations. Corner weights combine two factors:
- **Time weight**: full-throttle distance on the subsequent straight (longer straight = more time gained/lost from corner exit speed)
- **Energy weight**: integrated lateral g-force × speed through the corner

Weights are normalised and averaged (50/50) to give each corner an importance score.

#### 4. GMM Fitting and Label Alignment

Lap features are normalised using robust scaling (median + IQR) with 3-sigma clipping. A 3-component full-covariance GMM is fitted (`n_init=10`, seeded by `RANDOM_STATE`).

Clusters are ordered by an aggression score (`-Throttle_On_Dist_Norm + Throttle_Integral_Norm`) so that label IDs are stable across races without needing the Hungarian algorithm.

The 3 identified styles:
- **Cluster 0 — Exit Attack**: early throttle application, aggressive corner exit
- **Cluster 1 — Speed Carry**: high minimum speed through corners, smooth trajectory
- **Cluster 2 — Throttle Save**: save throttle for fuel or tyre management

Per-lap outputs: cluster probabilities (`P_0`, `P_1`, `P_2`), cluster ID, and style entropy (`-sum(p * log(p))`).

A K=2 variant of the same pipeline is generated by `run_clustering_k2.py` and stored under `*_k2` columns for the K=2 vs K=3 ablation study.

---

### Feature Shifting (no data leakage)

To make the model a genuine forecasting tool, 27 telemetry-derived features are shifted by 1 lap within each driver-stint group. This means the model uses lap N-1's telemetry (energy, dirty air, lateral offset, DRS, tyre wear, cluster probabilities, etc.) to predict lap N's time change. Rows with NaN from shifting are dropped.

---

### Prediction Target

The model predicts `Target_Delta` — the lap-to-lap time change ($LapTime_N - LapTime_{N-1}$), not the absolute lap time. This is because:
- it removes circuit-specific baseline effects (each track has different lap times)
- it centres the target around zero, which is easier for tree models to learn

Absolute times are recovered after prediction (`Predicted_Time = Prev_LapTime + Predicted_Delta`).

---

### Walk-Forward Validation

The model uses temporal walk-forward validation to prevent data leakage:

1. Races are sorted chronologically across all seasons
2. The first `MIN_TRAIN_RACES` (default: 5) races form the initial training set
3. For each subsequent race, the model trains on all previous races and predicts the current race
4. The race is added to the training set and the loop continues

This mirrors reality: the model only sees historical data and never learns from future runs.

### Models

Five regression models are compared (`MODELS_TO_COMPARE` in `config.py`):

- **Ridge** — linear baseline (wrapped in Pipeline with `SimpleImputer` + `StandardScaler`)
- **RandomForest** — decision trees (wrapped in Pipeline with `SimpleImputer`)
- **XGBoost** — gradient boosted trees
- **LightGBM** — gradient boosted trees with histogram-based splits
- **CatBoost** — gradient boosted trees with ordered boosting

XGBoost, LightGBM and CatBoost handle NaN values natively and do not require scaling.

---

### Hyperparameter Tuning

Bayesian optimisation via Optuna with `TimeSeriesSplit` (5 folds) as the cross-validation strategy. This ensures temporal ordering is preserved during tuning. Tuned parameters are saved to `best_params.json` at the project root and automatically loaded by training scripts.

---

### Ablation Studies

Four ablations quantify the contribution of different parts of the pipeline:

1. **Clustering ablation** (`run_model_no_clustering.py`) — train all 5 models with the full 65-feature set and again with 48 features (17 clustering-related features removed). Measures the predictive value of GMM clustering.
2. **K=2 vs K=3** (`run_model_comparison_k2.py`) — replaces K=3 cluster columns with their K=2 counterparts and re-runs all models.
3. **No-shift** (`run_model_training.py --no-shift`) — disables the 1-lap telemetry shift to quantify how much accuracy comes from data leakage.
4. **No-correlated features** (`run_model_no_correlated.py`) — drops one feature from each highly correlated pair (`HIGH_CORR_THRESHOLD = 0.75` in `config.py`).

### SHAP Analysis

`run_shap_analysis.py` computes Tree SHAP values on each walk-forward fold for tree-based models (LightGBM by default; switch with `--model XGBoost` or `--model CatBoost`). Three outputs are produced:
- `shap_global` — top 20 features by mean |SHAP|
- `shap_by_lag` — totals grouped by feature category (Lag-1, Lag-2, Rolling-3, Cluster, Telemetry, Static), driven by `FEATURE_BUCKETS` in `config.py`
- `shap_over_races` — mean |SHAP| of the top 8 features across the 78 walk-forward test races

---

## Feature Categories

Total of 65 features:

| Category | Count | Features |
|---|---|---|
| Meta | 1 | Year |
| Race state | 4 | LapNumber, Stint, TyreLife, Position |
| Weather | 6 | AirTemp, Humidity, Pressure, TrackTemp, WindSpeed, WindDirection |
| Encoded | 3 | Driver_Encoded, Location_Encoded, Team_Encoded |
| Pirelli raw | 11 | Traction_1_5, Asphalt_Grip_1_5, Asphalt_Abrasion_1_5, Track_Evolution_1_5, Tyre_Stress_1_5, Braking_1_5, Lateral_1_5, Downforce_1_5, Min_Pressure_Front_PSI, Min_Pressure_Rear_PSI, Circuit_Length_KM |
| Pirelli-derived | 7 | Wear_Severity_Index, Track_Flow_Type, Grip_Aero_Balance, Total_Min_Pressure, Pressure_Delta, Compound_Int, Tyre_Compound_Interaction |
| Lag features | 4 | Prev_LapTime, Lag_2, Rolling_Avg_3, Prev_Delta |
| Static physics | 3 | FuelLoad, Track_Evolution, Cumulative_Field_Dist_KM |
| Tyre model | 3 | Energy_Lap, Lap_Damage, Accumulated_Tyre_Wear |
| Aero model | 4 | Gap_To_Car_Ahead, Dirty_Air_Fraction, Aero_Loss, DRS_Fraction |
| Racing line | 2 | LatOffset_Mean, LatOffset_Std |
| Clustering | 17 | P_0, P_1, P_2, Style_Cluster_ID, Style_Entropy, Mean/Std (Apex_Speed_Ratio, Brake_Fraction, Brake_Point_Norm, Throttle_On_Dist_Norm, Throttle_Integral_Norm, Speed_Variability) |

## Setup

### Requirements

- Python 3.10+
- Dependencies: `pip install -r requirements.txt`

```
fastf1
pandas
numpy
scipy
scikit-learn
xgboost
lightgbm
catboost
matplotlib
seaborn
optuna
shap
```

### Running the Pipeline

1. Build the dataset (fetches telemetry, creates features)
   ```bash
   python scripts/run_data_pipeline.py
   ```
2. Build the Pirelli dataset (combine all years)
   ```bash
   python scripts/run_combine_tracks.py
   ```
3. (Optional) Tune hyperparameters
   ```bash
   python scripts/run_hyperparameter_tuning.py --trials 50
   ```
4. Train the primary model
   ```bash
   python scripts/run_model_training.py
   ```
5. Compare all models
   ```bash
   python scripts/run_model_comparison.py
   ```
6. Run ablation studies
   ```bash
   python scripts/run_model_no_clustering.py
   python scripts/run_model_comparison_k2.py
   python scripts/run_model_training.py --no-shift
   python scripts/run_model_no_correlated.py
   ```
7. Generate correlation analysis
   ```bash
   python scripts/run_correlation.py
   ```
8. Run SHAP analysis (LightGBM by default)
   ```bash
   python scripts/run_shap_analysis.py
   python scripts/run_shap_analysis.py --model XGBoost
   python scripts/run_shap_analysis.py --model CatBoost
   ```
9. Run GMM clustering for a specific race (with visualisations)
   ```bash
   python scripts/run_clustering_single.py --year 2023 --location Silverstone
   ```
10. K-comparison study
    ```bash
    python scripts/run_cluster_k_comparison.py
    ```

### Configuration

All configurable parameters are centralised in [src/config.py](src/config.py):

- `PRIMARY_MODEL` — which model `run_model_training.py` uses
- `MODELS_TO_COMPARE` — list of models for comparison and ablations
- `MIN_TRAIN_RACES` — races before first prediction (default: 5)
- `MODEL_FEATURES` — full 65-feature list
- `MODEL_FEATURES_NO_CLUSTER`, `MODEL_FEATURES_K2` — variants for ablations
- `MODEL_DEFAULTS` — default hyperparameters per model
- `BEST_PARAMS_FILE` — Optuna-tuned hyperparameters at the project root
- `N_CLUSTERS` — number of GMM clusters (default: 3)
- `RANDOM_STATE` — global seed for reproducibility (42)
- `HIGH_CORR_THRESHOLD` — Pearson |r| threshold for the no-correlated ablation (0.75)
- `DIRTY_AIR_THRESHOLD_SEC` — gap below which a lap is considered to be in dirty air (2.0)
- `FEATURE_BUCKETS` — feature categories used by SHAP grouping

## Results

### Model Comparison

All models evaluated with Optuna-tuned hyperparameters using walk-forward validation across 78 test races (initial 5 training races, 83 total races):

| Model | MAE (s) | RMSE (s) | R² | MAPE (%) | Training Time |
|---|---|---|---|---|---|
| LightGBM | 0.333 | 0.555 | 0.9973 | 0.38 | 51 s |
| RandomForest | 0.335 | 0.557 | 0.9972 | 0.38 | 294 s |
| XGBoost | 0.337 | 0.564 | 0.9972 | 0.38 | 34 s |
| CatBoost | 0.338 | 0.561 | 0.9972 | 0.39 | 38 s |
| Ridge | 1.924 | 11.012 | -0.080 | 2.05 | 21 s |

### Diagnostic Plots

Each model run generates the following diagnostic visualisations:
- **Predicted vs actual** scatter plot with 45-degree reference line
- **Residual analysis** (residuals vs predicted)
- **Per-race MAE** chronological line chart across all 78 test races
- **Compound breakdown** (MAE, RMSE, R², MAPE per tyre compound)
- **Driver MAE** horizontal bar chart (per-driver prediction accuracy)
- **Feature importance** (top features by gain importance)
- **Season degradation** slopes for a selected driver

The SHAP script additionally produces global SHAP, by-bucket, and over-races plots.

### Key Findings

1. **All tree-based models perform similarly** (MAE range: 0.333–0.338 s), with LightGBM slightly leading while being among the fastest to train.
2. **`Prev_Delta` dominates feature importance** in both gain-based and SHAP rankings — the previous lap-to-lap change is by far the strongest predictor of the next change.
3. **Clustering features have small predictive impact** — the clustering ablation shows that removing all 17 clustering features changes MAE by less than 0.001 s across all models. The driving style information is largely already captured by the underlying telemetry features (`Energy_Lap`, lateral offset, etc.) from which the clusters are derived. Clustering therefore serves more as an interpretability tool (driver profiling) than as an accuracy lever.
4. **K=3 vs K=2 makes essentially no difference** — MAE deltas are within 0.0005 s across all models.
5. **Disabling the 1-lap telemetry shift barely improves MAE** but inflates RMSE consistency — confirming that the leakage-free setup is a more honest estimate of forecasting accuracy.
6. **Ridge regression is significantly worse** (MAE ~1.92 s, negative R²), confirming non-linear relationships in lap time dynamics.
7. **Street circuits are hardest to predict** — Monaco and Marina Bay consistently show MAE > 0.6 s due to higher variance from traffic, safety car restarts, and narrow track characteristics.

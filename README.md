# Formula 1 Race Pace Prediction Using Machine Learning with Integrated Vehicle Dynamics and Driver Styles

A machine learning pipeline that predicts Formula 1 lap times by combining telemetry-derived vehicle dynamics, physics-based models, and GMM-based driving style clustering. The system uses walk-forward temporal validation across 4 seasons (2022-2025), achieving a best MAE of **0.334 seconds** (R^2 = 0.9973) on absolute lap times.

Telemetry features are shifted by 1 lap, making this a forecasting model: it uses data available up to lap N-1 to predict lap N time.

## Project Structure

```
formula1_thesis/
├── src/
│   ├── config.py                 # all paths, constants, feature lists, model defaults
│   ├── utils.py                  # shared logging utility
│   ├── logging_setup.py          # logger configuration
│   ├── clustering/               # Driving style clustering
│   │   ├── corners.py            # corner extraction, curvature, zone building
│   │   ├── gmm.py                # GMM fitting, normalization, label alignment
│   │   ├── pipeline.py           # clustering model pipeline
│   │   └── plots.py              # cluster verification plots
│   ├── telemetry/                # Telemetry feature engineering
│   │   ├── features.py           # energy, dirty air, lateral offset, DRS
│   │   ├── wear.py               # accumulated tyre wear, grip model
│   │   └── pipeline.py           # telemetry feature generation pipeline
│   ├── data_preparation/         # Data loading and cleaning
│   │   ├── loading.py            # FastF1 race loading, weather merge
│   │   ├── physics.py            # fuel load model, track evolution
│   │   ├── cleaning.py           # wet lap removal, lag features, encoding
│   │   └── combine_tracks.py     # per-year Pirelli track parameter merging
│   ├── modeling/                 # Model training and analysis
│   │   ├── training.py           # walk-forward CV, feature shifting, delta conversion
│   │   ├── analysis.py           # feature importance, slope analysis, model comparison
│   │   └── plots.py              # diagnostic plots
│   └── visualization/
│       └── clustering_plots.py   # centroid profiles, feature space, driver composition
│
├── scripts/                      # Runnable entry points
│   ├── run_data_pipeline.py      # build 4 year dataset
│   ├── run_model_training.py     # train model with walk-forward validation
│   ├── run_model_comparison.py   # compare all 5 models with diagnostic plots
│   ├── run_hyperparameter_tuning.py  # Optuna Bayesian tuning (TimeSeriesSplit CV)
│   ├── run_model_no_clustering.py    # with vs without clustering features model training
│   ├── run_clustering_single.py      # single-race clustering for inspection
│   ├── run_cluster_visualization.py  # cluster visualizations
│   ├── run_correlation.py            # correlation matrix analysis
│   └── run_combine_tracks.py         # merge per-year Pirelli track parameter CSVs
│
├── data/
│   ├── raw/                      # Pirelli track parameters
│   └── processed/                # generated dataset
├── results/
│   ├── model/                    # per-model results, plots, best_params.json
│   ├── clustering/               # cluster timelines, verification plots
│   ├── correlation/              # correlation matrix
│   └── visualizations/           # cluster visualizations
├── logs/                         # error logs and training summaries
├── cache/                        # FastF1 telemetry cache
├── notebooks/                    # Jupyter experiments
└── archive/                      # old experimental files
```

## Methodology

### Data Collection and Feature Engineering

Race data is fetched via the FastF1 API for seasons 2022-2025 (83 races, ~57,000 clean laps). The pipeline engineers 66 features across several domains:

#### 1. Data Cleaning

Raw laps are filtered to keep only green-flag, dry conditions:
- Wet laps are removed by detecting the first rainfall timestamp and dropping all subsequent laps (as track evolution is unpredictable)
- Safety car / VSC laps (`TrackStatus != '1'`) are excluded
- Pit in/out laps, deleted laps, and FastF1-generated (interpolated) laps are removed

#### 2. Physics-Based Features

**Fuel load model** estimates remaining fuel mass per lap. The model starts at 110 kg (assume that 1 kg should left at the end of the race) and applies a linear burn rate adjusted by track status:
- Green flag: 100% burn rate
- Safety car / VSC: 35% burn rate (reduced engine load)
- Pit in/out laps: 120% burn rate (limiter + acceleration)

**Track evolution** models rubber build-up as the cumulative distance driven by all cars multiplied by the Pirelli Track_Evolution rating. This captures the grip improvement over a race as more rubber is laid down.

#### 3. Telemetry Features

Per-lap telemetry is processed from ~7.5 Hz car data to extract:

- **Lap energy**: combined longitudinal and lateral acceleration integrated over the lap. 
    - Longitudinal acceleration is computed from speed gradients.
    - Lateral acceleration is derived from GPS curvature ($k = (dx*ddy - dy*ddx) / (dx^2 + dy^2)^{1.5}$) multiplied by $v^2$.

    The total $E_{lap} = sum(sqrt(a_{long}^2 + a_{lat}^2) * v * dt)$.
- **Dirty air fraction**: fraction of lap distance where the time gap to the car ahead is < 2.0 seconds, computed via FastF1's `add_driver_ahead()`. Dirty air reduces aerodynamic downforce.
- **DRS fraction**: fraction of lap distance where DRS is active.
- **Lateral offset**: deviation from the fastest lap's racing line. The reference line is built by interpolating the fastest qualifying lap's smoothed (X, Y) coordinates against normalized distance. Each target lap's offset is projected onto the normal vector of the reference line.

#### 4. Tyre Wear Model

A physics-inspired tyre degradation model accumulates wear per stint:

`Lap_Damage = E_lap * P_surface * M_aero`

where:
- `P_surface = Wear_Severity_Index * Compound_Int` (track abrasion * compound hardness)
- `M_aero = 1.15` if gap to car ahead < 2s (as dirty air causes sliding), else `1.0`

`Accumulated_Tyre_Wear` is the cumulative sum of `Lap_Damage` within each stint.

**Tyre grip index** models instantaneous grip as:

`Tyre_Grip_Index = compound_mu * (1 + 0.002*(TrackTemp - 30)) * (1 - 0.15*wear_norm)`

where `compound_mu` is 1.2/1.0/0.85 for SOFT/MEDIUM/HARD and `wear_norm` is the normalized cumulative wear within the stint.

#### 5. Track Parameters

Pirelli publishes per-circuit ratings (1-5 scale) for: Traction, Asphalt Grip, Asphalt Abrasion, Track Evolution, Tyre Stress, Braking, Lateral, and Downforce. These are merged per year along with minimum tyre pressures (front and back), circuit length, and compound hardness numbers.

Derived features include:
- `Wear_Severity_Index = Abrasion * Tyre_Stress`
- `Track_Flow_Type = Lateral / (Traction + Braking + 1)` (cornering or straight-line track)
- `Grip_Aero_Balance = Grip / (Downforce + 1)`

#### 6. Lag Features

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

Corners are identified from the circuit info provided by FastF1. For each corner, a +/-100m zone around the corner distance is defined. Curvature is computed from smoothed GPS coordinates using Savitzky-Golay filtering.

#### 2. Per-Corner Feature Extraction

For each corner zone on each lap, 6 metrics are extracted:
- **Apex Speed Ratio**: `min_speed / entry_speed` -- how much speed the driver carries through the apex
- **Brake Fraction**: fraction of corner distance spent braking
- **Brake Point Norm**: normalized distance of first braking point within the corner
- **Throttle On Dist Norm**: normalized distance from apex to first throttle application (> 20%)
- **Throttle Integral Norm**: integrated throttle area post-apex, normalized by distance and max throttle
- **Speed CV**: coefficient of variation of speed through the corner (smoothness)

#### 3. Lap Aggregation

Corner features are aggregated to lap level using weighted means and standard deviations. Corner weights combine two factors:
- **Time weight**: full-throttle distance on the subsequent straight (longer straight = more time gained/lost from corner exit speed)
- **Energy weight**: integrated lateral g-force * speed through the corner

Weights are normalized and averaged (50/50) to give each corner an importance score.

#### 4. GMM Fitting and Label Alignment

Lap features are normalized using robust scaling (median + IQR) with 3-sigma clipping. A 3-component full-covariance GMM is fitted (`n_init=10`).

Clusters are ordered by an aggression score (`-Throttle_On_Dist_Norm + Throttle_Integral_Norm`) and aligned across races using the Hungarian algorithm on centroid distances.

The 3 identified styles:
- **Cluster 0 -- Exit Attack**: early throttle application, aggressive corner exit
- **Cluster 1 -- Speed Carry**: high minimum speed through corners, smooth trajectory
- **Cluster 2 -- Throttle Save**: conservative throttle, fuel or tyre management

Per-lap outputs: cluster probabilities (P_0, P_1, P_2), cluster ID, and style entropy (`-sum(p * log(p))`).

---

### Feature Shifting (no data leakage)

To make the model a genuine forecasting tool, 27 telemetry-derived features are shifted by 1 lap within each driver-stint group. This means the model uses lap N-1's telemetry (energy, dirty air, lateral offset, DRS, tyre wear, cluster probabilities, etc.) to predict lap N's time change. Rows with NaN from shifting are dropped.

---

### Prediction Target

The model predicts `Target_Delta` -- the lap-to-lap time change ($LapTime_N - LapTime_{N-1}$), not the absolute lap time. This because:
- it removes circuit-specific baseline effects (as each track has different lap times)
- centers the target around zero, which is easier for tree models to learn

Absolute times are recovered after prediction (`Predicted_Time = Prev_LapTime + Predicted_Delta`)

---

### Walk-Forward Validation

The model uses temporal walk-forward validation to prevent data leakage:

1. Races are sorted chronologically across all seasons
2. The first `MIN_TRAIN_RACES` (default: 5) races form the initial training set
3. For each subsequent race, the model trains on all previous races and predicts the current race
4. Add the race to the training set and repeat

This shows the reality: the model only sees historical data and never learns from future runs.

### 6. Models

Five regression models are compared:

- **Ridge** -- linear baseline (wrapped in Pipeline with `SimpleImputer` + `StandardScaler` since it cannot handle NaN and requires normalization)
- **RandomForest** -- decision trees (wrapped in Pipeline with `SimpleImputer`)
- **XGBoost** -- gradient boosted trees
- **LightGBM** -- gradient boosted trees with histogram-based splits
- **CatBoost** -- gradient boosted trees with ordered boosting

XGBoost, LightGBM and CatBoost handle NaN values natively and do not require scaling.

---

### Hyperparameter Tuning

Bayesian optimization via Optuna with `TimeSeriesSplit` (5 folds) as the cross-validation strategy. This ensures temporal ordering is preserved during tuning. Tuned parameters are saved to `results/model/best_params.json` and automatically loaded by training scripts.

---

### Ablation Study

To assess the contribution of driving style clustering, the model is trained with the full 66-feature set and again with 49 features (17 clustering-related features removed: Mean/Std of 6 corner metrics, P_0, P_1, P_2, Style_Cluster_ID, Style_Entropy). The delta in $MAE/RMSE/R^2$ between the two runs quantifies the clustering contribution.

---

## Feature Categories

Extracted total 66 features

| Category | Count | Examples |
|---|---|---|
| Race state | 4 | LapNumber, Stint, TyreLife, Position |
| Weather | 6 | AirTemp, TrackTemp, Humidity, WindSpeed, WindDirection, Pressure |
| Physics | 3 | FuelLoad, Track_Evolution_Physics, Grip_Aero_Balance |
| Track parameters | 12 | Traction, Braking, Lateral, Downforce, Tyre_Stress, Wear_Severity_Index, etc. |
| Telemetry | 12 | E_lap, Dirty_Air_Fraction, DRS_Fraction, LatOffset_Mean, Aero_Loss, etc. |
| Tyre model | 5 | Tyre_Grip_Index, Accumulated_Tyre_Wear, Compound_Int, Tyre_Compound_Interaction, Lap_Damage |
| Clustering | 17 | P_0, P_1, P_2, Style_Cluster_ID, Style_Entropy, Mean/Std corner features |
| Lag features | 4 | Prev_LapTime, Lag_2, Rolling_Avg_3, Prev_Delta |
| Encoded | 3 | Driver_Encoded, Location_Encoded, Team_Encoded |

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
```

### Running the Pipeline

```bash
# 1. Build the dataset (fetches telemetry, creates features)
python scripts/run_data_pipeline.py

# 2. Build the Pirelli dataset (combine all years)
python scripts/run_combine_tracks.py

# 3. (Optional) Tune hyperparameters
python scripts/run_hyperparameter_tuning.py --trials 50

# 4. Train the primary model
python scripts/run_model_training.py

# Compare all models
python scripts/run_model_comparison.py

# Run ablation study
python scripts/run_model_no_clustering.py

# Generate correlation analysis
python scripts/run_correlation.py

# Runs GMM clustering model for a specific race (+ visualisations)
python scripts/run_clustering_single.py --year 2023 --location Silverstone

# Generate cluster visualizations for a specific race
python scripts/run_cluster_visualization.py --location Silverstone --year 2023
```

### Configuration

All configurable parameters are centralized in [config.py](src/config.py):

- `PRIMARY_MODEL` -- which model `run_model_training.py` uses
- `MIN_TRAIN_RACES` -- races before first prediction (default: 5)
- `MODEL_FEATURES` -- full 66-feature list
- `MODEL_DEFAULTS` -- default hyperparameters per model
- `N_CLUSTERS` -- number of GMM clusters (default: 3)

## Results

### Model Comparison

All models evaluated with Optuna-tuned hyperparameters using walk-forward validation across 78 test races (initial 5 races)

| Model | MAE (s) | RMSE (s) | R^2 | MAPE (%) | Training Time |
|---|---|---|---|---|---|
| LightGBM | 0.334 | 0.555 | 0.9973 | 0.38 | 37 s |
| RandomForest | 0.335 | 0.555 | 0.9973 | 0.38 | 1180 s |
| XGBoost | 0.336 | 0.561 | 0.9972 | 0.38 | 58 s |
| CatBoost | 0.339 | 0.561 | 0.9972 | 0.39 | 651 s |
| Ridge | 1.962 | 11.461 | -0.1669 | 2.09 | 36 s |

### Diagnostic Plots

Each model generates 7 diagnostic visualizations:
- **Predicted vs actual** scatter plot with 45-degree reference line
- **Residual analysis** (residuals vs predicted + histogram with mean/std)
- **Per-race MAE** chronological line chart across all 78 test races
- **Compound breakdown** (MAE, RMSE, R², MAPE per tyre compound)
- **Driver MAE** horizontal bar chart (per-driver prediction accuracy)
- **Feature importance** (top features ranked by model-specific importance)
- **Season degradation** slopes for a selected driver

### Key Findings

1. **All tree-based models perform similarly** (MAE range: 0.334-0.339 s), with LightGBM slightly leading while being the fastest to train
2. **Prev_Delta dominates feature importance** (12-75% depending on model), followed by lag features and tyre-related features. This is expected -- the previous lap-to-lap change is the strongest predictor of the next change
3. **Clustering features have snall predictive impact** -- the no clustering study shows removing all 17 clustering features changes MAE by less than 0.001 s across all models. The driving style information is already captured by the underlying telemetry features (E_lap, lateral offset, etc.) from which the clusters are derived
4. **Street circuits are hardest to predict** -- Monaco and Marina Bay consistently show MAE > 0.6 s due to higher variance from traffic, safety car restarts, and narrow track characteristics
5. **Ridge regression is significantly worse** (MAE ~1.96 s), confirming non-linear relationships in lap time dynamics
6. **Model accuracy improves over the season** -- early races (smaller training sets) show higher MAE, but became more stable as the training set increased

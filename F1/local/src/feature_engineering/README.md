# F1/local/src/feature_engineering - Feature Engineering Module

This module is responsible for creating and transforming features from the consolidated dataset, preparing it for the subsequent preprocessing and modeling stages.

## Key Components

-   **`feature_engineering.py`**:
    -   Contains a collection of functions, each designed to create specific types of features or perform data transformations.
    -   **Core Functionalities**:
        -   `create_target_variables()`: Defines the primary target (`tire_change_next_lap`) and secondary target (`next_tire_type`).
        -   `create_temporal_features()`: Generates features based on time, lap progression, stint information (e.g., `TyreLife`, `stint_progress`, `lap_progress`).
        -   `create_performance_features()`: Creates features related to driver performance and race context (e.g., `laptime_trend_3`, `delta_ahead_trend`, `is_top_3`, `tire_degradation_rate`).
        -   `create_weather_features()`: Derives features from weather data (e.g., `AirTemp_stability`, `difficult_conditions`).
        -   `create_domain_knowledge_features()`: Implements features based on F1 domain expertise (e.g., `expected_stint_length`, `in_pit_window_early/mid/late`).
        -   `handle_missing_values()`: Imputes missing values using strategies like median imputation for numerical features and constant filling for others. Saves imputation values to `F1/local/drive/artifacts/imputer_values.json`.
        -   `handle_categorical_variables()`: Encodes categorical features (e.g., `Compound`, `Team`, `Driver`) using `sklearn.preprocessing.LabelEncoder`. Saves fitted encoders to `F1/local/drive/artifacts/encoders.pkl`.
        -   `normalize_features()`: Scales numerical features using `sklearn.preprocessing.RobustScaler` (chosen for its resilience to outliers). Saves the fitted scaler to `F1/local/drive/artifacts/scaler.pkl`.

-   **`create_features_pipeline.py`**:
    -   The main script that orchestrates the entire feature engineering workflow.
    -   **Functionality**:
        -   Loads the consolidated dataset from `F1/local/drive/processed_data/dataset.parquet`.
        -   Sequentially calls the various feature creation and transformation functions from `feature_engineering.py`.
        -   Ensures that artifacts (imputation values, encoders, scaler) are saved correctly to `F1/local/drive/artifacts/`.
        -   Saves the final feature-engineered dataset to `F1/local/drive/feature_engineered_data/featured_dataset.parquet`.
        -   Logs progress and information about the created features.
    -   **Execution**:
        -   Can be run standalone if the consolidated dataset is available.
        -   Is called as the third step in the main `F1/local/run_pipeline.py` script.

## Input

-   Consolidated dataset: `F1/local/drive/processed_data/dataset.parquet`.

## Output

-   **Feature-Engineered Dataset**: `F1/local/drive/feature_engineered_data/featured_dataset.parquet`. This dataset contains all original and newly created features, ready for sequence creation.
-   **Artifacts**: Stored in `F1/local/drive/artifacts/`:
    -   `imputer_values.json`
    -   `encoders.pkl`
    -   `scaler.pkl`

## Purpose

This module aims to enrich the dataset with features that are potentially predictive for the tyre strategy tasks. It also handles essential data preparation steps like missing value imputation, categorical encoding, and numerical scaling to make the data suitable for machine learning models.

# F1/local/src - Source Code Directory

This directory contains the Python source code for the local F1 tyre strategy prediction pipeline. The code is organized into modules based on their role in the overall workflow.

## Modules

-   **`data_extraction/`**:
    -   Contains scripts responsible for fetching raw Formula 1 data using the FastF1 library.
    -   Key script: `fetch_fastf1_data.py` (loads configuration, fetches data for specified races/sessions, saves to `F1/local/drive/raw_data/`).

-   **`data_processing/`**:
    -   Contains scripts for cleaning, consolidating, and performing initial transformations on the raw data.
    -   Key script: `consolidate_data.py` (reads individual raw race files, merges them, adds metadata, and saves a consolidated dataset to `F1/local/drive/processed_data/`).

-   **`feature_engineering/`**:
    -   Contains scripts dedicated to creating new features from the consolidated dataset to be used for modeling.
    -   Key scripts:
        -   `feature_engineering.py`: Defines functions for creating various types of features (temporal, performance, weather-based, domain-specific) and handling categorical/numerical data.
        -   `create_features_pipeline.py`: Orchestrates the application of feature engineering functions, saves the feature-engineered dataset to `F1/local/drive/feature_engineered_data/`, and stores artifacts (encoders, scalers) in `F1/local/drive/artifacts/`.

-   **`modeling/`**:
    -   Contains all scripts related to the RNN model itself.
    -   Key scripts:
        -   `model_def.py`: Defines the `LSTMTirePredictor` (the RNN architecture) and `CombinedLoss` classes, along with helper functions to load model hyperparameters.
        -   `data_loaders.py`: Defines the `F1SequenceDataset` (custom PyTorch Dataset) for loading preprocessed sequence data and `create_dataloaders` to generate PyTorch DataLoaders.
        -   `training_utils.py`: Provides utility classes like `EarlyStopping`, `TrainingMetricsTracker`, and the main `ModelTrainer` class that encapsulates the training and validation loops. Also includes `find_optimal_threshold_on_pr_curve` and `create_trainer_components`.
        -   `train_model.py`: The main script to orchestrate the model training process, using components from other modeling scripts and configurations.
        -   `evaluate.py`: Contains the `ModelEvaluator` class and `generate_and_save_evaluation_report` function for comprehensive model performance assessment and visualization.

-   **`preprocessing/`**:
    -   Contains scripts for transforming the feature-engineered data into a format suitable for the RNN model (e.g., creating sequences).
    -   Key script: `prepare_model_data.py` (loads feature-engineered data, performs temporal train/val/test splits, creates fixed-length sequences, and saves them to `F1/local/drive/model_input/`).

## General Notes

-   Most scripts are designed to be run as part of the main pipeline orchestrated by `F1/local/run_pipeline.py`.
-   Configuration for these scripts is primarily managed through YAML files in the `F1/local/configs/` directory.
-   Logging is implemented in most scripts to track progress and errors.
-   The code aims to be modular, allowing individual components to be tested or modified.

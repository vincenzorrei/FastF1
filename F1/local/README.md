# F1 Local Development Environment

This directory (`F1/local/`) contains a self-contained local development and testing environment for the F1 Tyre Strategy Prediction project. Its purpose is to allow for rapid iteration, development, and testing of the entire data processing and modeling pipeline before transitioning to a cloud-based environment like Google Colab.

## Directory Structure

-   **`configs/`**: Contains YAML configuration files for various pipeline stages:
    -   `data_extraction_config.yaml`: Parameters for fetching data using FastF1 (e.g., years, races, API settings, paths).
    -   `model_config.yaml`: Hyperparameters for the RNN model, training settings (learning rate, epochs, batch size), and loss function parameters.
-   **`drive/`**: Simulates a cloud drive structure (like Google Drive) for storing data and artifacts locally.
    -   `artifacts/`: Stores reusable components generated during preprocessing (e.g., encoders, scalers, feature lists).
    -   `logs/`: Stores logs from various pipeline scripts (e.g., `run_pipeline.log`).
        -   `training_logs/`: Specific logs from model training, including TensorBoard logs and training summaries.
    -   `model_input/`: Stores the final preprocessed sequence data (`.npy` files) ready for model training.
    -   `models/`: Stores saved model checkpoints (e.g., `best_model.pth`).
    -   `processed_data/`: Stores the consolidated dataset after initial cleaning and merging (e.g., `dataset.parquet`).
    -   `raw_data/`: Stores raw data fetched by FastF1, typically one Parquet file per race session.
    -   `data_download_log.csv`: Logs the status of data extraction for each race.
    -   `evaluation_reports/`: Stores evaluation reports and plots.
-   **`logs/`**: General logging directory for scripts that might not write to `drive/logs/`. (e.g. `consolidate_data.log`)
-   **`notebooks/`**: (Currently empty) Intended for Jupyter notebooks for experimentation, analysis, or visualization.
-   **`src/`**: Contains all Python source code for the pipeline, organized into sub-modules:
    -   `data_extraction/`: Scripts for fetching data from FastF1.
    -   `data_processing/`: Scripts for cleaning and consolidating raw data.
    -   `feature_engineering/`: Scripts for creating features from consolidated data.
    -   `modeling/`: Scripts for model definition, data loading for the model, training, and evaluation.
    -   `preprocessing/`: Scripts for preparing feature-engineered data into sequences for the RNN model.
-   `requirements.txt`: Python package dependencies for this local environment.
-   `run_pipeline.py`: Main orchestration script to run the entire local pipeline end-to-end.
-   `steps.md`: Markdown file tracking the development progress and to-do items for the `F1/local/` setup.

## Workflow

The primary workflow is orchestrated by `run_pipeline.py`, which executes the following stages in sequence:

1.  **Data Extraction** (`src/data_extraction/fetch_fastf1_data.py`):
    -   Reads `configs/data_extraction_config.yaml`.
    -   Fetches specified F1 race data using the FastF1 library.
    -   Saves raw lap and weather data to `drive/raw_data/`.
    -   Updates `drive/data_download_log.csv`.
2.  **Data Consolidation** (`src/data_processing/consolidate_data.py`):
    -   Reads individual raw race files from `drive/raw_data/`.
    -   Cleans, merges, and adds metadata.
    -   Saves the consolidated dataset to `drive/processed_data/dataset.parquet`.
3.  **Feature Engineering** (`src/feature_engineering/create_features_pipeline.py`):
    -   Loads `drive/processed_data/dataset.parquet`.
    -   Applies feature creation logic from `src/feature_engineering/feature_engineering.py`.
    -   Saves the feature-engineered dataset to `drive/feature_engineered_data/featured_dataset.parquet`.
    -   Saves artifacts like encoders and scalers to `drive/artifacts/`.
4.  **Model Data Preparation** (`src/preprocessing/prepare_model_data.py`):
    -   Loads `drive/feature_engineered_data/featured_dataset.parquet` and artifacts.
    -   Performs temporal train/test (and potentially validation) splits.
    -   Creates input sequences for the RNN model.
    -   Saves sequences (`X_train.npy`, `y_train_change.npy`, etc.) to `drive/model_input/`.
    -   Saves the list of feature columns used to `drive/artifacts/feature_columns.joblib`.
5.  **Model Training** (`src/modeling/train_model.py`):
    -   Loads `configs/model_config.yaml`.
    -   Loads prepared model input data from `drive/model_input/` and artifacts from `drive/artifacts/`.
    -   Initializes the RNN model (`src/modeling/model_def.py`), optimizer, and loss function.
    -   Trains the model using utilities from `src/modeling/training_utils.py` and data loaders from `src/modeling/data_loaders.py`.
    -   Saves model checkpoints to `drive/models/` and logs/summaries to `drive/logs/training_logs/`.
6.  **Final Model Evaluation** (within `run_pipeline.py` calling `src/modeling/evaluate.py`):
    -   Loads the best trained model from `drive/models/best_model.pth`.
    -   Evaluates it on the test set data from `drive/model_input/`.
    -   Saves evaluation reports and plots to `drive/evaluation_reports/pipeline_test_set/`.

## Running the Pipeline

To run the entire local pipeline:
1.  Ensure all dependencies from `requirements.txt` are installed in your Python environment.
2.  Verify configurations in `F1/local/configs/`. For initial testing, ensure `data_extraction_config.yaml` specifies a small number of races and `model_config.yaml` specifies a small number of epochs.
3.  From the root directory of the project (`FASTF1/`), execute:
    ```bash
    python F1/local/run_pipeline.py
    ```
    Alternatively, navigate to `F1/local/` and run `python run_pipeline.py`.

Individual pipeline scripts can also be run for more granular testing, but ensure preceding steps have generated the necessary inputs.

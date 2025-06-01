# F1/local/configs - Configuration Files

This directory stores YAML configuration files used by various scripts in the `F1/local/` pipeline.

## Files

-   **`data_extraction_config.yaml`**:
    -   Governs the behavior of the `F1/local/src/data_extraction/fetch_fastf1_data.py` script.
    -   **Key Parameters**:
        -   `races_to_download`: A list specifying which years and race events (by name or round number) to download. Can also specify session types (e.g., 'R' for Race).
        -   `fastf1_cache_path`: Path to the directory where FastF1 will cache its data (e.g., API responses).
        -   `output_dir_raw_data`: Base directory where raw Parquet files for each race session will be saved (typically `F1/local/drive/raw_data/`).
        -   `log_path`: Path for the data extraction log file (`F1/local/drive/data_download_log.csv`).
        -   `api_settings`: Parameters for FastF1 API interaction, like `rate_limit_pause_seconds` and `max_retries`.
        -   `data_quality`: Settings like `nan_threshold_critical_cols` for data validation during extraction.
        -   `columns_to_extract`: Specifies the exact list of raw columns to be extracted for lap data and weather data.

-   **`model_config.yaml`**:
    -   Provides configuration for the model architecture, training process, and related data paths. Used by scripts in `F1/local/src/modeling/`.
    -   **Key Sections**:
        -   `model`:
            -   `rnn`: Parameters for the LSTM layers (e.g., `hidden_size`, `num_layers`, `dropout_lstm`).
            -   `heads`: Parameters for the task-specific prediction heads (e.g., `dropout_head`).
            -   `loss`: Parameters for the `CombinedLoss` function (e.g., `alpha`, `beta` weights for tasks, `pos_weight_tire_change` for class imbalance).
        -   `training`:
            -   General training settings: `device` (`auto`, `cpu`, `cuda`), `batch_size`, `learning_rate`, `epochs`.
            -   `sequence_length`: Must match the sequence length used during model data preparation.
            -   `early_stopping` (optional): Parameters like `patience` and `min_delta`.
            -   `scheduler` (optional): Configuration for learning rate schedulers (e.g., `ReduceLROnPlateau`, `StepLR`) with their respective parameters.
        -   `data`: (Primarily for reference, as scripts often derive these paths)
            -   `artifacts_dir`: Path to where artifacts like encoders, scalers, and feature lists are stored (e.g., `F1/local/drive/artifacts/`).
            -   `model_input_dir`: Path to where the final `.npy` sequence files for model input are stored (e.g., `F1/local/drive/model_input/`).

## Usage

These configuration files allow for easy modification of pipeline behavior without changing the source code. Scripts load these YAML files at runtime to get their operational parameters. Ensure paths are correctly specified, especially if running scripts from different working directories or if the project structure is modified. Paths are generally expected to be relative to the `F1/local/` directory or handled appropriately by the scripts.

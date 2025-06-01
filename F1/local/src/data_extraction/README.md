# F1/local/src/data_extraction - Data Extraction Module

This module is responsible for fetching raw Formula 1 race data using the FastF1 Python library.

## Key Components

-   **`fetch_fastf1_data.py`**:
    -   The main script for data extraction.
    -   **Functionality**:
        -   Loads configuration from `F1/local/configs/data_extraction_config.yaml`. This includes specifying years, race events, session types (typically 'R' for Race), FastF1 cache path, output directories, API settings, and columns to extract.
        -   Iterates through the specified races and sessions.
        -   Uses FastF1 to load session data, including lap timings and weather information.
        -   Performs basic data quality checks (e.g., percentage of NaNs in critical columns).
        -   Saves the extracted raw data for each session as a Parquet file in `F1/local/drive/raw_data/YEAR/SESSION_NAME/`.
        -   Logs the status of each download attempt (success, failure, reasons) to `F1/local/drive/data_download_log.csv`.
        -   Includes error handling for API issues, connection problems, and missing data, with retry mechanisms.
    -   **Execution**:
        -   Can be run standalone for data extraction tasks.
        -   Is called as the first step in the main `F1/local/run_pipeline.py` script.

## Configuration

-   The behavior of `fetch_fastf1_data.py` is primarily controlled by `F1/local/configs/data_extraction_config.yaml`.
-   Ensure the `fastf1_cache_path` is set to a valid directory to enable FastF1's caching mechanism, which significantly speeds up subsequent data loads for the same sessions.

## Output

-   **Raw Data**: Parquet files stored in `F1/local/drive/raw_data/`, organized by year and session.
-   **Download Log**: `F1/local/drive/data_download_log.csv` provides a record of all download attempts and their outcomes.

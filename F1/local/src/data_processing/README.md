# F1/local/src/data_processing - Data Processing Module

This module handles the cleaning, consolidation, and initial transformation of raw Formula 1 data fetched by the `data_extraction` module.

## Key Components

-   **`consolidate_data.py`**:
    -   The main script for processing and consolidating raw data.
    -   **Functionality**:
        -   Scans the `F1/local/drive/raw_data/` directory for individual race session Parquet files.
        -   For each file:
            -   Performs data cleaning (e.g., handling missing critical values, ensuring correct data types).
            -   Adds essential metadata columns such as `Year`, `EventName`, `Country`, `CircuitName`, `RaceID` (unique ID for each race event), `DriverRaceID` (unique ID for each driver within a race), and `GlobalLapID` (unique ID for each lap across all data).
        -   Concatenates all processed individual race DataFrames into a single, unified DataFrame.
        -   Performs final validation checks on the consolidated dataset.
        -   Saves the consolidated dataset as `dataset.parquet` in `F1/local/drive/processed_data/`.
        -   Generates a `consolidation_report.txt` in the same directory, summarizing the process (e.g., files processed, rows before/after cleaning, total rows).
    -   **Execution**:
        -   Can be run standalone if raw data is present.
        -   Is called as the second step in the main `F1/local/run_pipeline.py` script, after data extraction.

## Input

-   Raw Parquet files from `F1/local/drive/raw_data/`, typically organized by year and session.

## Output

-   **Consolidated Dataset**: `F1/local/drive/processed_data/dataset.parquet`. This file serves as the primary input for the feature engineering stage.
-   **Consolidation Report**: `F1/local/drive/processed_data/consolidation_report.txt`.

## Purpose

The goal of this module is to transform the collection of raw, per-session data files into a single, clean, and consistently structured dataset that is ready for feature engineering. This involves standardizing data types, handling inconsistencies, and enriching the data with unique identifiers crucial for subsequent analysis and modeling.

# F1/local/src/preprocessing - Data Preprocessing for Model Module

This module focuses on transforming the feature-engineered dataset into a format suitable for direct input into the RNN (Recurrent Neural Network) model. This primarily involves creating sequences of data.

## Key Components

-   **`prepare_model_data.py`**:
    -   The main script for this preprocessing stage.
    -   **Functionality**:
        -   Loads the feature-engineered dataset from `F1/local/drive/feature_engineered_data/featured_dataset.parquet`.
        -   Loads necessary artifacts, such as encoders, from `F1/local/drive/artifacts/` (though encoders are mainly used by the `create_sequences_for_rnn` function if it needs to re-encode anything, typically target variables).
        -   Performs a temporal train-validation-test split on the data. The splitting strategy is pragmatic:
            -   If 1 year of data: All data for train, empty val/test.
            -   If 2 years (Y1, Y2): Train = Y1, Test = Y2, empty Val.
            -   If 3+ years (Y1, ..., Yn-2, Yn-1, Yn): Train = Y1...Yn-2, Val = Yn-1, Test = Yn.
        -   For each data split (train, val, test):
            -   Calls `create_sequences_for_rnn()` to convert the tabular data into sequences of a fixed length (defined by `SEQUENCE_LENGTH` in the script, typically matching `model_config.yaml`). Each sequence represents a window of past laps' data leading up to a target lap.
            -   The target variables (`tire_change_next_lap`, `next_tire_type`) for each sequence correspond to the last timestep of that sequence.
        -   Saves the generated sequences (X features, y_change targets, y_type targets) as NumPy arrays (`.npy` files) into the `F1/local/drive/model_input/` directory (e.g., `X_train.npy`, `y_change_train.npy`).
        -   Saves the list of feature columns used to create the sequences to `F1/local/drive/artifacts/feature_columns.joblib` (done once, typically with the training set).
        -   If a split is empty (e.g., validation set with only two years of data), it saves empty array representations to ensure consistency and avoid using stale data.
    -   **Execution**:
        -   Can be run standalone if the feature-engineered dataset and necessary artifacts are available.
        -   Is called as the fourth step in the main `F1/local/run_pipeline.py` script.

## Input

-   Feature-engineered dataset: `F1/local/drive/feature_engineered_data/featured_dataset.parquet`.
-   Encoders (primarily for target variable encoding if needed): `F1/local/drive/artifacts/encoders.pkl`.

## Output

-   **Model Input Sequences**: NumPy arrays (`.npy`) stored in `F1/local/drive/model_input/`, separated for train, validation (if any), and test sets.
    -   `X_<split>.npy`: Input feature sequences.
    -   `y_change_<split>.npy`: Target labels for tire change prediction.
    -   `y_type_<split>.npy`: Target labels for tire compound prediction.
-   **Feature List Artifact**: `F1/local/drive/artifacts/feature_columns.joblib`, containing the list of columns used to form the input sequences.

## Purpose

The primary goal of this module is to structure the data into the sequential format required by RNNs. This involves selecting relevant features, creating overlapping windows of data, and aligning them with their corresponding future targets. This step is critical for enabling the model to learn temporal patterns from the F1 lap data.

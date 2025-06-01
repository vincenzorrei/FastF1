# Steps for Local Development and Testing (F1/local/)

This file tracks the defined and completed steps for the local development environment.

## Phase 1: Project Setup and Initial Data Handling

1.  **[x] Setup Project Structure:**
    *   [x] Create necessary subdirectories within `F1/local/` (e.g., `src/data_extraction`, `src/data_processing`, `src/feature_engineering`, `src/modeling`, `notebooks`, `configs`, `drive/raw_data`, `drive/processed_data`, `drive/models`).
    *   [x] Initialize `requirements.txt` for the local environment.

2.  **[x] Data Extraction Module:**
    *   **[x] Analisi Chiamate API FastF1, Rate Limit e Tipi di Sessione:**
        *   Consultare la documentazione di FastF1 (e `domenicoDL/` per eventuali spunti su chiamate API passate) per identificare le chiamate API per ottenere i dati grezzi richiesti.
        *   Approfondire la gestione del rate limiting di FastF1.
        *   **Nota Chiave:** L'estrazione si concentrerà **esclusivamente su sessioni di Gara ('R') per eventi Gran Premio standard**.
        *   **Nota Chiave:** Estrarre sempre le variabili nella loro forma grezza (non normalizzata). La normalizzazione sarà una fase separata del preprocessing.

    *   **[x] Implementazione Script di Estrazione Dati Avanzato (`F1/local/src/data_extraction/fetch_fastf1_data.py`):**
        *   [x] Struttura base dello script creata (caricamento config, logging).
        *   [x] Implementata logica principale di estrazione dati (caricamento sessione, estrazione lap/weather, selezione colonne, controllo NaN critico, salvataggio Parquet).
        *   [x] Implementata gestione degli errori (incl. HTTP, connessione, dati mancanti) e meccanismo di retry.
        *   [x] Lo script legge la configurazione da `F1/local/configs/data_extraction_config.yaml`.
        *   [x] Lo script utilizza FastF1 con caching configurato.
        *   [x] Lo script gestisce il rate limit con pause e retry.
        *   [x] Lo script processa le gare specificate, estrae dati grezzi, e salva in Parquet.
        *   [x] Lo script aggiorna il file di log `F1/local/drive/data_download_log.csv`.
        *   [x] Verifica della qualità dei dati implementata (NaN su colonne critiche).

    *   **[x] Creazione File di Configurazione (`F1/local/configs/data_extraction_config.yaml`):**
        *   Definire i percorsi per cache, output dati grezzi, file di log.
        *   Specificare parametri per rate limit, soglia NaN, tentativi massimi.
        *   Elencare le gare da scaricare (inizialmente un piccolo set per test, solo sessioni 'R' di GP standard).
        *   Definire il formato di output e l'elenco preciso delle colonne grezze da estrarre.

    *   **[x] Strutturazione File di Log dei Download (`F1/local/drive/data_download_log.csv`):**
        *   Definire le colonne: `Year`, `EventName`, `SessionType` (sarà sempre 'R'), `DownloadTimestamp`, `Status` ('Success', 'Failed - RateLimit', 'Failed - ConnectionError', 'Incomplete - HighNaN', 'Skipped - NonStandardEvent'), `NaNPercentage`, `FilePath`, `RetriesAttempted`, `Notes`.

3.  **[x] Data Processing & Consolidation Module:**
    *   [x] Analyze `Vincenzo/dataset/data_consolidation.py` and related scripts for data cleaning, merging, and initial transformation logic.
    *   [x] Implement Data Consolidation Script (`F1/local/src/data_processing/consolidate_data.py`):
        *   [x] Adapt `Vincenzo/dataset/data_consolidation.py` for `F1/local/` structure.
        *   [x] Configure input from `F1/local/drive/raw_data/` and output to `F1/local/drive/processed_data/`.
        *   [x] Ensure logging to `F1/local/logs/data_consolidation.log`.
        *   [x] Align column names (`DriverNumber`, `EventName`) with `data_extraction_config.yaml`.
        *   [x] Implement data validation, cleaning, metadata addition (`RaceID`, `DriverRaceID`, `GlobalLapID`), and concatenation.
        *   [x] Save consolidated `dataset.parquet` and `consolidation_report.txt`.
    *   [x] Test `consolidate_data.py` with sample raw data.
    *   [x] Refine cleaning and validation logic based on test results if necessary.

## Phase 2: Feature Engineering and Preprocessing

4.  **[x] Feature Engineering Module:**
    *   [x] Analyze `Vincenzo/` directory (especially notebooks and Python scripts related to feature creation) for existing feature engineering logic (e.g., `TyreLife`, `LapNumber`, time differences, stint information).
    *   [x] Implement/Refactor feature engineering scripts in `F1/local/src/feature_engineering/` (created `feature_engineering.py` with core functions).
        *   Goal: Create relevant features for predicting tyre change and compound (e.g., `TyreAgeLaps`, `StintLap`, `TimeSinceLastPit`, weather changes, track status). (Largely covered)
        *   Goal: Add target variables (`NextLapPitStop`, `NextLapTyreCompound`). (Covered)
        *   Goal: Save feature-engineered data, possibly augmenting data in `F1/local/drive/processed_data/` or a new `F1/local/drive/feature_engineered_data/` location. (Handled by pipeline script)
    *   [x] Create and test `F1/local/src/feature_engineering/create_features_pipeline.py` to orchestrate feature creation, save feature-engineered data to `F1/local/drive/feature_engineered_data/featured_dataset.parquet`, and save artifacts to `F1/local/drive/artifacts/`. (Completed, noted NaN warning for PitIn/OutTime and missing delta columns to investigate later).

5.  **[x] Data Preprocessing for Model Module:**
    *   [x] Analyze `Vincenzo/dataset/data_preprocessing.py` for scaling, encoding, and sequence creation. (Analysis complete, relevant functions identified).
    *   [x] Implement/Refactor preprocessing scripts in `F1/local/src/preprocessing/` (created `F1/local/src/preprocessing/prepare_model_data.py` with split and sequence creation logic).
        *   [x] Test `F1/local/src/preprocessing/prepare_model_data.py` to ensure correct data splitting, sequence creation, and saving of model inputs to `F1/local/drive/model_input/`. (Script executed successfully, data splits and sequences saved).
        *   Goal: Handle categorical features (e.g., one-hot encoding for `Compound`). (Handled by LabelEncoder in feature engineering; further OHE if needed would be here or in model data loader).
        *   Goal: Scale numerical features. (Handled by RobustScaler in feature engineering).
        *   Goal: Create sequences for RNN input (e.g., a rolling window of previous laps' data for each target lap). (Implemented in `prepare_model_data.py`).
        *   Goal: Split data into training, validation, and test sets. (Implemented in `prepare_model_data.py`).

## Phase 3: Modeling and Evaluation

6.  **[x] Model Definition (RNN/LSTM):**
    *   [x] Analyze `Vincenzo/dataset/models/lstm_architecture.py`.
    *   [x] Implement/Refactor RNN/LSTM model definition in `F1/local/src/modeling/model_def.py` (includes `LSTMTirePredictor`, `CombinedLoss`, and hyperparameter loading logic).
    *   [x] Create a configuration file in `F1/local/configs/model_config.yaml` for model architecture and training parameters.
    *   [x] Basic test of `model_def.py` (initialization and forward pass with dummy data) successful.

7.  **[x] Model Training:**
    *   [x] Analyze `Vincenzo/train_model.py`, `Vincenzo/model_training.ipynb` (analysis based on related scripts due to read error), and `Vincenzo/dataset/models/training_utils.py`.
    *   [x] Implement/Refactor training scripts in `F1/local/src/modeling/` (created `training_utils.py`, `data_loaders.py`, and main `train_model.py`).
        *   [x] Test `F1/local/src/modeling/train_model.py` to ensure it runs, loads data, trains for a few epochs (2 epochs completed), and saves outputs (logs, summary). (Note: Validation set was empty as expected for 2 years of data; `pos_weight` taken from config).
        *   Goal: Load preprocessed data. (Implemented in `data_loaders.py` and used by `train_model.py`)
        *   Goal: Implement training loop, loss functions, optimizers, and callbacks (e.g., early stopping, model checkpointing). (Implemented in `training_utils.py` and orchestrated by `train_model.py`)
        *   Goal: Save trained models and training history to `F1/local/drive/models/` and logs to `F1/local/drive/logs/training_logs/`. (Implemented)

8.  **[x] Model Evaluation:**
    *   [x] Analyze `Vincenzo/dataset/models/evaluation.py`.
    *   [x] Implement/Refactor evaluation scripts in `F1/local/src/modeling/evaluate.py` (includes `ModelEvaluator` and `generate_and_save_evaluation_report`).
        *   [x] Basic test of `evaluate.py` with dummy data successful (metrics calculated, plots generated, report saved).
        *   Goal: Load trained model and test data. (Implemented)
        *   Goal: Calculate relevant metrics (e.g., accuracy, precision, recall, F1-score, ROC AUC for pit stop prediction; appropriate metrics for compound prediction). (Implemented)
        *   Goal: Visualize results (e.g., confusion matrices, ROC curves). (Implemented)

## Phase 4: Workflow Orchestration and Documentation

9.  **[x] Configuration Management:**
    *   [x] Analyze existing config files (`Vincenzo/config.json`, `Vincenzo/dataset/configs/model_config.yaml`, etc.).
    *   [x] Consolidate and create centralized configuration files in `F1/local/configs/` (created `data_extraction_config.yaml` and `model_config.yaml`). (Note: Further centralization of all paths into a single project config could be a future refinement if needed.)

10. **[x] Workflow Orchestration & Testing:**
    *   [x] Create a main script (`F1/local/run_pipeline.py`) to execute the entire workflow:
        *   Data Extraction -> Data Consolidation -> Feature Engineering -> Model Data Preparation -> Model Training -> Final Model Evaluation.
    *   [x] Ensure all components integrate correctly by testing the `F1/local/run_pipeline.py` script end-to-end using local test data. (Pipeline ran successfully; noted data quality warnings for delta columns and PitIn/OutTime NaNs for future attention. Final evaluation used a potentially stale `best_model.pth` due to no validation set in the test run, which is expected behavior.)
    *   [x] Implement basic logging throughout the pipeline (each script has logging; `run_pipeline.py` also logs to `F1/local/logs/run_pipeline.log`).

11. **[x] Documentation and Cleanup:**
    *   [x] Add README files to key directories explaining their contents and purpose (`F1/local/README.md`, `F1/local/src/README.md`, `F1/local/configs/README.md`, `F1/local/drive/README.md`, and READMEs for `src` subdirectories created).
    *   [ ] Ensure code is well-commented and follows consistent styling. (Code has been commented during development. A full pass for styling consistency can be done if required.)
    *   [ ] Clean up any unused scripts or data from the `Vincenzo/` and `domenicoDL/` explorations once relevant logic is migrated. (This is a manual step for the user.)

---
*Mark steps with `[x]` when completed.*
*Sub-steps can be added as needed.*

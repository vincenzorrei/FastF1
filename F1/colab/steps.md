# Steps for Google Colab Development and Execution (F1/colab/)

This file tracks the defined and completed steps for adapting and running the F1 tyre prediction project on Google Colab.

## Phase 1: Colab Environment Setup & Google Drive Integration

1.  **[x] Setup Colab Environment:**
    *   [x] Define `requirements.txt` specific to Colab (if different from local, e.g., for GPU libraries).
    *   [x] Create a setup script/notebook cell for installing dependencies in Colab.
2.  **[x] Google Drive Integration:**
    *   [x] Implement robust mounting of Google Drive in Colab notebooks/scripts.
    *   [x] Define standardized paths for accessing data, configurations, models, and logs on Google Drive within the `F1/colab/drive/` mirrored structure.
    *   [x] Ensure `F1/colab/drive/` subdirectories (e.g., `raw_data`, `processed_data`, `feature_engineered_data`, `model_input`, `models`, `artifacts`, `logs`) are correctly mapped and accessible.

## Phase 2: Data Extraction for Colab

1.  **[x] Adapt Data Extraction Script for Colab:**
    *   [x] Review and adapt `F1/local/src/data_extraction/fetch_fastf1_data.py` to create/finalize `F1/colab/src/colab_fetch_fastf1_data.py`.
    *   [x] Ensure it uses `F1/colab/configs/colab_data_extraction_config.yaml`.
    *   [x] Verify FastF1 caching works correctly when using Google Drive paths.
    *   [x] Test data extraction for a small number of races, saving to the designated Google Drive `raw_data` folder.
    *   [x] Ensure the `data_download_log.csv` is correctly updated on Google Drive.
2.  **[x] Create/Update Colab Data Extraction Notebook:**
    *   [x] Develop/Update `F1/colab/notebooks/01_Data_Extraction_Colab.ipynb` to:
        *   [x] Mount Google Drive.
        *   [x] Install dependencies.
        *   [x] Run the `colab_fetch_fastf1_data.py` script.
        *   [x] Display logs and outputs.

## Phase 3: Data Processing & Consolidation for Colab

1.  **[x] Adapt Data Consolidation Script for Colab:**
    *   [x] Adapt `F1/local/src/data_processing/consolidate_data.py` to create `F1/colab/src/colab_consolidate_data.py` (or a similar name).
    *   [x] Configure input from Google Drive `raw_data` and output to Google Drive `processed_data`.
    *   [x] Ensure logging works correctly to Google Drive `logs`.
    *   [x] Test consolidation with data extracted in the previous step.
2.  **[x] Create Colab Data Consolidation Notebook:**
    *   [x] Develop `F1/colab/notebooks/02_Data_Consolidation_Colab.ipynb` to orchestrate the consolidation step.

## Phase 4: Feature Engineering for Colab

1.  **[x] Adapt Feature Engineering Scripts for Colab:**
    *   [x] Adapt `F1/local/src/feature_engineering/feature_engineering.py` and `F1/local/src/feature_engineering/create_features_pipeline.py` for Colab.
    *   [x] Ensure scripts read from Google Drive `processed_data` and write to Google Drive `feature_engineered_data` and `artifacts`.
    *   [ ] Test feature engineering.
2.  **[x] Create Colab Feature Engineering Notebook:**
    *   [x] Develop `F1/colab/notebooks/03_Feature_Engineering_Colab.ipynb`.

## Phase 5: Data Preprocessing for Model for Colab

1.  **[ ] Adapt Preprocessing Scripts for Colab:**
    *   [ ] Adapt `F1/local/src/preprocessing/prepare_model_data.py` for Colab.
    *   [ ] Ensure scripts read from Google Drive `feature_engineered_data` and write sequences to Google Drive `model_input`.
    *   [ ] Test preprocessing.
2.  **[ ] Create Colab Data Preprocessing Notebook:**
    *   [ ] Develop `F1/colab/notebooks/04_Data_Preprocessing_Colab.ipynb`.

## Phase 6: Modeling (Training & Evaluation) on Colab

1.  **[ ] Adapt Model Definition and Configuration for Colab:**
    *   [ ] Ensure `F1/local/src/modeling/model_def.py` is compatible or adapt as `F1/colab/src/modeling/colab_model_def.py`.
    *   [ ] Adapt `F1/local/configs/model_config.yaml` for Colab, potentially creating `F1/colab/configs/colab_model_config.yaml` if parameters differ (e.g., batch sizes for GPU).
2.  **[ ] Adapt Model Training Scripts for Colab:**
    *   [ ] Adapt `F1/local/src/modeling/data_loaders.py`, `F1/local/src/modeling/training_utils.py`, and `F1/local/src/modeling/train_model.py` for Colab.
    *   [ ] Ensure data is loaded from Google Drive `model_input`.
    *   [ ] Implement GPU utilization if available.
    *   [ ] Save trained models, checkpoints, and logs to Google Drive `models` and `logs`.
    *   [ ] Test training for a few epochs.
3.  **[ ] Adapt Model Evaluation Scripts for Colab:**
    *   [ ] Adapt `F1/local/src/modeling/evaluate.py` for Colab.
    *   [ ] Ensure it loads models and data from Google Drive.
    *   [ ] Save evaluation reports and plots to Google Drive `artifacts` or `reports`.
    *   [ ] Test evaluation.
4.  **[ ] Create Colab Model Training & Evaluation Notebook:**
    *   [ ] Develop `F1/colab/notebooks/05_Model_Training_Evaluation_Colab.ipynb`.

## Phase 7: Full Pipeline Orchestration and Documentation for Colab

1.  **[ ] Create Colab Full Pipeline Notebook/Script:**
    *   [ ] Develop a master notebook (e.g., `F1/colab/notebooks/00_Run_Full_Pipeline_Colab.ipynb`) or script to execute the entire workflow on Colab.
    *   [ ] Ensure seamless execution of all adapted scripts/notebooks.
2.  **[ ] Documentation for Colab:**
    *   [ ] Create/Update `F1/colab/README.md` explaining the Colab setup, Google Drive structure, and how to run the notebooks/pipeline.
    *   [ ] Ensure individual Colab notebooks are well-documented.

---
*Mark steps with `[ ]` (to do), `[/]` (in progress), or `[x]` (completed).*
*Sub-steps can be added as needed.*

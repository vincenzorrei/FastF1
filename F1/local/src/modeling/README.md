# F1/local/src/modeling - Modeling Module

This module encompasses all aspects of the RNN model for F1 tyre strategy prediction, including its definition, data loading, training procedures, and evaluation.

## Key Components

-   **`model_def.py`**:
    -   Defines the core neural network architecture.
    -   `LSTMTirePredictor`: The main PyTorch `nn.Module` class for the RNN model. It typically consists of LSTM layers followed by separate prediction heads for the two primary tasks:
        1.  Predicting whether a pit stop (tire change) will occur on the next lap (binary classification).
        2.  Predicting the tyre compound that will be fitted if a pit stop occurs (multi-class classification).
    -   `CombinedLoss`: A custom loss function that combines the losses from the two tasks (e.g., `BCEWithLogitsLoss` for tire change and `CrossEntropyLoss` for tire type), potentially weighted. It handles class imbalance for the tire change task using `pos_weight`.
    -   `get_model_hyperparameters()`: A utility function to load and derive necessary hyperparameters for model instantiation (e.g., `input_size` from `feature_columns.joblib`, `num_compounds` from encoders, LSTM hidden sizes from `model_config.yaml`).

-   **`data_loaders.py`**:
    -   Responsible for loading the preprocessed sequence data and preparing it for PyTorch.
    -   `F1SequenceDataset`: A custom PyTorch `Dataset` class that loads the `.npy` sequence files (X features, y_change targets, y_type targets) from `F1/local/drive/model_input/`. It handles data augmentation (e.g., oversampling positive class for tire change) and provides individual samples.
    -   `create_dataloaders()`: A factory function that creates PyTorch `DataLoader` instances for training, validation, and test sets using `F1SequenceDataset`. It manages batching, shuffling (for training), and optional weighted sampling to address class imbalance.

-   **`training_utils.py`**:
    -   Provides a suite of utilities to facilitate the model training process.
    -   `EarlyStopping`: Monitors a validation metric (e.g., F1-score) and stops training if it doesn't improve for a specified number of epochs, preventing overfitting. Can restore best model weights.
    -   `TrainingMetricsTracker`: A class to accumulate predictions and targets during an epoch and compute various classification metrics (F1, precision, recall, ROC AUC, PR AUC, confusion matrix components).
    -   `find_optimal_threshold_on_pr_curve()`: A function to determine an optimal decision threshold for the binary tire change prediction task by analyzing the precision-recall curve, potentially targeting a specific recall level.
    -   `ModelTrainer`: The main class orchestrating the training loop. It handles:
        -   Epoch iteration.
        -   Forward and backward passes.
        -   Optimizer steps.
        -   Metric calculation using `TrainingMetricsTracker`.
        -   Learning rate scheduling.
        -   Logging to console and TensorBoard.
        -   Model checkpointing (saving best model and periodic checkpoints).
        -   Early stopping integration.
    -   `create_trainer_components()`: A helper function to initialize optimizer, scheduler, and loss function details based on the configuration.

-   **`train_model.py`**:
    -   The main executable script for training the model.
    -   **Functionality**:
        -   Loads configurations from `F1/local/configs/model_config.yaml`.
        -   Initializes the `LSTMTirePredictor` model, data loaders, optimizer, scheduler, and loss function using components from `model_def.py`, `data_loaders.py`, and `training_utils.py`.
        -   Instantiates and runs the `ModelTrainer` to perform the training.
        -   Saves the final trained model, training history, and logs.
    -   **Execution**: Called as the fifth step in `F1/local/run_pipeline.py`.

-   **`evaluate.py`**:
    -   Contains tools for evaluating a trained model.
    -   `ModelEvaluator`: A class to perform comprehensive evaluation on a dataset (typically the test set). It calculates metrics for both primary and secondary tasks.
    -   `plot_evaluation_graphics()`: Generates plots like ROC curve, Precision-Recall curve, confusion matrix, and prediction distributions.
    -   `generate_and_save_evaluation_report()`: Orchestrates the evaluation process, calls plotting functions, and saves a JSON report with all metrics and results to `F1/local/drive/evaluation_reports/`.
    -   **Execution**: The `run_final_evaluation` function in `F1/local/run_pipeline.py` uses these components to evaluate the best model from the training phase.

## Input

-   Model configuration: `F1/local/configs/model_config.yaml`.
-   Preprocessed sequence data: `.npy` files in `F1/local/drive/model_input/`.
-   Artifacts (for `input_size`, `num_compounds`): `F1/local/drive/artifacts/`.

## Output

-   **Trained Models**: Checkpoints saved in `F1/local/drive/models/` (e.g., `best_model.pth`).
-   **Training Logs**: TensorBoard logs and JSON summaries in `F1/local/drive/logs/training_logs/`.
-   **Evaluation Reports**: JSON reports and plots in `F1/local/drive/evaluation_reports/`.

## Purpose

This module forms the core of the machine learning pipeline, handling everything from defining the neural network to training it, tracking its performance, and evaluating its predictions on unseen data.

"""
Main Training Script for F1 Tyre Strategy RNN Model
===================================================

This script orchestrates the complete training pipeline for the
LSTMTirePredictor model. It handles:
- Configuration loading.
- Data loading and preparation.
- Model initialization.
- Training loop execution with utilities for metrics, early stopping, checkpointing.
- Saving of the trained model and training summary.
"""

import torch
import logging
from pathlib import Path
import yaml
import numpy as np
import sys
from typing import Optional # Import Optional

# Ensure src directory is in path for module imports
try:
    current_script_path = Path(__file__).resolve()
    project_root_f1_local = current_script_path.parent.parent.parent # Should be F1/local
    
    # Add F1/local/src to sys.path
    src_path = project_root_f1_local / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # Add F1/local to sys.path for imports like from modeling.model_def
    if str(project_root_f1_local) not in sys.path:
         sys.path.insert(0, str(project_root_f1_local))

    from modeling.model_def import LSTMTirePredictor, get_model_hyperparameters
    from modeling.data_loaders import create_dataloaders, F1SequenceDataset # F1SequenceDataset for pos_weight calc
    from modeling.training_utils import ModelTrainer, create_trainer_components
except ImportError as e:
    print(f"Error during initial imports: {e}")
    print(f"Ensure PYTHONPATH or sys.path includes the project's src directory.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)


# --- Configuration ---
CONFIG_FILE_PATH = project_root_f1_local / "configs/model_config.yaml"
# Paths for data are relative to F1/local/ as defined in config or derived
MODEL_INPUT_DIR = project_root_f1_local / "drive/model_input" # From prepare_model_data.py
ARTIFACTS_DIR = project_root_f1_local / "drive/artifacts"     # From feature_engineering


# --- Logging Setup ---
# Basic logging config, can be expanded (e.g., file handler)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Loads the main configuration YAML file."""
    logger.info(f"Loading configuration from: {config_path}")
    if not config_path.exists():
        logger.error(f"Configuration file not found at {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded successfully.")
    return config

def determine_device(config_device: str) -> str:
    """Determines the device for training (cuda or cpu)."""
    if config_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config_device
    logger.info(f"Using device: {device}")
    return device

def calculate_pos_weight(train_dataset: F1SequenceDataset) -> Optional[float]:
    """Calculates pos_weight for BCEWithLogitsLoss from training dataset."""
    if len(train_dataset) == 0:
        logger.warning("Training dataset is empty, cannot calculate pos_weight.")
        return None
    
    targets_change = train_dataset.y_change_augmented if train_dataset.augment_positive else train_dataset.y_change_original
    if len(targets_change) == 0:
        logger.warning("No targets in training dataset, cannot calculate pos_weight.")
        return None

    num_positives = np.sum(targets_change == 1)
    num_negatives = np.sum(targets_change == 0)

    if num_positives == 0:
        logger.warning("No positive samples in training data for pos_weight calculation. Defaulting to None (or 1.0).")
        return None # Or 1.0, depending on BCEWithLogitsLoss default behavior for None
    
    pos_weight = num_negatives / num_positives
    logger.info(f"Calculated pos_weight for tire_change_loss: {pos_weight:.2f} (Negatives: {num_negatives}, Positives: {num_positives})")
    return float(pos_weight)


def main():
    logger.info("====== Starting F1 Tyre Strategy Model Training Pipeline ======")

    # 1. Load Configuration
    config = load_config(CONFIG_FILE_PATH)
    model_cfg = config.get('model', {})
    train_cfg = config.get('training', {})
    data_cfg = config.get('data', {}) # Contains relative paths for log/checkpoint dirs

    # 2. Determine Device
    device = determine_device(train_cfg.get('device', 'auto'))

    # 3. Load Model Hyperparameters (input_size, num_compounds derived from artifacts)
    # artifacts_dir_abs = project_root_f1_local / data_cfg.get('artifacts_dir', 'drive/artifacts')
    try:
        model_hyperparams = get_model_hyperparameters(
            config_path=str(CONFIG_FILE_PATH), # Pass path to full config
            artifacts_dir=str(ARTIFACTS_DIR),
            device=device
        )
    except Exception as e:
        logger.error(f"Failed to get model hyperparameters: {e}", exc_info=True)
        return
    logger.info(f"Model hyperparameters: {model_hyperparams}")

    # 4. Initialize Model
    model = LSTMTirePredictor(**model_hyperparams)
    logger.info(f"Model '{type(model).__name__}' initialized on {device}.")
    logger.info(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 5. Create DataLoaders
    # model_input_dir_abs = project_root_f1_local / data_cfg.get('model_input_dir', 'drive/model_input')
    dataloaders = create_dataloaders(
        model_input_dir=str(MODEL_INPUT_DIR),
        batch_size=train_cfg.get('batch_size', 64),
        num_workers=train_cfg.get('num_workers', 0 if device == 'cpu' else 2),
        pin_memory=(device == 'cuda'),
        augment_train_positive_samples=train_cfg.get('augment_train_positive_samples', True),
        augmentation_factor=train_cfg.get('augmentation_factor', 2),
        use_weighted_sampler_for_train=train_cfg.get('use_weighted_sampler_for_train', True)
    )
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    # test_loader = dataloaders['test'] # For final evaluation after training

    if len(train_loader.dataset) == 0:
        logger.error("Training dataset is empty. Cannot proceed with training.")
        return
    if val_loader and len(val_loader.dataset) == 0:
        logger.warning("Validation dataset is empty. Training will proceed without validation.")
        val_loader = None # Explicitly set to None if empty

    # 6. Calculate pos_weight if not set or set to auto-calculate indicator
    loss_params_cfg = model_cfg.get('loss', {})
    pos_weight = loss_params_cfg.get('pos_weight_tire_change')
    if pos_weight is None or pos_weight <= 0: # Assuming <=0 means auto-calculate
        logger.info("pos_weight_tire_change not specified or invalid in config, calculating from training data.")
        pos_weight = calculate_pos_weight(train_loader.dataset)
        if pos_weight is None: # Fallback if calculation failed
            logger.warning("Failed to calculate pos_weight, using default (None for BCEWithLogitsLoss).")
    loss_params_cfg['pos_weight_tire_change'] = pos_weight # Update config dict for trainer components

    # 7. Create Trainer Components (Optimizer, Scheduler, Loss details, Log/Checkpoint Dirs)
    # The create_trainer_components expects the full config dict
    optimizer, scheduler, loss_fn_details, log_dir_abs, checkpoint_dir_abs = create_trainer_components(
        model=model,
        config=config, # Pass the full config
        device=device
    )
    logger.info(f"Optimizer: {type(optimizer).__name__}, LR: {optimizer.defaults['lr']}")
    if scheduler:
        logger.info(f"Scheduler: {type(scheduler).__name__}")
    logger.info(f"Loss function details: {loss_fn_details}")
    logger.info(f"Log directory: {log_dir_abs}")
    logger.info(f"Checkpoint directory: {checkpoint_dir_abs}")


    # 8. Initialize ModelTrainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader, # Can be None if val_loader was empty
        optimizer=optimizer,
        loss_fn_details=loss_fn_details, # Pass the dict
        device=device,
        log_dir=str(log_dir_abs),
        checkpoint_dir=str(checkpoint_dir_abs),
        config_training_params=train_cfg # Pass training sub-config for params like target_recall
    )

    # 9. Start Training
    logger.info("====== Starting Model Training Loop ======")
    training_history = trainer.train_model(
        num_epochs=train_cfg.get('epochs', 50),
        scheduler=scheduler,
        save_freq=train_cfg.get('save_frequency', 5)
    )
    
    logger.info("====== Training Pipeline Finished ======")
    logger.info(f"Best validation F1-score: {trainer.best_val_f1:.4f} at threshold {trainer.best_threshold_val:.4f}")
    logger.info(f"Model checkpoints and logs saved in: {checkpoint_dir_abs} and {log_dir_abs}")

    # (Optional) Add final evaluation on test set here if desired
    # test_evaluation_report = evaluate_model_on_test_set(model, test_loader, trainer.best_threshold_val, device)
    # logger.info(f"Final Test Set Evaluation: {test_evaluation_report}")

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as fnf_err:
        logger.error(f"File not found error during training setup: {fnf_err}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during training: {e}", exc_info=True)
        sys.exit(1)

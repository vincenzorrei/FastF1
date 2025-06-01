"""
Main Orchestration Script for F1 Tyre Strategy Project (Local)
==============================================================

This script runs the entire local data processing and model training pipeline:
1. Data Extraction: Fetches raw data using FastF1.
2. Data Consolidation: Merges and cleans raw data.
3. Feature Engineering: Creates features for modeling.
4. Model Data Preparation: Prepares sequences for RNN.
5. Model Training: Trains the RNN model.
6. Model Evaluation (on test set): Evaluates the trained model.

Each step is executed sequentially. Ensure configurations are set correctly
in the `F1/local/configs/` directory.
"""

import sys
import logging
from pathlib import Path
import importlib # For more robust module loading if needed
import yaml # Added import
import torch # Added import for torch.load and torch.cuda.is_available

# --- Setup Project Root and Paths ---
# Assuming this script is in F1/local/
# project_root = Path(__file__).resolve().parent # F1/local/
# src_path = project_root / "src"
# For direct execution from any CWD, it's better to define relative to a known structure
# If run as `python F1/local/run_pipeline.py` from project root `FASTF1/`
# then F1/local/ is the path.
# If run as `python run_pipeline.py` from `FASTF1/F1/local/`
# then current dir is fine.

# Let's assume it's run from project root `FASTF1/`
# So, modules are in `F1.local.src.*`
# Or, adjust sys.path to add `F1/local/src` and `F1/local`

try:
    # This structure assumes the script is run from the project root (FASTF1)
    # or that F1/local is in PYTHONPATH.
    # For robustness, let's add F1/local and F1/local/src to sys.path
    # This script itself is in F1/local/
    
    current_script_dir = Path(__file__).resolve().parent # This is F1/local
    src_dir = current_script_dir / "src"

    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if str(current_script_dir) not in sys.path: # To allow `from modeling import ...`
         sys.path.insert(0, str(current_script_dir))


    # Import main functions from each module
    from data_extraction import fetch_fastf1_data
    from data_processing import consolidate_data
    from feature_engineering import create_features_pipeline
    from preprocessing import prepare_model_data
    from modeling import train_model
    from modeling import evaluate # For final evaluation
    from modeling.model_def import LSTMTirePredictor, get_model_hyperparameters # To load model and params
    from modeling.data_loaders import create_dataloaders # To load test data

except ImportError as e:
    print(f"Error importing pipeline modules: {e}")
    print("Please ensure this script is run from the 'F1/local/' directory,")
    print("or that the F1/local/src directory is in your PYTHONPATH.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(current_script_dir / "logs/run_pipeline.log", mode='a') # Log to a file in F1/local/logs
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration Paths ---
CONFIG_FILE_PATH = current_script_dir / "configs/model_config.yaml"
ARTIFACTS_DIR = current_script_dir / "drive/artifacts"
BEST_MODEL_PATH = current_script_dir / "drive/models/best_model.pth" # Default path where best model is saved
EVALUATION_REPORT_DIR = current_script_dir / "drive/evaluation_reports/pipeline_test_set"


def run_final_evaluation():
    """Loads the best trained model and evaluates it on the test set."""
    logger.info("====== Starting Final Model Evaluation on Test Set ======")
    
    # 1. Load Configuration
    with open(CONFIG_FILE_PATH, 'r') as f:
        config = yaml.safe_load(f)
    train_cfg = config.get('training', {})
    
    # 2. Determine Device
    device = "cuda" if torch.cuda.is_available() and train_cfg.get('device', 'auto') != 'cpu' else "cpu"
    logger.info(f"Using device: {device} for final evaluation.")

    # 3. Load Model Hyperparameters
    try:
        model_hyperparams = get_model_hyperparameters(
            config_path=str(CONFIG_FILE_PATH),
            artifacts_dir=str(ARTIFACTS_DIR),
            device=device
        )
    except Exception as e:
        logger.error(f"Failed to get model hyperparameters for evaluation: {e}", exc_info=True)
        return

    # 4. Initialize Model
    model = LSTMTirePredictor(**model_hyperparams)
    
    # 5. Load Best Trained Weights
    if not BEST_MODEL_PATH.exists():
        logger.error(f"Best model checkpoint not found at {BEST_MODEL_PATH}. Cannot evaluate.")
        return
    
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    logger.info(f"Loaded best model weights from {BEST_MODEL_PATH} (Epoch {checkpoint.get('epoch', 'N/A')}, Val F1: {checkpoint.get('best_val_f1',0.0):.4f})")
    
    decision_threshold = checkpoint.get('best_threshold_val', 0.5) # Use threshold from training
    logger.info(f"Using decision threshold from training: {decision_threshold:.4f}")

    # 6. Create Test DataLoader
    # model_input_dir_abs = current_script_dir / config.get('data', {}).get('model_input_dir', 'drive/model_input')
    model_input_dir_abs = current_script_dir / "drive/model_input"

    dataloaders = create_dataloaders(
        model_input_dir=str(model_input_dir_abs),
        batch_size=train_cfg.get('batch_size', 64), # Use same batch size as training for consistency
        num_workers=0, # Typically 0 for test set unless very large
        pin_memory=(device == 'cuda'),
        augment_train_positive_samples=False, # No augmentation for test
        use_weighted_sampler_for_train=False  # No sampling for test
    )
    test_loader = dataloaders.get('test')

    if not test_loader or len(test_loader.dataset) == 0:
        logger.error("Test data loader is empty or could not be created. Cannot evaluate.")
        return

    # 7. Generate and Save Evaluation Report
    EVALUATION_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = evaluate.generate_and_save_evaluation_report(
        model=model,
        data_loader=test_loader,
        device=device,
        decision_threshold=decision_threshold,
        report_dir=EVALUATION_REPORT_DIR,
        split_name="FinalTestSet"
    )
    logger.info(f"Final evaluation report generated: {report}")
    logger.info("====== Final Model Evaluation Completed ======")


def main_orchestration():
    """Runs the full pipeline."""
    logger.info("############################################################")
    logger.info("### STARTING F1 LOCAL PIPELINE ORCHESTRATION ###")
    logger.info("############################################################")

    pipeline_steps = [
        ("Data Extraction", fetch_fastf1_data.main),
        ("Data Consolidation", consolidate_data.main),
        ("Feature Engineering", create_features_pipeline.main_pipeline),
        ("Model Data Preparation", prepare_model_data.main_preparation_pipeline),
        ("Model Training", train_model.main),
        ("Final Model Evaluation", run_final_evaluation) 
    ]

    for step_name, step_function in pipeline_steps:
        logger.info(f"\n--- Running Step: {step_name} ---")
        try:
            step_function()
            logger.info(f"--- Step: {step_name} COMPLETED SUCCESSFULLY ---\n")
        except Exception as e:
            logger.error(f"--- Step: {step_name} FAILED ---", exc_info=True)
            logger.error("Stopping pipeline due to error.")
            sys.exit(1) # Exit if a step fails

    logger.info("############################################################")
    logger.info("### F1 LOCAL PIPELINE ORCHESTRATION COMPLETED SUCCESSFULLY ###")
    logger.info("############################################################")

if __name__ == "__main__":
    # Ensure logs directory exists for run_pipeline.log
    (current_script_dir / "logs").mkdir(parents=True, exist_ok=True)
    main_orchestration()

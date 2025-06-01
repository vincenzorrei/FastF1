"""
Pipeline Script for Feature Engineering (Colab)
===============================================

This script orchestrates the feature engineering process in the Colab environment:
1. Loads the consolidated dataset from Google Drive.
2. Applies feature engineering functions defined in 'colab_feature_engineering.py'.
3. Saves the feature-engineered dataset to Google Drive.
4. Saves feature engineering artifacts (encoders, scalers, imputer values) to Google Drive.
"""
import pandas as pd
import logging
from pathlib import Path
import sys
import yaml # For loading colab_path_config

# --- Colab Path Configuration ---
def load_colab_paths():
    """Loads and resolves paths from F1/colab/configs/colab_path_config.yaml."""
    # Assuming this script is in F1/colab/src/feature_engineering/
    path_config_file = Path(__file__).parent.parent.parent / "configs" / "colab_path_config.yaml"
    if not path_config_file.exists():
        print(f"ERROR: Colab path config file not found at {path_config_file}")
        return None
    with open(path_config_file, 'r') as f:
        raw_config = yaml.safe_load(f)

    base_drive_path_str = raw_config.get('base_project_drive_path')
    if not base_drive_path_str or "path/to/your/FASTF1" in base_drive_path_str:
        print("ERROR: 'base_project_drive_path' in 'colab_path_config.yaml' is not configured.")
        return None
    
    base_drive_path = Path(base_drive_path_str)
    colab_root_str = raw_config.get('colab_root', 'F1/colab')

    resolved_paths = {}
    for key, value in raw_config.items():
        if isinstance(value, str):
            formatted_value = value.format(
                base_project_drive_path=str(base_drive_path),
                colab_root=colab_root_str,
                drive_simulation_root=str(base_drive_path / colab_root_str / "drive"),
                logs_directory=str(base_drive_path / colab_root_str / "drive" / "logs"),
                colab_configs_dir=str(base_drive_path / colab_root_str / "configs")
            )
            resolved_paths[key] = Path(formatted_value)
        else:
            resolved_paths[key] = value
    return resolved_paths

COLAB_PATHS = load_colab_paths()

# Dynamically add the 'src' directory (F1/colab/src) to sys.path
# to allow importing colab_feature_engineering
if COLAB_PATHS and COLAB_PATHS.get('colab_src_dir'):
    src_dir = COLAB_PATHS['colab_src_dir']
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir.parent)) # Add F1/colab/ to path for `from src...`
    # Now import should work if colab_feature_engineering.py is in F1/colab/src/feature_engineering/
    from feature_engineering import colab_feature_engineering as fe
else:
    print("CRITICAL ERROR: COLAB_PATHS not loaded or 'colab_src_dir' not defined. Cannot import feature_engineering module.")
    # Fallback for local testing if COLAB_PATHS fails but script needs to be parsable
    try:
        # This assumes a specific local structure if the Colab paths fail, for basic parsing/testing
        current_path = Path(__file__).resolve()
        project_root = current_path.parent.parent.parent 
        sys.path.insert(0, str(project_root))
        from src.feature_engineering import colab_feature_engineering as fe # type: ignore
        print("WARNING: Using fallback import for colab_feature_engineering due to COLAB_PATHS issue.")
    except ImportError:
         print("Fallback import for colab_feature_engineering also failed.")
         sys.exit(1)


# Configure logging
if COLAB_PATHS and COLAB_PATHS.get('feature_engineering_log_file'):
    LOG_FILE = COLAB_PATHS['feature_engineering_log_file']
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
else:
    fallback_log_dir = Path(__file__).resolve().parent.parent.parent / "logs_colab_fallback"
    fallback_log_dir.mkdir(parents=True, exist_ok=True)
    LOG_FILE = fallback_log_dir / "colab_create_features_pipeline_fallback.log"
    print(f"WARNING: Using fallback log file: {LOG_FILE}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main_pipeline():
    """
    Main function to run the feature engineering pipeline for Colab.
    """
    logger.info("=== STARTING FEATURE ENGINEERING PIPELINE (COLAB) ===")

    if not COLAB_PATHS:
        logger.error("COLAB_PATHS not loaded. Cannot determine file paths. Exiting.")
        return

    # Define paths using COLAB_PATHS
    consolidated_data_path = COLAB_PATHS.get('processed_data_directory') / "dataset.parquet"
    feature_engineered_data_dir = COLAB_PATHS.get('feature_engineered_data_directory')
    artifacts_dir = COLAB_PATHS.get('artifacts_directory')
    
    if not all([consolidated_data_path, feature_engineered_data_dir, artifacts_dir]):
        logger.error("One or more critical paths are missing from colab_path_config.yaml. Exiting.")
        return

    feature_engineered_data_output_path = feature_engineered_data_dir / "featured_dataset.parquet"
    encoders_path = artifacts_dir / "colab_encoders.pkl"
    scaler_path = artifacts_dir / "colab_scaler.pkl"
    imputer_values_path = artifacts_dir / "colab_imputer_values.json"

    # Create output directories if they don't exist
    feature_engineered_data_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured output directories exist: {feature_engineered_data_dir}, {artifacts_dir}")

    # 1. Load consolidated dataset
    logger.info(f"Loading consolidated dataset from: {consolidated_data_path}")
    if not consolidated_data_path.exists():
        logger.error(f"Consolidated dataset not found at {consolidated_data_path}. Exiting.")
        return
    
    try:
        df_consolidated = pd.read_parquet(consolidated_data_path)
        logger.info(f"Successfully loaded consolidated dataset: {len(df_consolidated):,} rows, {len(df_consolidated.columns)} columns.")
    except Exception as e:
        logger.error(f"Error loading consolidated dataset: {e}")
        return

    df_processed = df_consolidated.copy()
    training_mode = True # Assume training mode to generate artifacts

    logger.info("Step 1: Creating target variables...")
    df_processed = fe.create_target_variables(df_processed)
    
    logger.info("Step 2: Creating temporal features...")
    df_processed = fe.create_temporal_features(df_processed)
    
    logger.info("Step 3: Creating performance features...")
    df_processed = fe.create_performance_features(df_processed)
    
    logger.info("Step 4: Creating weather features...")
    df_processed = fe.create_weather_features(df_processed)
    
    logger.info("Step 5: Creating domain knowledge features...")
    df_processed = fe.create_domain_knowledge_features(df_processed)
    
    logger.info("Step 6: Handling missing values...")
    df_processed = fe.handle_missing_values(df_processed, training_mode=training_mode, imputer_values_path=imputer_values_path)
    
    logger.info("Step 7: Handling categorical variables...")
    df_processed, _ = fe.handle_categorical_variables(df_processed, training_mode=training_mode, encoders_path=encoders_path)
    
    logger.info("Step 8: Normalizing features...")
    df_processed, _ = fe.normalize_features(df_processed, training_mode=training_mode, scaler_path=scaler_path)

    logger.info(f"Saving feature-engineered dataset to: {feature_engineered_data_output_path}")
    try:
        df_processed.to_parquet(feature_engineered_data_output_path, index=False)
        logger.info(f"Successfully saved feature-engineered dataset: {len(df_processed):,} rows, {len(df_processed.columns)} columns.")
    except Exception as e:
        logger.error(f"Error saving feature-engineered dataset: {e}")
        return

    logger.info("Artifacts (encoders, scaler, imputer values) saved during the process to Google Drive.")
    logger.info("=== FEATURE ENGINEERING PIPELINE (COLAB) COMPLETED SUCCESSFULLY ===")
    logger.info(f"Final DataFrame info:")
    df_processed.info(verbose=True, show_counts=True)

if __name__ == "__main__":
    if COLAB_PATHS:
        main_pipeline()
    else:
        logger.critical("COLAB_PATHS not loaded at script start. Ensure F1/colab/configs/colab_path_config.yaml is correct. Exiting.")

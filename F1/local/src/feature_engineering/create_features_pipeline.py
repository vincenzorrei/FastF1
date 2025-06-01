"""
Pipeline Script for Feature Engineering
=======================================

This script orchestrates the feature engineering process:
1. Loads the consolidated dataset.
2. Applies feature engineering functions defined in 'feature_engineering.py'.
3. Saves the feature-engineered dataset.
4. Saves feature engineering artifacts (encoders, scalers, imputer values).

Autore: Cline (AI Software Engineer)
Data: 2025-05-31
"""
import pandas as pd
import logging
from pathlib import Path
import sys

# Add src directory to Python path to allow direct import of feature_engineering
# This assumes the script is run from the F1/local/ directory or the project root.
# Adjust if running from F1/local/src/feature_engineering/
try:
    # Attempt to add the parent of 'src' to sys.path if 'src' is in the current path parts
    # This makes the script more robust to where it's called from within the project
    current_path = Path(__file__).resolve() # Get the absolute path of the current script
    project_root = current_path.parent.parent.parent # F1/local/
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(project_root)) # Add F1/local to path
        sys.path.insert(0, str(src_path.parent)) # Add F1 to path for `from src.feature_engineering...`
        
    # from feature_engineering import feature_engineering as fe # if script in F1/local/src/feature_engineering
    from src.feature_engineering import feature_engineering as fe # if script in F1/local/ or F1/local/src
except ImportError as e:
    print(f"Error importing feature_engineering module: {e}")
    print("Ensure the script is run from a location where 'src.feature_engineering' can be resolved,")
    print("or adjust the sys.path modification. Current sys.path:", sys.path)
    sys.exit(1)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(), # Log to console
        # Consider adding a FileHandler if persistent logs are needed for this script
    ]
)
logger = logging.getLogger(__name__)

# Define paths (relative to F1/local/ for consistency)
BASE_PATH = Path("F1/local/") # Assuming script might be run from project root or F1/local
CONSOLIDATED_DATA_PATH = BASE_PATH / "drive/processed_data/dataset.parquet"
FEATURE_ENGINEERED_DATA_DIR = BASE_PATH / "drive/feature_engineered_data"
ARTIFACTS_DIR = BASE_PATH / "drive/artifacts"
FEATURE_ENGINEERED_DATA_OUTPUT_PATH = FEATURE_ENGINEERED_DATA_DIR / "featured_dataset.parquet"

# Artifact paths
ENCODERS_PATH = ARTIFACTS_DIR / "encoders.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
IMPUTER_VALUES_PATH = ARTIFACTS_DIR / "imputer_values.json"

def main_pipeline():
    """
    Main function to run the feature engineering pipeline.
    """
    logger.info("=== STARTING FEATURE ENGINEERING PIPELINE ===")

    # Create output directories if they don't exist
    FEATURE_ENGINEERED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured output directories exist: {FEATURE_ENGINEERED_DATA_DIR}, {ARTIFACTS_DIR}")

    # 1. Load consolidated dataset
    logger.info(f"Loading consolidated dataset from: {CONSOLIDATED_DATA_PATH}")
    if not CONSOLIDATED_DATA_PATH.exists():
        logger.error(f"Consolidated dataset not found at {CONSOLIDATED_DATA_PATH}. Exiting.")
        return
    
    try:
        df_consolidated = pd.read_parquet(CONSOLIDATED_DATA_PATH)
        logger.info(f"Successfully loaded consolidated dataset: {len(df_consolidated):,} rows, {len(df_consolidated.columns)} columns.")
    except Exception as e:
        logger.error(f"Error loading consolidated dataset: {e}")
        return

    df_processed = df_consolidated.copy()

    # --- Apply Feature Engineering Steps ---
    # For this pipeline, we assume we are in "training_mode" to generate artifacts.
    # In a real scenario, you might have a config to switch between training/inference.
    training_mode = True 

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
    df_processed = fe.handle_missing_values(df_processed, training_mode=training_mode, imputer_values_path=str(IMPUTER_VALUES_PATH))
    
    logger.info("Step 7: Handling categorical variables...")
    df_processed, _ = fe.handle_categorical_variables(df_processed, training_mode=training_mode, encoders_path=str(ENCODERS_PATH))
    
    logger.info("Step 8: Normalizing features...")
    df_processed, _ = fe.normalize_features(df_processed, training_mode=training_mode, scaler_path=str(SCALER_PATH))

    # --- Save Feature Engineered Data ---
    logger.info(f"Saving feature-engineered dataset to: {FEATURE_ENGINEERED_DATA_OUTPUT_PATH}")
    try:
        df_processed.to_parquet(FEATURE_ENGINEERED_DATA_OUTPUT_PATH, index=False)
        logger.info(f"Successfully saved feature-engineered dataset: {len(df_processed):,} rows, {len(df_processed.columns)} columns.")
    except Exception as e:
        logger.error(f"Error saving feature-engineered dataset: {e}")
        return

    logger.info("Artifacts (encoders, scaler, imputer values) saved during the process.")
    logger.info("=== FEATURE ENGINEERING PIPELINE COMPLETED SUCCESSFULLY ===")
    logger.info(f"Final DataFrame info:")
    df_processed.info(verbose=True, show_counts=True)

if __name__ == "__main__":
    # Adjust base path if running the script directly from its location
    # This is a simple way to handle running from different locations.
    # For robust projects, consider using environment variables or config files for paths.
    script_path = Path(__file__).resolve()
    if "F1/local/src/feature_engineering" in str(script_path):
         # Running from within F1/local/src/feature_engineering
        BASE_PATH = script_path.parent.parent.parent # Should be F1/local
        CONSOLIDATED_DATA_PATH = BASE_PATH / "drive/processed_data/dataset.parquet"
        FEATURE_ENGINEERED_DATA_DIR = BASE_PATH / "drive/feature_engineered_data"
        ARTIFACTS_DIR = BASE_PATH / "drive/artifacts"
        FEATURE_ENGINEERED_DATA_OUTPUT_PATH = FEATURE_ENGINEERED_DATA_DIR / "featured_dataset.parquet"
        ENCODERS_PATH = ARTIFACTS_DIR / "encoders.pkl"
        SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
        IMPUTER_VALUES_PATH = ARTIFACTS_DIR / "imputer_values.json"
        
        # Update sys.path for imports if running directly
        project_root_for_direct_run = BASE_PATH.parent # F1
        if str(project_root_for_direct_run) not in sys.path:
             sys.path.insert(0, str(project_root_for_direct_run))
        from src.feature_engineering import feature_engineering as fe


    main_pipeline()

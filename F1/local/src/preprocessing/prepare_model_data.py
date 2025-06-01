"""
Model Data Preparation Script
=============================

This script prepares the feature-engineered data for RNN model training.
It includes:
1. Loading feature-engineered data and artifacts.
2. Temporal splitting of data into train, validation, and test sets.
3. Creating sequences suitable for RNN input.
4. Saving the processed model inputs (sequences, targets, feature list).

Autore: Cline (AI Software Engineer)
Data: 2025-05-31
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import joblib # For loading .pkl files
import sys
from typing import Tuple, Dict, List

# Add project root to Python path for module imports
try:
    current_path = Path(__file__).resolve()
    project_root = current_path.parent.parent.parent # Should be F1/local/
    src_path = project_root / "src"
    if str(project_root) not in sys.path: # Add F1/local
        sys.path.insert(0, str(project_root))
    if str(src_path.parent) not in sys.path: # Add F1
         sys.path.insert(0, str(src_path.parent))
    # No direct import from feature_engineering needed here, but good practice for structure
except Exception as e:
    print(f"Error adjusting sys.path: {e}")
    # Continue if path adjustment fails, imports might still work if run from correct CWD

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Define paths (relative to F1/local/ for consistency)
BASE_PATH = Path("F1/local/")
FEATURE_ENGINEERED_DATA_PATH = BASE_PATH / "drive/feature_engineered_data/featured_dataset.parquet"
ARTIFACTS_DIR = BASE_PATH / "drive/artifacts"
MODEL_INPUT_DIR = BASE_PATH / "drive/model_input"

ENCODERS_PATH = ARTIFACTS_DIR / "encoders.pkl"
# SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl" # Scaler already applied, not directly needed for sequence creation here
# IMPUTER_VALUES_PATH = ARTIFACTS_DIR / "imputer_values.json" # Imputation already done

# Configuration for sequence creation
SEQUENCE_LENGTH = 10 # Example sequence length, can be configured

def load_data_and_artifacts() -> Tuple[pd.DataFrame, Dict]:
    """Loads the feature-engineered dataset and encoders."""
    logger.info(f"Loading feature-engineered dataset from: {FEATURE_ENGINEERED_DATA_PATH}")
    if not FEATURE_ENGINEERED_DATA_PATH.exists():
        logger.error(f"Feature-engineered dataset not found at {FEATURE_ENGINEERED_DATA_PATH}. Exiting.")
        raise FileNotFoundError(f"Dataset not found: {FEATURE_ENGINEERED_DATA_PATH}")
    
    df = pd.read_parquet(FEATURE_ENGINEERED_DATA_PATH)
    logger.info(f"Successfully loaded feature-engineered dataset: {len(df):,} rows, {len(df.columns)} columns.")

    logger.info(f"Loading encoders from: {ENCODERS_PATH}")
    if not ENCODERS_PATH.exists():
        logger.error(f"Encoders not found at {ENCODERS_PATH}. Exiting.")
        raise FileNotFoundError(f"Encoders not found: {ENCODERS_PATH}")
    
    encoders = joblib.load(ENCODERS_PATH)
    logger.info("Successfully loaded encoders.")
    
    return df, encoders

def temporal_train_val_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset temporally based on 'Year' using a pragmatic approach for limited data.
    - If 1 year: All data for train, empty val/test.
    - If 2 years (Y1, Y2): Train = Y1, Test = Y2, empty Val.
    - If 3+ years (Y1, ..., Yn-2, Yn-1, Yn): Train = Y1...Yn-2, Val = Yn-1, Test = Yn.
    """
    logger.info("Performing pragmatic temporal train-validation-test split...")
    
    available_years = sorted(df['Year'].unique().astype(int)) # Ensure years are int
    logger.info(f"Available years in dataset: {available_years}")

    train_years, val_years, test_years = [], [], []

    if not available_years:
        logger.error("No data available to split.")
        return pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns)

    if len(available_years) == 1:
        train_years = available_years
        logger.info("Only one year of data. Using all for training.")
    elif len(available_years) == 2:
        train_years = [available_years[0]]
        test_years = [available_years[1]]
        logger.info("Two years of data. Using first for train, second for test. No validation set.")
    else: # 3 or more years
        test_years = [available_years[-1]]
        val_years = [available_years[-2]]
        train_years = available_years[:-2]
        logger.info(f"Three or more years. Train: {train_years}, Val: {val_years}, Test: {test_years}")

    train_df = df[df['Year'].isin(train_years)].copy() if train_years else pd.DataFrame(columns=df.columns)
    val_df = df[df['Year'].isin(val_years)].copy() if val_years else pd.DataFrame(columns=df.columns)
    test_df = df[df['Year'].isin(test_years)].copy() if test_years else pd.DataFrame(columns=df.columns)
    
    logger.info("Temporal split completed:")
    logger.info(f"  Train: {len(train_df):,} rows from years {train_years}. Target mean: {train_df['tire_change_next_lap'].mean()*100:.2f}%" if not train_df.empty else "  Train: 0 rows")
    logger.info(f"  Val:   {len(val_df):,} rows from years {val_years}. Target mean: {val_df['tire_change_next_lap'].mean()*100:.2f}%" if not val_df.empty else "  Val: 0 rows")
    logger.info(f"  Test:  {len(test_df):,} rows from years {test_years}. Target mean: {test_df['tire_change_next_lap'].mean()*100:.2f}%" if not test_df.empty else "  Test: 0 rows")
    
    if train_df.empty or (val_df.empty and test_df.empty):
        logger.warning("One or more splits are empty. Check year configuration and data availability.")
        
    return train_df, val_df, test_df

def create_sequences_for_rnn(df: pd.DataFrame, sequence_length: int, encoders: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Creates sequences for RNN input from a DataFrame.
    """
    logger.info(f"Creating RNN sequences with length {sequence_length} for {len(df)} rows...")
    if df.empty:
        logger.warning("Input DataFrame is empty. Returning empty arrays.")
        # Define number of features based on a template or a fixed list if possible
        # For now, returning empty arrays with 0 features if df is empty.
        # This might need adjustment if a fixed feature set is known.
        return np.array([]).reshape(0, sequence_length, 0), np.array([]), np.array([]), []

    sequences = []
    targets_change = []
    targets_type = []

    # Define feature columns for RNN input
    # Exclude IDs, raw categoricals, other targets, and non-numeric/non-input features
    # Encoded categoricals and scaled numericals should be used.
    exclude_cols = [
        'Year', 'DriverID', 'RaceID', 'DriverRaceID', 'GlobalLapID', # IDs
        'GranPrix', 'Location', 'Driver', 'Team', 'Compound', # Raw categoricals
        'compound_strategy_pattern', # High cardinality string
        'tire_change_next_lap', 'next_tire_type', # Targets
        'PitInTime', 'PitOutTime' # Original timedelta columns, if still present
    ]
    # Select columns that are likely numeric (float, int) or already encoded
    # This assumes feature_engineering.py has converted relevant columns to numeric types
    # and created _encoded columns for categoricals.
    
    potential_feature_cols = [col for col in df.columns if col not in exclude_cols]
    feature_cols = []
    for col in potential_feature_cols:
        if df[col].dtype in ['float64', 'int64', 'float32', 'int32', 'int8', 'uint8'] or col.endswith('_encoded') or col.endswith('_freq'):
             # Check for boolean columns that were not converted to int
            if df[col].dtype == 'bool':
                logger.info(f"Converting boolean column {col} to int for RNN features.")
                df[col] = df[col].astype(int)
            feature_cols.append(col)
        else:
            logger.warning(f"Excluding column {col} (dtype: {df[col].dtype}) from RNN features.")
            
    if not feature_cols:
        logger.error("No feature columns selected for RNN. Check data types and exclusions.")
        return np.array([]).reshape(0, sequence_length, 0), np.array([]), np.array([]), []
    
    logger.info(f"Using {len(feature_cols)} features for RNN: {feature_cols}")

    # Group by driver-race to maintain temporal continuity within sequences
    for _, group in df.groupby('DriverRaceID'):
        group_sorted = group.sort_values('LapNumber')
        
        features_values = group_sorted[feature_cols].values.astype(np.float32)
        change_targets_values = group_sorted['tire_change_next_lap'].values
        
        # Encode 'next_tire_type' using the 'Compound' encoder as it shares labels
        compound_encoder = encoders.get('Compound')
        if not compound_encoder:
            logger.error("Compound encoder not found in encoders. Cannot encode 'next_tire_type'.")
            # Fallback or raise error
            type_targets_values = np.full(len(group_sorted), -1) # Placeholder for error
        else:
            # Ensure all values in next_tire_type are known to the encoder
            # Add 'NO_CHANGE' to classes if not present during fit (should be handled in feature_eng)
            unknown_labels = set(group_sorted['next_tire_type'].unique()) - set(compound_encoder.classes_)
            if unknown_labels:
                logger.warning(f"Unknown labels found in 'next_tire_type' for Compound encoder: {unknown_labels}. Mapping to -1 or a specific class.")
                # Simple strategy: map unknown to a specific code (e.g., -1 or len(classes))
                # This requires careful handling in model and evaluation.
                # For now, let transform error or handle as per encoder's unknown_value if set.
                # The current LabelEncoder will error.
                # A robust solution is to ensure 'NO_CHANGE' and other expected values are in classes.
                # The `handle_categorical_variables` in feature_engineering.py should have added 'NO_CHANGE'.
            try:
                type_targets_values = compound_encoder.transform(group_sorted['next_tire_type'].astype(str))
            except ValueError as e:
                logger.error(f"Error transforming 'next_tire_type' for group: {e}. Filling with -1.")
                type_targets_values = np.full(len(group_sorted), -1)


        for i in range(len(features_values) - sequence_length + 1):
            seq_features = features_values[i : i + sequence_length]
            # Target is for the *last* timestep in the sequence
            seq_change_target = change_targets_values[i + sequence_length - 1]
            seq_type_target = type_targets_values[i + sequence_length - 1]
            
            sequences.append(seq_features)
            targets_change.append(seq_change_target)
            targets_type.append(seq_type_target)

    X = np.array(sequences, dtype=np.float32)
    y_change = np.array(targets_change, dtype=np.int64) # Usually int for classification
    y_type = np.array(targets_type, dtype=np.int64)   # Usually int for classification
    
    if X.shape[0] > 0:
        logger.info(f"Sequences created: X shape {X.shape}, y_change shape {y_change.shape}, y_type shape {y_type.shape}")
        logger.info(f"Target 'tire_change_next_lap' distribution in sequences: {y_change.mean()*100:.2f}% positive.")
    else:
        logger.warning("No sequences were created. Check input data and sequence_length.")
        # Return empty arrays with correct number of dimensions if X is empty
        if X.ndim == 1: # If np.array([]) was created
            X = X.reshape(0, sequence_length, len(feature_cols) if feature_cols else 0)


    return X, y_change, y_type, feature_cols


def save_model_inputs(X, y_change, y_type, feature_cols: List[str], dataset_name: str):
    """Saves the processed sequences and target arrays."""
    MODEL_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True) # Ensure artifacts dir also exists
    
    path_X = MODEL_INPUT_DIR / f"X_{dataset_name}.npy"
    path_y_change = MODEL_INPUT_DIR / f"y_change_{dataset_name}.npy"
    path_y_type = MODEL_INPUT_DIR / f"y_type_{dataset_name}.npy"
    # Save feature_columns.joblib to ARTIFACTS_DIR
    path_features = ARTIFACTS_DIR / "feature_columns.joblib" 

    np.save(path_X, X)
    np.save(path_y_change, y_change)
    np.save(path_y_type, y_type)
    
    if dataset_name == "train": # Save feature_cols list only once with train set
        joblib.dump(feature_cols, path_features)
        logger.info(f"Saved feature column list to {path_features}")

    logger.info(f"Saved {dataset_name} data: X to {path_X}, y_change to {path_y_change}, y_type to {path_y_type}")


def main_preparation_pipeline():
    logger.info("=== STARTING MODEL DATA PREPARATION PIPELINE ===")
    
    try:
        df_featured, encoders = load_data_and_artifacts()
    except FileNotFoundError:
        return # Errors logged in load_data_and_artifacts

    # Temporal split
    # Temporal split uses the new pragmatic logic based on available years
    train_df, val_df, test_df = temporal_train_val_test_split(df_featured)

    # Create sequences for each set
    if not train_df.empty:
        X_train, y_change_train, y_type_train, feature_cols = create_sequences_for_rnn(train_df, SEQUENCE_LENGTH, encoders)
        save_model_inputs(X_train, y_change_train, y_type_train, feature_cols, "train")
    else:
        logger.warning("Training DataFrame is empty. Skipping sequence creation for train set.")

    # Create sequences for each set
    # Ensure feature_cols is defined from train set even if train_df is empty for some reason
    feature_cols_list = []
    if not train_df.empty:
        X_train, y_change_train, y_type_train, feature_cols_list = create_sequences_for_rnn(train_df, SEQUENCE_LENGTH, encoders)
        save_model_inputs(X_train, y_change_train, y_type_train, feature_cols_list, "train")
    else:
        logger.warning("Training DataFrame is empty. Saving empty arrays for train split.")
        # Save empty arrays to ensure consistency and overwrite old files if any
        # Need to know num_features for X shape, get from encoders or a config if train_df is empty.
        # Fallback: if feature_cols_list is empty, create a dummy one for shape.
        # This case should be rare if data extraction and processing work.
        # For now, assume feature_cols_list would be populated if train_df had data.
        # If train_df is empty, feature_cols_list will be empty from create_sequences_for_rnn.
        # Let's get num_features from a loaded artifact if possible, or define a default.
        # This is complex if train_df is truly empty. For now, rely on create_sequences_for_rnn returning empty feature_cols.
        num_features_for_empty = len(joblib.load(ARTIFACTS_DIR / "feature_columns.joblib")) if (ARTIFACTS_DIR / "feature_columns.joblib").exists() else 0

        save_model_inputs(
            np.array([]).reshape(0, SEQUENCE_LENGTH, num_features_for_empty), 
            np.array([]), np.array([]), 
            [], "train" # feature_cols_list would be empty
        )


    if not val_df.empty:
        X_val, y_change_val, y_type_val, _ = create_sequences_for_rnn(val_df, SEQUENCE_LENGTH, encoders)
        save_model_inputs(X_val, y_change_val, y_type_val, [], "val") # feature_cols not saved for val/test
    else:
        logger.warning("Validation DataFrame is empty. Saving empty arrays for val split.")
        num_features_for_empty = len(feature_cols_list) if feature_cols_list else (len(joblib.load(ARTIFACTS_DIR / "feature_columns.joblib")) if (ARTIFACTS_DIR / "feature_columns.joblib").exists() else 0)
        save_model_inputs(
            np.array([]).reshape(0, SEQUENCE_LENGTH, num_features_for_empty), 
            np.array([]), np.array([]), 
            [], "val"
        )

    if not test_df.empty:
        X_test, y_change_test, y_type_test, _ = create_sequences_for_rnn(test_df, SEQUENCE_LENGTH, encoders)
        save_model_inputs(X_test, y_change_test, y_type_test, [], "test")
    else:
        logger.warning("Test DataFrame is empty. Saving empty arrays for test split.")
        num_features_for_empty = len(feature_cols_list) if feature_cols_list else (len(joblib.load(ARTIFACTS_DIR / "feature_columns.joblib")) if (ARTIFACTS_DIR / "feature_columns.joblib").exists() else 0)
        save_model_inputs(
            np.array([]).reshape(0, SEQUENCE_LENGTH, num_features_for_empty), 
            np.array([]), np.array([]), 
            [], "test"
        )

    logger.info("=== MODEL DATA PREPARATION PIPELINE COMPLETED ===")
    logger.info(f"Model inputs saved in: {MODEL_INPUT_DIR}")

if __name__ == "__main__":
    # Adjust paths if running script directly from its location
    script_path = Path(__file__).resolve()
    if "F1/local/src/preprocessing" in str(script_path):
        BASE_PATH = script_path.parent.parent.parent # F1/local
        FEATURE_ENGINEERED_DATA_PATH = BASE_PATH / "drive/feature_engineered_data/featured_dataset.parquet"
        ARTIFACTS_DIR = BASE_PATH / "drive/artifacts"
        MODEL_INPUT_DIR = BASE_PATH / "drive/model_input"
        ENCODERS_PATH = ARTIFACTS_DIR / "encoders.pkl"
        
        project_root_for_direct_run = BASE_PATH.parent # F1
        if str(project_root_for_direct_run) not in sys.path:
             sys.path.insert(0, str(project_root_for_direct_run))
    
    main_preparation_pipeline()

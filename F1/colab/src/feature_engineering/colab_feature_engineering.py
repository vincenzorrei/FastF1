"""
Feature Engineering Script for F1 Tyre Strategy Prediction (Colab)
=================================================================

This script defines functions to create features from the consolidated F1 dataset.
It is adapted for the Colab environment, using paths configured via
`F1/colab/configs/colab_path_config.yaml`.

Key areas:
- Target variable creation
- Temporal features
- Performance-based features
- Weather-related features
- Domain-knowledge features
- Categorical variable encoding
- Numerical feature scaling
- Missing value imputation
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import RobustScaler, LabelEncoder
import yaml # For loading colab_path_config
import joblib # For saving/loading encoders/scalers
import json # For saving/loading imputer values

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

# Configure logging
if COLAB_PATHS and COLAB_PATHS.get('feature_engineering_log_file'):
    LOG_FILE = COLAB_PATHS['feature_engineering_log_file']
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
else:
    fallback_log_dir = Path(__file__).resolve().parent.parent.parent / "logs_colab_fallback"
    fallback_log_dir.mkdir(parents=True, exist_ok=True)
    LOG_FILE = fallback_log_dir / "colab_feature_engineering_fallback.log"
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


def create_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates target variables for multi-task learning.
    """
    logger.info("Creating target variables...")
    df_target = df.copy()
    df_target = df_target.sort_values(['DriverRaceID', 'LapNumber'])
    df_target['NextStint'] = df_target.groupby('DriverRaceID')['Stint'].shift(-1)
    df_target['tire_change_next_lap'] = (
        (df_target['Stint'] != df_target['NextStint']) & 
        df_target['NextStint'].notna()
    ).astype(int)
    df_target['NextCompound'] = df_target.groupby('DriverRaceID')['Compound'].shift(-1)
    df_target['next_tire_type'] = df_target['NextCompound'].where(
        df_target['tire_change_next_lap'] == 1, 
        'NO_CHANGE'
    )
    df_target = df_target.drop(['NextStint', 'NextCompound'], axis=1)
    n_changes = df_target['tire_change_next_lap'].sum()
    total_laps = len(df_target)
    if total_laps > 0:
        logger.info(f"Target created: {n_changes:,} tire changes out of {total_laps:,} laps ({n_changes/total_laps*100:.2f}%)")
    else:
        logger.info("Target created: 0 laps processed.")
    return df_target

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates temporal and progress-related features."""
    logger.info("Creating temporal features...")
    df_temp = df.copy()
    if 'RaceID' in df_temp.columns and 'LapNumber' in df_temp.columns:
        df_temp['lap_progress'] = df_temp.groupby('RaceID')['LapNumber'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0
        )
    else:
        df_temp['lap_progress'] = 0
    if 'Compound' in df_temp.columns and 'TyreLife' in df_temp.columns:
        compound_stint_avg = df_temp.groupby('Compound')['TyreLife'].quantile(0.75).fillna(20)
        df_temp['expected_stint_length'] = df_temp['Compound'].map(compound_stint_avg).fillna(20)
        df_temp['stint_progress'] = (df_temp['TyreLife'] / df_temp['expected_stint_length']).clip(0, 2)
    else:
        df_temp['expected_stint_length'] = 20
        df_temp['stint_progress'] = 0
    if 'Position' in df_temp.columns:
        df_temp['position_inverted'] = 21 - df_temp['Position'] 
        df_temp['is_top_3'] = (df_temp['Position'] <= 3).astype(int)
        df_temp['is_points_position'] = (df_temp['Position'] <= 10).astype(int)
    else:
        df_temp['position_inverted'], df_temp['is_top_3'], df_temp['is_points_position'] = 0, 0, 0
    logger.info("Temporal features created successfully.")
    return df_temp

def create_performance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates performance and trend-based features."""
    logger.info("Creating performance features...")
    df_perf = df.copy()
    for col_name, source_col, group_col in [
        ('laptime_trend_3', 'LapTime', 'DriverRaceID'),
        ('delta_ahead_trend', 'TimeDeltaToDriverAhead', 'DriverRaceID')
    ]:
        if group_col in df_perf.columns and source_col in df_perf.columns:
            temp_col = f"{source_col}_sec_temp"
            if pd.api.types.is_timedelta64_dtype(df_perf[source_col]):
                df_perf[temp_col] = df_perf[source_col].dt.total_seconds()
            else:
                df_perf[temp_col] = pd.to_numeric(df_perf[source_col], errors='coerce')
            
            # Fill NaNs before rolling to avoid issues with polyfit on all-NaN windows
            df_perf[temp_col] = df_perf.groupby(group_col)[temp_col].ffill().bfill()

            rolling_result = df_perf.groupby(group_col)[temp_col].rolling(
                window=3, min_periods=2
            ).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 and not np.isnan(x).all() else 0, raw=True)
            df_perf[col_name] = rolling_result.reset_index(level=0, drop=True).fillna(0)
            if temp_col in df_perf.columns: df_perf.drop(columns=[temp_col], inplace=True)
        else:
            df_perf[col_name] = 0
    
    if 'DriverRaceID' in df_perf.columns and 'Stint' in df_perf.columns and 'LapTime' in df_perf.columns:
        temp_laptime_col = 'LapTime_sec_degrad_temp'
        if pd.api.types.is_timedelta64_dtype(df_perf['LapTime']):
            df_perf[temp_laptime_col] = df_perf['LapTime'].dt.total_seconds()
        else:
            df_perf[temp_laptime_col] = pd.to_numeric(df_perf['LapTime'], errors='coerce')
        df_perf[temp_laptime_col] = df_perf.groupby(['DriverRaceID', 'Stint'])[temp_laptime_col].ffill().bfill()
        df_perf['tire_degradation_rate'] = df_perf.groupby(['DriverRaceID', 'Stint'])[temp_laptime_col].pct_change().fillna(0)
        if temp_laptime_col in df_perf.columns: df_perf.drop(columns=[temp_laptime_col], inplace=True)
    else:
        df_perf['tire_degradation_rate'] = 0

    if 'RaceID' in df_perf.columns and 'Compound' in df_perf.columns and 'TyreLife' in df_perf.columns:
        df_perf['compound_age_ratio'] = df_perf.groupby(['RaceID', 'Compound'])['TyreLife'].transform(
            lambda x: x / x.quantile(0.9) if x.quantile(0.9) > 0 else (x / x.mean() if x.mean() > 0 else 1)
        ).fillna(1)
    else:
        df_perf['compound_age_ratio'] = 1
        
    for col_name, source_col in [('log_delta_ahead', 'TimeDeltaToDriverAhead'), ('log_delta_behind', 'TimeDeltaToDriverBehind')]:
        if source_col in df_perf.columns:
            numeric_source = df_perf[source_col].dt.total_seconds() if pd.api.types.is_timedelta64_dtype(df_perf[source_col]) else pd.to_numeric(df_perf[source_col], errors='coerce')
            df_perf[col_name] = np.log1p(np.abs(numeric_source.fillna(0)))
        else:
            df_perf[col_name] = 0
    logger.info("Performance features created successfully.")
    return df_perf

def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates derived weather-based features."""
    logger.info("Creating weather features...")
    df_weather = df.copy()
    weather_cols_stability = ['AirTemp', 'TrackTemp', 'Humidity', 'WindSpeed']
    for col in weather_cols_stability:
        if col in df_weather.columns and 'RaceID' in df_weather.columns:
            df_weather[f'{col}_stability'] = df_weather.groupby('RaceID')[col].rolling(window=5, min_periods=3).std().reset_index(0, drop=True).fillna(0)
        else:
            df_weather[f'{col}_stability'] = 0
    conditions = pd.Series([False] * len(df_weather), index=df_weather.index)
    if 'Rainfall' in df_weather.columns: conditions = conditions | (df_weather['Rainfall'] == True)
    if 'Humidity' in df_weather.columns: conditions = conditions | (df_weather['Humidity'] > 80)
    if 'WindSpeed' in df_weather.columns: conditions = conditions | (df_weather['WindSpeed'] > 15)
    df_weather['difficult_conditions'] = conditions.astype(int)
    if 'TrackTemp' in df_weather.columns and 'AirTemp' in df_weather.columns:
        df_weather['temp_delta'] = df_weather['TrackTemp'] - df_weather['AirTemp']
    else:
        df_weather['temp_delta'] = 0
    logger.info("Weather features created successfully.")
    return df_weather

def create_domain_knowledge_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates features based on F1 domain expertise."""
    logger.info("Creating domain knowledge features...")
    df_domain = df.copy()
    typical_stint_lengths = {'SOFT': 15, 'MEDIUM': 25, 'HARD': 35, 'SUPERSOFT': 12, 'ULTRASOFT': 10, 'INTERMEDIATE': 20, 'WET': 15, 'HYPERSOFT': 8}
    if 'Compound' in df_domain.columns:
        df_domain['expected_stint_length_domain'] = df_domain['Compound'].map(typical_stint_lengths).fillna(20)
    else:
        df_domain['expected_stint_length_domain'] = 20
    if 'TyreLife' in df_domain.columns:
        df_domain['stint_length_ratio'] = df_domain['TyreLife'] / df_domain['expected_stint_length_domain']
    else:
        df_domain['stint_length_ratio'] = 0
    if 'LapNumber' in df_domain.columns:
        df_domain['in_pit_window_early'] = (df_domain['LapNumber'].between(10, 20)).astype(int)
        df_domain['in_pit_window_mid'] = (df_domain['LapNumber'].between(30, 45)).astype(int)
        df_domain['in_pit_window_late'] = (df_domain['LapNumber'].between(50, 65)).astype(int)
    else:
        df_domain['in_pit_window_early'], df_domain['in_pit_window_mid'], df_domain['in_pit_window_late'] = 0,0,0
    if 'Stint' in df_domain.columns and 'stint_length_ratio' in df_domain.columns:
        df_domain['likely_one_stop'] = ((df_domain['Stint'] == 1) & (df_domain['stint_length_ratio'] > 1.3)).astype(int)
        df_domain['likely_two_stop'] = ((df_domain['Stint'] >= 2) & (df_domain['stint_length_ratio'] < 1.1)).astype(int)
    else:
        df_domain['likely_one_stop'], df_domain['likely_two_stop'] = 0,0
    if 'DriverRaceID' in df_domain.columns and 'Compound' in df_domain.columns:
        df_domain['compound_strategy_pattern'] = df_domain.groupby('DriverRaceID')['Compound'].transform(lambda x: '_'.join(x.astype(str).unique()) if x.nunique() > 0 else 'UNKNOWN')
    else:
        df_domain['compound_strategy_pattern'] = 'UNKNOWN'
    logger.info("Domain knowledge features created successfully.")
    return df_domain

def handle_categorical_variables(df: pd.DataFrame, training_mode: bool = True, encoders_path: Optional[Path] = None) -> Tuple[pd.DataFrame, Optional[Dict]]:
    logger.info(f"Handling categorical variables. Training mode: {training_mode}")
    df_cat = df.copy()
    categorical_columns = ['Compound', 'Team', 'Location', 'Driver', 'next_tire_type'] # Added next_tire_type for consistent encoding
    encoders = {}
    if training_mode:
        for col in categorical_columns:
            if col in df_cat.columns:
                df_cat[col] = df_cat[col].astype(str).fillna('UNKNOWN')
                le = LabelEncoder()
                unique_values = list(df_cat[col].unique())
                if 'UNKNOWN' not in unique_values: unique_values.append('UNKNOWN')
                if col == 'next_tire_type' and 'NO_CHANGE' not in unique_values: unique_values.append('NO_CHANGE') # Specific for target
                le.fit(unique_values)
                encoders[col] = le
                df_cat[f'{col}_encoded'] = le.transform(df_cat[col])
        if 'compound_strategy_pattern' in df_cat.columns:
            strategy_freq = df_cat['compound_strategy_pattern'].value_counts(normalize=True)
            encoders['compound_strategy_pattern_freq_map'] = strategy_freq.to_dict() # Store as dict
            df_cat['compound_strategy_freq'] = df_cat['compound_strategy_pattern'].map(strategy_freq).fillna(0)
        if encoders_path:
            encoders_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(encoders, encoders_path)
            logger.info(f"Saved encoders to {encoders_path}")
    else: # Inference mode
        if encoders_path and encoders_path.exists():
            encoders = joblib.load(encoders_path)
            logger.info(f"Loaded encoders from {encoders_path}")
            for col in categorical_columns:
                if col in df_cat.columns and col in encoders:
                    df_cat[col] = df_cat[col].astype(str).fillna('UNKNOWN')
                    le = encoders[col]
                    # Handle unseen labels by mapping them to 'UNKNOWN' if 'UNKNOWN' is a known class
                    known_classes = set(le.classes_)
                    df_cat[f'{col}_encoded'] = df_cat[col].apply(lambda x: x if x in known_classes else 'UNKNOWN')
                    df_cat[f'{col}_encoded'] = le.transform(df_cat[f'{col}_encoded'])
                elif col in df_cat.columns:
                     df_cat[f'{col}_encoded'] = -1 # Fallback if encoder for col is missing
            if 'compound_strategy_pattern' in df_cat.columns and 'compound_strategy_pattern_freq_map' in encoders:
                strategy_freq_map = encoders['compound_strategy_pattern_freq_map']
                df_cat['compound_strategy_freq'] = df_cat['compound_strategy_pattern'].map(strategy_freq_map).fillna(0)
            else:
                if 'compound_strategy_pattern' in df_cat.columns: df_cat['compound_strategy_freq'] = 0
        else:
            logger.error(f"Encoders path {encoders_path} not found for non-training mode. Categorical features not properly encoded.")
            # Add placeholder columns to avoid downstream errors
            for col in categorical_columns:
                if col in df_cat.columns: df_cat[f'{col}_encoded'] = -1
            if 'compound_strategy_pattern' in df_cat.columns: df_cat['compound_strategy_freq'] = 0

    logger.info("Categorical variables handled.")
    return df_cat, encoders if training_mode else None


def normalize_features(df: pd.DataFrame, training_mode: bool = True, scaler_path: Optional[Path] = None) -> Tuple[pd.DataFrame, Optional[RobustScaler]]:
    logger.info(f"Normalizing features. Training mode: {training_mode}")
    df_norm = df.copy()
    exclude_from_scaling = [
        'Year', 'RaceID', 'DriverID', 'DriverRaceID', 'GlobalLapID', 'LapNumber', 'Stint', 'Position',
        'IsFreshTire', 'Rainfall', 'tire_change_next_lap', # Targets
        'Compound', 'Team', 'Location', 'Driver', 'next_tire_type', 'compound_strategy_pattern', # Raw categoricals
        'PitInTime', 'PitOutTime'
    ]
    encoded_cols = [col for col in df_norm.columns if col.endswith('_encoded')]
    exclude_from_scaling.extend(encoded_cols)
    freq_encoded_cols = [col for col in df_norm.columns if col.endswith('_freq')]
    exclude_from_scaling.extend(freq_encoded_cols)

    numeric_cols_to_scale = [col for col in df_norm.columns if df_norm[col].dtype in ['float64', 'int64', 'float32', 'int32'] and col not in exclude_from_scaling]
    valid_numeric_cols = [col for col in numeric_cols_to_scale if pd.to_numeric(df_norm[col], errors='coerce').notna().all()]
    
    if not valid_numeric_cols:
        logger.warning("No valid numeric columns found to scale.")
        return df_norm, None
    
    scaler = None
    if training_mode:
        scaler = RobustScaler()
        df_norm[valid_numeric_cols] = scaler.fit_transform(df_norm[valid_numeric_cols])
        if scaler_path:
            scaler_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, scaler_path)
            logger.info(f"Saved scaler to {scaler_path}")
    else:
        if scaler_path and scaler_path.exists():
            scaler = joblib.load(scaler_path)
            df_norm[valid_numeric_cols] = scaler.transform(df_norm[valid_numeric_cols])
        else:
            logger.error(f"Scaler path {scaler_path} not found for non-training mode. Features not scaled.")
            return df_norm, None
    return df_norm, scaler

def handle_missing_values(df: pd.DataFrame, training_mode: bool = True, imputer_values_path: Optional[Path] = None) -> pd.DataFrame:
    logger.info(f"Handling missing values. Training mode: {training_mode}")
    df_clean = df.copy()
    ffill_cols = ['TyreLife', 'Stint', 'Compound', 'AirTemp', 'TrackTemp', 'Humidity', 'WindSpeed'] # Keep 'Compound' for ffill before encoding
    median_cols_config = {'LapTime': None, 'TimeDeltaToDriverAhead': None, 'TimeDeltaToDriverBehind': None} # Store actual median values
    zero_fill_cols = ['tire_degradation_rate', 'laptime_trend_3', 'delta_ahead_trend']
    imputation_values_storage = {}

    if training_mode:
        for col in median_cols_config.keys():
            if col in df_clean.columns:
                # Convert to numeric (seconds for timedelta) before median calculation
                numeric_series = df_clean[col].dt.total_seconds() if pd.api.types.is_timedelta64_dtype(df_clean[col].dtype) else pd.to_numeric(df_clean[col], errors='coerce')
                median_val = numeric_series.median()
                imputation_values_storage[col] = median_val if pd.notna(median_val) else 0 # Store numeric median
        if imputer_values_path:
            imputer_values_path.parent.mkdir(parents=True, exist_ok=True)
            with open(imputer_values_path, 'w') as f:
                json.dump({k: (v.item() if isinstance(v, np.generic) else v) for k, v in imputation_values_storage.items()}, f)
            logger.info(f"Saved imputation values to {imputer_values_path}")
    else:
        if imputer_values_path and imputer_values_path.exists():
            with open(imputer_values_path, 'r') as f:
                imputation_values_storage = json.load(f)
            logger.info(f"Loaded imputation values from {imputer_values_path}")
        else:
            logger.error(f"Imputation values path {imputer_values_path} not found. Using current data medians (suboptimal).")
            for col in median_cols_config.keys(): # Fallback
                if col in df_clean.columns:
                    numeric_series = df_clean[col].dt.total_seconds() if pd.api.types.is_timedelta64_dtype(df_clean[col].dtype) else pd.to_numeric(df_clean[col], errors='coerce')
                    imputation_values_storage[col] = numeric_series.median() if pd.notna(numeric_series.median()) else 0


    for col in ffill_cols:
        if col in df_clean.columns:
            if 'DriverRaceID' in df_clean.columns:
                 df_clean[col] = df_clean.groupby('DriverRaceID')[col].ffill()
            else: df_clean[col] = df_clean[col].ffill()
            df_clean[col] = df_clean[col].bfill()

    for col in median_cols_config.keys():
        if col in df_clean.columns and col in imputation_values_storage:
            fill_value = imputation_values_storage[col]
            # Convert column to numeric (seconds) if it's timedelta, then fill
            if pd.api.types.is_timedelta64_dtype(df_clean[col].dtype):
                df_clean[col] = df_clean[col].dt.total_seconds()
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(fill_value)
        elif col in df_clean.columns: # Fallback if value not in storage (e.g. new column)
             df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)


    for col in zero_fill_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
            
    remaining_na = df_clean.isnull().sum().sum()
    if remaining_na > 0:
        logger.warning(f"Remaining NaNs after imputation: {remaining_na}. Applying global fillna(0).")
        df_clean = df_clean.fillna(0) # Final catch-all
    else:
        logger.info("Missing values handled successfully.")
    return df_clean

# Main execution for testing (if needed)
if __name__ == '__main__':
    logger.info("--- Running Feature Engineering Demo (Colab Version) ---")
    # Dummy DataFrame for demonstration (same as local version)
    data = {
        'RaceID': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
        'DriverRaceID': ['1_A', '1_A', '1_A', '1_B', '1_B', '1_B', '2_C', '2_C', '2_C', '2_C'],
        'LapNumber': [1, 2, 3, 1, 2, 3, 1, 2, 3, 4],
        'Stint': [1, 1, 2, 1, 1, 1, 1, 1, 2, 2],
        'Compound': ['SOFT', 'SOFT', 'MEDIUM', 'HARD', 'HARD', 'HARD', 'MEDIUM', 'MEDIUM', 'SOFT', 'SOFT'],
        'TyreLife': [1, 2, 1, 1, 2, 3, 1, 2, 1, 2],
        'Position': [1, 2, 1, 3, 2, 2, 1, 1, 2, 1],
        'LapTime': pd.to_timedelta([90.1, 90.5, 90.3, 91.0, 90.9, 90.8, 89.5, 89.4, 89.8, 89.7], unit='s'),
        'TimeDeltaToDriverAhead': pd.to_timedelta([np.nan, 0.5, 1.2, np.nan, 0.3, 0.6, np.nan, 0.1, 2.0, 0.5], unit='s'),
        'TimeDeltaToDriverBehind': pd.to_timedelta([0.8, 0.4, np.nan, 1.0, 0.7, np.nan, 0.5, 0.8, np.nan, 1.1], unit='s'),
        'AirTemp': [25, 25.1, 25, 28, 28.1, 28, 22, 22, 22.1, 22.1],
        'TrackTemp': [35, 35.2, 35, 40, 40.1, 40, 30, 30, 30.2, 30.2],
        'Humidity': [60, 60.1, 60, 55, 55.1, 55, 70, 70, 70.1, 70.1],
        'WindSpeed': [5, 5.1, 5, 3, 3.1, 3, 8, 8, 8.1, 8.1],
        'Rainfall': [False, False, False, False, False, False, True, True, False, False],
        'Team': ['TeamA', 'TeamA', 'TeamA', 'TeamB', 'TeamB', 'TeamB', 'TeamC', 'TeamC', 'TeamC', 'TeamC'],
        'Location': ['CircuitX', 'CircuitX', 'CircuitX', 'CircuitX', 'CircuitX', 'CircuitX', 'CircuitY', 'CircuitY', 'CircuitY', 'CircuitY'],
        'Driver': ['Driver1', 'Driver1', 'Driver1', 'Driver2', 'Driver2', 'Driver2', 'Driver3', 'Driver3', 'Driver3', 'Driver3']
    }
    sample_df = pd.DataFrame(data)

    if COLAB_PATHS:
        artifacts_dir = COLAB_PATHS.get('artifacts_directory')
        if artifacts_dir:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            encoders_demo_path = artifacts_dir / "colab_encoders.pkl"
            scaler_demo_path = artifacts_dir / "colab_scaler.pkl"
            imputer_demo_path = artifacts_dir / "colab_imputer_values.json"

            df_processed = create_target_variables(sample_df.copy())
            df_processed = create_temporal_features(df_processed)
            df_processed = create_performance_features(df_processed)
            df_processed = create_weather_features(df_processed)
            df_processed = create_domain_knowledge_features(df_processed)
            df_processed = handle_missing_values(df_processed, training_mode=True, imputer_values_path=imputer_demo_path)
            df_processed, _ = handle_categorical_variables(df_processed, training_mode=True, encoders_path=encoders_demo_path)
            df_processed, _ = normalize_features(df_processed, training_mode=True, scaler_path=scaler_demo_path)
            
            logger.info("--- Feature Engineering Demo (Colab) Completed ---")
            logger.info("Processed DataFrame head:")
            logger.info(df_processed.head())
            logger.info("Processed DataFrame info:")
            df_processed.info()
        else:
            logger.error("Artifacts directory not found in COLAB_PATHS. Demo cannot save artifacts.")
    else:
        logger.error("COLAB_PATHS not loaded. Demo cannot run.")

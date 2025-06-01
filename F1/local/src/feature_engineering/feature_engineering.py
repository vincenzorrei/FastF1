"""
Feature Engineering Script for F1 Tyre Strategy Prediction
=========================================================

This script defines functions to create features from the consolidated F1 dataset.
It adapts and refactors logic from the 'Vincenzo/dataset/data_preprocessing.py' script.

Key areas:
- Target variable creation
- Temporal features
- Performance-based features
- Weather-related features
- Domain-knowledge features
- Categorical variable encoding
- Numerical feature scaling
- Missing value imputation

Autore: Cline (AI Software Engineer)
Data: 2025-05-31
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import RobustScaler, LabelEncoder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates target variables for multi-task learning.
    
    Target primario: tire_change_next_lap (binary)
    Target secondario: next_tire_type (categorical)
    """
    logger.info("Creating target variables...")
    
    df_target = df.copy()
    
    # Ensure correct temporal order
    df_target = df_target.sort_values(['DriverRaceID', 'LapNumber'])
    
    # Primary target: tire change in the next lap
    df_target['NextStint'] = df_target.groupby('DriverRaceID')['Stint'].shift(-1)
    df_target['tire_change_next_lap'] = (
        (df_target['Stint'] != df_target['NextStint']) & 
        df_target['NextStint'].notna()
    ).astype(int)
    
    # Secondary target: tire compound in the next stint
    df_target['NextCompound'] = df_target.groupby('DriverRaceID')['Compound'].shift(-1)
    
    # Secondary target is valid only when there's a tire change
    df_target['next_tire_type'] = df_target['NextCompound'].where(
        df_target['tire_change_next_lap'] == 1, 
        'NO_CHANGE' # Placeholder for no change
    )
    
    # Cleanup temporary columns
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
    
    # 1. Normalized race progress (0-1)
    if 'RaceID' in df_temp.columns and 'LapNumber' in df_temp.columns:
        df_temp['lap_progress'] = df_temp.groupby('RaceID')['LapNumber'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0
        )
    else:
        df_temp['lap_progress'] = 0
        logger.warning("Columns 'RaceID' or 'LapNumber' not found for 'lap_progress'. Defaulting to 0.")

    # 2. Tire stint progress
    if 'Compound' in df_temp.columns and 'TyreLife' in df_temp.columns:
        compound_stint_avg = df_temp.groupby('Compound')['TyreLife'].quantile(0.75).fillna(20) # Default if compound new
        df_temp['expected_stint_length'] = df_temp['Compound'].map(compound_stint_avg)
        df_temp['expected_stint_length'] = df_temp['expected_stint_length'].fillna(20) # Fill for any remaining NaNs
        df_temp['stint_progress'] = df_temp['TyreLife'] / df_temp['expected_stint_length']
        df_temp['stint_progress'] = df_temp['stint_progress'].clip(0, 2)  # Cap at 200%
    else:
        df_temp['expected_stint_length'] = 20
        df_temp['stint_progress'] = 0
        logger.warning("Columns 'Compound' or 'TyreLife' not found for 'stint_progress'. Defaulting to 0.")

    # 3. Position-derived features
    if 'Position' in df_temp.columns:
        df_temp['position_inverted'] = 21 - df_temp['Position'] 
        df_temp['is_top_3'] = (df_temp['Position'] <= 3).astype(int)
        df_temp['is_points_position'] = (df_temp['Position'] <= 10).astype(int)
    else:
        df_temp['position_inverted'] = 0
        df_temp['is_top_3'] = 0
        df_temp['is_points_position'] = 0
        logger.warning("Column 'Position' not found for position features. Defaulting to 0.")
        
    logger.info("Temporal features created successfully.")
    return df_temp

def create_performance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates performance and trend-based features."""
    logger.info("Creating performance features...")
    
    df_perf = df.copy()
    
    # 1. Laptime degradation trend (rolling window)
    if 'DriverRaceID' in df_perf.columns and 'LapTime' in df_perf.columns:
        if pd.api.types.is_timedelta64_dtype(df_perf['LapTime']):
            df_perf['LapTime_sec_temp'] = df_perf['LapTime'].dt.total_seconds()
            source_col_for_lap_trend = 'LapTime_sec_temp'
        else: # If LapTime is already numeric
            source_col_for_lap_trend = 'LapTime'
            
        rolling_apply_result_lap = df_perf.groupby('DriverRaceID')[source_col_for_lap_trend].rolling(
            window=3, min_periods=2
        ).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0, raw=True)
        
        df_perf['laptime_trend_3'] = rolling_apply_result_lap.reset_index(level=0, drop=True)
        df_perf['laptime_trend_3'] = df_perf['laptime_trend_3'].fillna(0)
        
        if 'LapTime_sec_temp' in df_perf.columns:
            df_perf.drop(columns=['LapTime_sec_temp'], inplace=True)
    else:
        df_perf['laptime_trend_3'] = 0
        logger.warning("Columns 'DriverRaceID' or 'LapTime' not found for 'laptime_trend_3'. Defaulting to 0.")

    # 2. Trend in gap to driver ahead
    if 'DriverRaceID' in df_perf.columns and 'TimeDeltaToDriverAhead' in df_perf.columns:
        if pd.api.types.is_timedelta64_dtype(df_perf['TimeDeltaToDriverAhead']):
            df_perf['TimeDeltaToDriverAhead_sec_temp'] = df_perf['TimeDeltaToDriverAhead'].dt.total_seconds()
            source_col_for_delta_trend = 'TimeDeltaToDriverAhead_sec_temp'
        else: # If already numeric
            source_col_for_delta_trend = 'TimeDeltaToDriverAhead'

        rolling_apply_result_delta = df_perf.groupby('DriverRaceID')[source_col_for_delta_trend].rolling(
            window=3, min_periods=2
        ).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0, raw=True)

        df_perf['delta_ahead_trend'] = rolling_apply_result_delta.reset_index(level=0, drop=True)
        df_perf['delta_ahead_trend'] = df_perf['delta_ahead_trend'].fillna(0)

        if 'TimeDeltaToDriverAhead_sec_temp' in df_perf.columns:
            df_perf.drop(columns=['TimeDeltaToDriverAhead_sec_temp'], inplace=True)
    else:
        df_perf['delta_ahead_trend'] = 0
        logger.warning("Columns 'DriverRaceID' or 'TimeDeltaToDriverAhead' not found for 'delta_ahead_trend'. Defaulting to 0.")
        
    # 3. Tire degradation rate (LapTime percentage change within stint)
    if 'DriverRaceID' in df_perf.columns and 'Stint' in df_perf.columns and 'LapTime' in df_perf.columns:
        if pd.api.types.is_timedelta64_dtype(df_perf['LapTime']):
            df_perf['LapTime_sec_degrad_temp'] = df_perf['LapTime'].dt.total_seconds()
            # Ensure NaNs in the seconds column are handled if necessary before pct_change
            # For example, fill with a value or ensure pct_change handles it as desired.
            # pct_change on a series with NaNs might propagate them.
            df_perf['LapTime_sec_degrad_temp'] = df_perf['LapTime_sec_degrad_temp'].ffill().bfill() # Example: fill NaNs
            df_perf['tire_degradation_rate'] = df_perf.groupby(['DriverRaceID', 'Stint'])['LapTime_sec_degrad_temp'].pct_change()
            df_perf.drop(columns=['LapTime_sec_degrad_temp'], inplace=True)
        else: # If LapTime is already numeric
            df_perf['tire_degradation_rate'] = df_perf.groupby(['DriverRaceID', 'Stint'])['LapTime'].pct_change()
        df_perf['tire_degradation_rate'] = df_perf['tire_degradation_rate'].fillna(0)
    else:
        df_perf['tire_degradation_rate'] = 0
        logger.warning("Columns 'DriverRaceID', 'Stint', or 'LapTime' not found for 'tire_degradation_rate'. Defaulting to 0.")

    # 4. Relative tire age for compound in race
    if 'RaceID' in df_perf.columns and 'Compound' in df_perf.columns and 'TyreLife' in df_perf.columns:
        df_perf['compound_age_ratio'] = df_perf.groupby(['RaceID', 'Compound'])['TyreLife'].transform(
            lambda x: x / x.quantile(0.9) if x.quantile(0.9) > 0 else x / (x.mean() if x.mean() > 0 else 1) # Robust handling
        )
        df_perf['compound_age_ratio'] = df_perf['compound_age_ratio'].fillna(1) # Default if no data
    else:
        df_perf['compound_age_ratio'] = 1
        logger.warning("Columns 'RaceID', 'Compound', or 'TyreLife' not found for 'compound_age_ratio'. Defaulting to 1.")

    # 5. Log-transformed gap times
    if 'TimeDeltaToDriverAhead' in df_perf.columns:
        if pd.api.types.is_timedelta64_dtype(df_perf['TimeDeltaToDriverAhead']):
            df_perf['log_delta_ahead'] = np.log1p(np.abs(df_perf['TimeDeltaToDriverAhead'].dt.total_seconds().fillna(0)))
        else: # If already numeric
            df_perf['log_delta_ahead'] = np.log1p(np.abs(df_perf['TimeDeltaToDriverAhead'].fillna(0)))
    else:
        df_perf['log_delta_ahead'] = 0
        logger.warning("Column 'TimeDeltaToDriverAhead' not found for 'log_delta_ahead'. Defaulting to 0.")
        
    if 'TimeDeltaToDriverBehind' in df_perf.columns:
        if pd.api.types.is_timedelta64_dtype(df_perf['TimeDeltaToDriverBehind']):
            df_perf['log_delta_behind'] = np.log1p(np.abs(df_perf['TimeDeltaToDriverBehind'].dt.total_seconds().fillna(0)))
        else: # If already numeric
            df_perf['log_delta_behind'] = np.log1p(np.abs(df_perf['TimeDeltaToDriverBehind'].fillna(0)))
    else:
        df_perf['log_delta_behind'] = 0
        logger.warning("Column 'TimeDeltaToDriverBehind' not found for 'log_delta_behind'. Defaulting to 0.")
        
    logger.info("Performance features created successfully.")
    return df_perf

def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates derived weather-based features."""
    logger.info("Creating weather features...")
    
    df_weather = df.copy()
    
    weather_cols_stability = ['AirTemp', 'TrackTemp', 'Humidity', 'WindSpeed']
    
    for col in weather_cols_stability:
        if col in df_weather.columns and 'RaceID' in df_weather.columns:
            df_weather[f'{col}_stability'] = df_weather.groupby('RaceID')[col].rolling(
                window=5, min_periods=3
            ).std().reset_index(0, drop=True).fillna(0)
        else:
            df_weather[f'{col}_stability'] = 0
            logger.warning(f"Column '{col}' or 'RaceID' not found for '{col}_stability'. Defaulting to 0.")

    # Difficult weather conditions index
    conditions = pd.Series([False] * len(df_weather), index=df_weather.index) # Default to False Series
    if 'Rainfall' in df_weather.columns:
        conditions = conditions | (df_weather['Rainfall'] == True)
    if 'Humidity' in df_weather.columns:
        conditions = conditions | (df_weather['Humidity'] > 80) # This was the specific warning line
    if 'WindSpeed' in df_weather.columns:
        conditions = conditions | (df_weather['WindSpeed'] > 15)
    df_weather['difficult_conditions'] = conditions.astype(int)
    
    # Temperature delta (Track - Air)
    if 'TrackTemp' in df_weather.columns and 'AirTemp' in df_weather.columns:
        df_weather['temp_delta'] = df_weather['TrackTemp'] - df_weather['AirTemp']
    else:
        df_weather['temp_delta'] = 0
        logger.warning("Columns 'TrackTemp' or 'AirTemp' not found for 'temp_delta'. Defaulting to 0.")
        
    logger.info("Weather features created successfully.")
    return df_weather

def create_domain_knowledge_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates features based on F1 domain expertise."""
    logger.info("Creating domain knowledge features...")
    
    df_domain = df.copy()
    
    typical_stint_lengths = {
        'SOFT': 15, 'MEDIUM': 25, 'HARD': 35,
        'SUPERSOFT': 12, 'ULTRASOFT': 10,
        'INTERMEDIATE': 20, 'WET': 15,
        'HYPERSOFT': 8 # Added for completeness
    }
    
    if 'Compound' in df_domain.columns:
        df_domain['expected_stint_length_domain'] = df_domain['Compound'].map(
            typical_stint_lengths
        ).fillna(20) # Default for unknown/new compounds
    else:
        df_domain['expected_stint_length_domain'] = 20
        logger.warning("Column 'Compound' not found for 'expected_stint_length_domain'. Defaulting to 20.")

    if 'TyreLife' in df_domain.columns:
        df_domain['stint_length_ratio'] = df_domain['TyreLife'] / df_domain['expected_stint_length_domain']
    else:
        df_domain['stint_length_ratio'] = 0
        logger.warning("Column 'TyreLife' not found for 'stint_length_ratio'. Defaulting to 0.")

    # Pit stop windows
    if 'LapNumber' in df_domain.columns:
        df_domain['in_pit_window_early'] = (df_domain['LapNumber'].between(10, 20)).astype(int)
        df_domain['in_pit_window_mid'] = (df_domain['LapNumber'].between(30, 45)).astype(int) # Adjusted mid window
        df_domain['in_pit_window_late'] = (df_domain['LapNumber'].between(50, 65)).astype(int) # Adjusted late window
    else:
        df_domain['in_pit_window_early'] = 0
        df_domain['in_pit_window_mid'] = 0
        df_domain['in_pit_window_late'] = 0
        logger.warning("Column 'LapNumber' not found for pit window features. Defaulting to 0.")

    # Inferred strategy
    if 'Stint' in df_domain.columns and 'stint_length_ratio' in df_domain.columns:
        df_domain['likely_one_stop'] = ((df_domain['Stint'] == 1) & (df_domain['stint_length_ratio'] > 1.3)).astype(int) # Adjusted threshold
        df_domain['likely_two_stop'] = ((df_domain['Stint'] >= 2) & (df_domain['stint_length_ratio'] < 1.1)).astype(int) # Adjusted threshold
    else:
        df_domain['likely_one_stop'] = 0
        df_domain['likely_two_stop'] = 0
        logger.warning("Columns 'Stint' or 'stint_length_ratio' not found for strategy features. Defaulting to 0.")
    
    # Compound strategy pattern
    if 'DriverRaceID' in df_domain.columns and 'Compound' in df_domain.columns:
        df_domain['compound_strategy_pattern'] = df_domain.groupby('DriverRaceID')['Compound'].transform(
            lambda x: '_'.join(x.astype(str).unique()) if x.nunique() > 0 else 'UNKNOWN'
        )
    else:
        df_domain['compound_strategy_pattern'] = 'UNKNOWN'
        logger.warning("Columns 'DriverRaceID' or 'Compound' not found for 'compound_strategy_pattern'. Defaulting to UNKNOWN.")
        
    logger.info("Domain knowledge features created successfully.")
    return df_domain

def handle_categorical_variables(df: pd.DataFrame, training_mode: bool = True, encoders_path: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """
    Handles categorical variables using LabelEncoding.
    Saves/loads encoders if path is provided.
    """
    logger.info(f"Handling categorical variables. Training mode: {training_mode}")
    df_cat = df.copy()
    
    categorical_columns = ['Compound', 'Team', 'Location', 'Driver'] # Base categoricals
    # 'next_tire_type' is a target, handled separately if needed for input features.
    
    if training_mode:
        encoders = {}
    else:
        if encoders_path and Path(encoders_path).exists():
            import joblib
            encoders = joblib.load(encoders_path)
            logger.info(f"Loaded encoders from {encoders_path}")
        else:
            logger.error(f"Encoders path {encoders_path} not found for non-training mode.")
            return df_cat, None # Or raise error

    for col in categorical_columns:
        if col in df_cat.columns:
            df_cat[col] = df_cat[col].astype(str).fillna('UNKNOWN') # Ensure string type and handle NaNs
            
            if training_mode:
                le = LabelEncoder()
                # Fit on all unique values including 'UNKNOWN' and 'NO_CHANGE' (for Compound-like)
                unique_values = list(df_cat[col].unique())
                if 'NO_CHANGE' not in unique_values: # Relevant for Compound if used as feature
                     unique_values.append('NO_CHANGE')
                if 'UNKNOWN' not in unique_values:
                    unique_values.append('UNKNOWN')
                
                le.fit(unique_values)
                encoders[col] = le
                logger.info(f"Fitted LabelEncoder for {col}: {len(le.classes_)} classes.")
            
            if col in encoders:
                # Transform: handle unseen values by mapping to a special 'UNKNOWN' class if possible
                # For simplicity, we assume classes learned in training cover test data,
                # or unseen values will cause errors if not handled by LabelEncoder's parameters.
                # A robust way is to add all unique values from test to encoder's classes if not present,
                # or map them to an 'unknown' category.
                # Current LabelEncoder will error on unseen. For now, we rely on 'UNKNOWN' filling.
                
                # Filter classes to only those present in the current column, before transform
                current_col_values = df_cat[col].unique()
                # Ensure all values in current_col_values are in le.classes_
                # This is tricky with scikit-learn's LabelEncoder if new labels appear in test.
                # A common strategy is to fit on combined train+test unique values,
                # or map unknown test values to a specific category.
                # For now, we assume 'UNKNOWN' handles most cases.
                
                df_cat[f'{col}_encoded'] = encoders[col].transform(df_cat[col])
            else:
                logger.warning(f"No encoder found for column {col} in non-training mode. Skipping encoding.")
                df_cat[f'{col}_encoded'] = -1 # Placeholder for missing encoding

    # Frequency encoding for high-cardinality 'compound_strategy_pattern'
    if 'compound_strategy_pattern' in df_cat.columns:
        if training_mode:
            strategy_freq = df_cat['compound_strategy_pattern'].value_counts(normalize=True)
            encoders['compound_strategy_pattern_freq_map'] = strategy_freq
            df_cat['compound_strategy_freq'] = df_cat['compound_strategy_pattern'].map(strategy_freq).fillna(0)
        elif 'compound_strategy_pattern_freq_map' in encoders:
            strategy_freq_map = encoders['compound_strategy_pattern_freq_map']
            df_cat['compound_strategy_freq'] = df_cat['compound_strategy_pattern'].map(strategy_freq_map).fillna(0) # Fill new strategies with 0 freq
        else:
            df_cat['compound_strategy_freq'] = 0
            logger.warning("No frequency map for 'compound_strategy_pattern' in non-training mode.")
            
    if training_mode and encoders_path:
        import joblib
        Path(encoders_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(encoders, encoders_path)
        logger.info(f"Saved encoders to {encoders_path}")
        
    logger.info("Categorical variables handled.")
    return df_cat, encoders if training_mode else None


def normalize_features(df: pd.DataFrame, training_mode: bool = True, scaler_path: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[RobustScaler]]:
    """
    Normalizes numerical features using RobustScaler.
    Saves/loads scaler if path is provided.
    """
    logger.info(f"Normalizing features. Training mode: {training_mode}")
    df_norm = df.copy()

    # Define columns to exclude from normalization
    # These include IDs, raw categoricals, targets, and already discrete/boolean features
    exclude_from_scaling = [
        'Year', 'RaceID', 'DriverID', 'DriverRaceID', 'GlobalLapID', 'LapNumber', 
        'Stint', 'Position', # Discrete numericals often not scaled or scaled differently
        'IsFreshTire', 'Rainfall', # Booleans
        'tire_change_next_lap', 'next_tire_type', # Targets
        # Raw categorical columns (before encoding)
        'Compound', 'Team', 'Location', 'Driver', 'GranPrix', 'compound_strategy_pattern',
        # Pit time columns if they are datetimes or specific objects
        'PitInTime', 'PitOutTime' 
    ]
    
    # Also exclude columns that are already encoded (they are discrete int)
    encoded_cols = [col for col in df_norm.columns if col.endswith('_encoded')]
    exclude_from_scaling.extend(encoded_cols)
    # Also exclude frequency encoded columns
    freq_encoded_cols = [col for col in df_norm.columns if col.endswith('_freq')]
    exclude_from_scaling.extend(freq_encoded_cols)


    numeric_cols_to_scale = [
        col for col in df_norm.columns 
        if df_norm[col].dtype in ['float64', 'int64', 'float32', 'int32'] 
        and col not in exclude_from_scaling
    ]
    
    # Ensure all selected columns are indeed numeric and do not have issues
    valid_numeric_cols = []
    for col in numeric_cols_to_scale:
        try:
            pd.to_numeric(df_norm[col])
            valid_numeric_cols.append(col)
        except ValueError:
            logger.warning(f"Column {col} could not be converted to numeric. Skipping from scaling.")
    numeric_cols_to_scale = valid_numeric_cols
    
    if not numeric_cols_to_scale:
        logger.warning("No numeric columns found to scale.")
        return df_norm, None

    scaler = None
    if training_mode:
        scaler = RobustScaler()
        df_norm[numeric_cols_to_scale] = scaler.fit_transform(df_norm[numeric_cols_to_scale])
        logger.info(f"RobustScaler fitted and applied to {len(numeric_cols_to_scale)} features.")
        if scaler_path:
            import joblib
            Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, scaler_path)
            logger.info(f"Saved scaler to {scaler_path}")
    else:
        if scaler_path and Path(scaler_path).exists():
            import joblib
            scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")
            df_norm[numeric_cols_to_scale] = scaler.transform(df_norm[numeric_cols_to_scale])
            logger.info(f"Applied loaded RobustScaler to {len(numeric_cols_to_scale)} features.")
        else:
            logger.error(f"Scaler path {scaler_path} not found for non-training mode. Features not scaled.")
            return df_norm, None # Or raise error
            
    return df_norm, scaler

def handle_missing_values(df: pd.DataFrame, training_mode: bool = True, imputer_values_path: Optional[str] = None) -> pd.DataFrame:
    """
    Handles missing values with appropriate strategies.
    In training_mode, calculates imputation values. Otherwise, uses provided ones.
    """
    logger.info(f"Handling missing values. Training mode: {training_mode}")
    df_clean = df.copy()
    
    # Define imputation strategies for columns
    # Using ffill for sequentially dependent data, median for others to be robust to outliers.
    # Zero for rates/trends where NaN might mean "no change" or "not applicable yet".
    
    # Columns for forward fill (grouped by DriverRaceID to respect sequence)
    ffill_cols = ['TyreLife', 'Stint', 'Compound', 'AirTemp', 'TrackTemp', 'Humidity', 'WindSpeed']
    
    # Columns for median imputation (calculated from training data)
    median_cols = ['LapTime', 'TimeDeltaToDriverAhead', 'TimeDeltaToDriverBehind']
    
    # Columns for zero imputation
    zero_fill_cols = ['tire_degradation_rate', 'laptime_trend_3', 'delta_ahead_trend']

    imputation_values = {}

    if training_mode:
        # Calculate medians from the current (training) dataframe
        for col in median_cols:
            if col in df_clean.columns:
                median_val = df_clean[col].median()
                # Convert timedelta medians to seconds for JSON serialization
                if pd.api.types.is_timedelta64_dtype(df_clean[col].dtype) or isinstance(median_val, pd.Timedelta):
                    imputation_values[col] = median_val.total_seconds()
                    # Also convert the column in df_clean to seconds if it's a median_col and timedelta
                    # This ensures consistency if we impute with seconds.
                    # However, let's only convert for storage, and impute into original dtype if possible,
                    # or decide to convert these columns to seconds permanently earlier.
                    # For now, just store seconds. Imputation will handle type.
                else:
                    imputation_values[col] = median_val
        if imputer_values_path:
            import json
            Path(imputer_values_path).parent.mkdir(parents=True, exist_ok=True)
            # Convert all numpy types to native Python types for JSON
            for key, value in imputation_values.items():
                if isinstance(value, (np.generic, pd.Timedelta)): # np.generic covers numpy int, float, bool
                    if isinstance(value, pd.Timedelta):
                         imputation_values[key] = value.total_seconds()
                    else:
                         imputation_values[key] = value.item()

            with open(imputer_values_path, 'w') as f:
                json.dump(imputation_values, f)
            logger.info(f"Saved imputation values to {imputer_values_path}")
    else:
        if imputer_values_path and Path(imputer_values_path).exists():
            import json
            with open(imputer_values_path, 'r') as f:
                imputation_values = json.load(f)
            logger.info(f"Loaded imputation values from {imputer_values_path}")
        else:
            logger.error(f"Imputation values path {imputer_values_path} not found for non-training mode.")
            # Fallback: calculate medians from current data, though not ideal for test set
            for col in median_cols:
                if col in df_clean.columns:
                    imputation_values[col] = df_clean[col].median()
            logger.warning("Using medians from current (non-training) data as fallback for imputation.")

    # Apply imputation
    for col in ffill_cols:
        if col in df_clean.columns:
            if 'DriverRaceID' in df_clean.columns:
                 df_clean[col] = df_clean.groupby('DriverRaceID')[col].ffill()
            else: # Fallback if no DriverRaceID
                 df_clean[col] = df_clean[col].ffill()
            df_clean[col] = df_clean[col].bfill() # Backfill any remaining NaNs at the start of groups

    for col in median_cols:
        if col in df_clean.columns and col in imputation_values:
            fill_value = imputation_values[col]
            # If original column was timedelta and fill_value is now float (seconds),
            # we might need to convert fill_value back to timedelta before filling,
            # or convert the column to float seconds permanently.
            # For simplicity, if the column is timedelta, we assume it should be filled with a compatible type.
            # The stored imputation_values are already float (seconds) for timedelta columns.
            if pd.api.types.is_timedelta64_dtype(df_clean[col].dtype):
                # Convert the column to numeric (seconds) before filling with numeric median
                # This effectively changes the dtype of these columns to float.
                logger.info(f"Column {col} is timedelta. Converting to total_seconds() before median imputation.")
                df_clean[col] = df_clean[col].dt.total_seconds()
                df_clean[col] = df_clean[col].fillna(fill_value) # fill_value is already in seconds
            else:
                df_clean[col] = df_clean[col].fillna(fill_value)
        elif col in df_clean.columns: # Fallback if value not found (e.g. new column)
            # This fallback should ideally not be hit if imputer_values_path is always present in non-training
            fallback_median = df_clean[col].median()
            if pd.api.types.is_timedelta64_dtype(df_clean[col].dtype) or isinstance(fallback_median, pd.Timedelta):
                 df_clean[col] = df_clean[col].dt.total_seconds()
                 df_clean[col] = df_clean[col].fillna(fallback_median.total_seconds())
            else:
                 df_clean[col] = df_clean[col].fillna(fallback_median)
            logger.warning(f"Used fallback median for column {col} during imputation.")


    for col in zero_fill_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)
            
    # Final check for any remaining NaNs
    remaining_na = df_clean.isnull().sum().sum()
    if remaining_na > 0:
        logger.warning(f"Remaining NaNs after imputation: {remaining_na}. Consider a final catch-all fill (e.g., with 0 or median).")
        # Example catch-all: df_clean = df_clean.fillna(0)
    else:
        logger.info("Missing values handled successfully.")
        
    return df_clean

# Example of how to run the feature engineering pipeline
if __name__ == '__main__':
    # This is a placeholder for actual pipeline execution.
    # You would typically load data, then call these functions sequentially.
    
    # Create a dummy DataFrame for demonstration
    data = {
        'RaceID': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
        'DriverRaceID': ['1_A', '1_A', '1_A', '1_B', '1_B', '1_B', '2_C', '2_C', '2_C', '2_C'],
        'LapNumber': [1, 2, 3, 1, 2, 3, 1, 2, 3, 4],
        'Stint': [1, 1, 2, 1, 1, 1, 1, 1, 2, 2],
        'Compound': ['SOFT', 'SOFT', 'MEDIUM', 'HARD', 'HARD', 'HARD', 'MEDIUM', 'MEDIUM', 'SOFT', 'SOFT'],
        'TyreLife': [1, 2, 1, 1, 2, 3, 1, 2, 1, 2],
        'Position': [1, 2, 1, 3, 2, 2, 1, 1, 2, 1],
        'LapTime': [90.1, 90.5, 90.3, 91.0, 90.9, 90.8, 89.5, 89.4, 89.8, 89.7],
        'TimeDeltaToDriverAhead': [np.nan, 0.5, 1.2, np.nan, 0.3, 0.6, np.nan, 0.1, 2.0, 0.5],
        'TimeDeltaToDriverBehind': [0.8, 0.4, np.nan, 1.0, 0.7, np.nan, 0.5, 0.8, np.nan, 1.1],
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

    logger.info("--- Running Feature Engineering Demo ---")
    
    # 1. Create target variables
    df_processed = create_target_variables(sample_df.copy())
    
    # 2. Create temporal features
    df_processed = create_temporal_features(df_processed)
    
    # 3. Create performance features
    df_processed = create_performance_features(df_processed)
    
    # 4. Create weather features
    df_processed = create_weather_features(df_processed)
    
    # 5. Create domain knowledge features
    df_processed = create_domain_knowledge_features(df_processed)
    
    # 6. Handle missing values (in training mode for demo)
    #    Paths for encoders, scalers, imputers would be specified in a real pipeline
    encoders_demo_path = "F1/local/drive/artifacts/encoders.pkl"
    scaler_demo_path = "F1/local/drive/artifacts/scaler.pkl"
    imputer_demo_path = "F1/local/drive/artifacts/imputer_values.json"

    df_processed = handle_missing_values(df_processed, training_mode=True, imputer_values_path=imputer_demo_path)
    
    # 7. Handle categorical variables
    df_processed, _ = handle_categorical_variables(df_processed, training_mode=True, encoders_path=encoders_demo_path)
    
    # 8. Normalize features
    df_processed, _ = normalize_features(df_processed, training_mode=True, scaler_path=scaler_demo_path)
    
    logger.info("--- Feature Engineering Demo Completed ---")
    logger.info("Processed DataFrame head:")
    logger.info(df_processed.head())
    logger.info("Processed DataFrame info:")
    df_processed.info()

    # Further steps would involve splitting data, creating sequences for RNN, etc.
    # These are part of the broader preprocessing pipeline.

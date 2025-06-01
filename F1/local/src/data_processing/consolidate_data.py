"""
Data Consolidation Script for F1/local/
========================================

This script consolidates all individual F1 race parquet files from the
F1/local/drive/raw_data/ directory into a single dataset for further processing
and neural network training.

Functionality:
- Reads all .parquet files from F1/local/drive/raw_data/.
- Validates and cleans data, focusing on essential identifiers.
- Adds consolidation metadata (RaceID, DriverRaceID, GlobalLapID).
- Saves the unified dataset to F1/local/drive/processed_data/dataset.parquet.
- Generates a summary report.

Adapted from Vincenzo/dataset/data_consolidation.py.
"""

import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any
import warnings

# Configure logging
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "data_consolidation.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_paths() -> Tuple[Path, Path, Path]:
    """
    Configures paths for data input and output.
    
    Returns:
        Tuple[Path, Path, Path]: (base_project_dir, input_dir, output_dir)
    """
    # F1/local/src/data_processing/consolidate_data.py -> F1/local/
    base_project_dir = Path(__file__).resolve().parent.parent.parent
    input_dir = base_project_dir / "drive" / "raw_data"
    output_dir = base_project_dir / "drive" / "processed_data"
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
    
    return base_project_dir, input_dir, output_dir

def get_parquet_files(input_dir: Path) -> List[Path]:
    """
    Finds all .parquet files in the input directory.
    
    Args:
        input_dir (Path): Directory containing the parquet files.
        
    Returns:
        List[Path]: List of parquet files, sorted by name.
    """
    # Recursively search for .parquet files in all subdirectories
    parquet_files = sorted(list(input_dir.glob("**/*.parquet")))
    
    if not parquet_files:
        logger.warning(f"No .parquet files found recursively in {input_dir}")
        # Not raising an error immediately, consolidate_data will handle empty list
    else:
        logger.info(f"Found {len(parquet_files)} parquet files in {input_dir}:")
        for file in parquet_files[:5]: # Show first 5
            logger.info(f"  - {file.name}")
        if len(parquet_files) > 5:
            logger.info(f"  ... and {len(parquet_files) - 5} more files.")
            
    return parquet_files

def validate_dataframe(df: pd.DataFrame, filename: str) -> Dict[str, Any]:
    """
    Validates a DataFrame and collects statistics.
    Uses 'DriverNumber' and 'EventName' as per F1/local/ configuration.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        filename (str): Name of the file for logging.
        
    Returns:
        Dict[str, Any]: Validation statistics.
    """
    stats: Dict[str, Any] = {
        'filename': filename,
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'n_drivers': df['DriverNumber'].nunique() if 'DriverNumber' in df.columns else 0,
        'n_laps': df['LapNumber'].nunique() if 'LapNumber' in df.columns else 0,
        'missing_values': df.isnull().sum().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
        'years': df['Year'].unique().tolist() if 'Year' in df.columns else [],
        'event_names': df['EventName'].unique().tolist() if 'EventName' in df.columns else []
    }
    
    issues = []
    
    # Check essential columns (adapted from original and data_extraction_config.yaml)
    essential_columns = ['Year', 'DriverNumber', 'LapNumber', 'Position', 'LapTime', 'EventName']
    missing_cols = [col for col in essential_columns if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing essential columns: {missing_cols}")
    
    # Check for anomalous negative or zero values in key numeric columns
    if 'LapNumber' in df.columns and (df['LapNumber'] <= 0).any():
        issues.append("LapNumber values <= 0 found.")
    
    if 'Position' in df.columns and (df['Position'] <= 0).any():
        issues.append("Position values <= 0 found.")
        
    # Check for duplicate entries based on a unique key for a lap
    # Using Year, EventName, DriverNumber, LapNumber
    id_cols = ['Year', 'EventName', 'DriverNumber', 'LapNumber']
    if all(col in df.columns for col in id_cols):
        # Drop NaN in id_cols before checking duplicates to avoid errors with groupby
        df_no_nan_ids = df.dropna(subset=id_cols)
        if not df_no_nan_ids.empty:
            duplicates = df_no_nan_ids.duplicated(subset=id_cols, keep=False)
            if duplicates.any():
                n_duplicates = duplicates.sum()
                issues.append(f"{n_duplicates} duplicate rows found based on Year, EventName, DriverNumber, LapNumber.")
    else:
        issues.append(f"One or more ID columns ({', '.join(id_cols)}) not present for duplicate check.")

    stats['issues'] = issues
    
    if issues:
        logger.warning(f"Issues found in {filename}: {'; '.join(issues)}")
    
    return stats

def clean_dataframe(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """
    Cleans a DataFrame by applying standard corrections.
    Uses 'DriverNumber' and 'EventName'.
    
    Args:
        df (pd.DataFrame): DataFrame to clean.
        filename (str): Name of the file for logging.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    original_rows = len(df_clean)
    
    # Essential columns for non-null checks
    # 'EventName' is also essential for creating unique IDs later.
    key_cols_for_na_check = ['LapNumber', 'DriverNumber', 'Year', 'EventName', 'LapTime']
    existing_key_cols = [col for col in key_cols_for_na_check if col in df_clean.columns]
    
    if existing_key_cols:
        df_clean = df_clean.dropna(subset=existing_key_cols)
        logger.debug(f"[{filename}] Rows after dropping NA in key columns ({', '.join(existing_key_cols)}): {len(df_clean)}")

    # Convert types appropriately
    if 'LapNumber' in df_clean.columns:
        df_clean['LapNumber'] = pd.to_numeric(df_clean['LapNumber'], errors='coerce').astype('Int64') # Allow NA
    if 'DriverNumber' in df_clean.columns:
        df_clean['DriverNumber'] = pd.to_numeric(df_clean['DriverNumber'], errors='coerce').astype('Int64')
    if 'Position' in df_clean.columns:
        df_clean['Position'] = pd.to_numeric(df_clean['Position'], errors='coerce').astype('Int64')
    if 'Year' in df_clean.columns:
        df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce').astype('Int64')
    # LapTime is often timedelta, ensure it's handled if it's string or numeric
    if 'LapTime' in df_clean.columns and not pd.api.types.is_timedelta64_dtype(df_clean['LapTime']):
        try:
            df_clean['LapTime'] = pd.to_timedelta(df_clean['LapTime'], errors='coerce')
            logger.debug(f"[{filename}] Converted LapTime to timedelta.")
        except Exception as e:
            logger.warning(f"[{filename}] Could not convert LapTime to timedelta: {e}. It might remain as is or become NaT.")
            
    # Re-drop NA for columns that might have become NA after type conversion
    df_clean = df_clean.dropna(subset=[col for col in ['LapNumber', 'DriverNumber', 'Year'] if col in df_clean.columns])
    logger.debug(f"[{filename}] Rows after NA drop post-type conversion: {len(df_clean)}")

    # Remove evident anomalous values
    if 'LapNumber' in df_clean.columns:
        df_clean = df_clean[df_clean['LapNumber'] > 0]
        df_clean = df_clean[df_clean['LapNumber'] <= 100] # Reasonable max for F1
    if 'Position' in df_clean.columns:
        df_clean = df_clean[df_clean['Position'] > 0]
        df_clean = df_clean[df_clean['Position'] <= 30] # Reasonable max for F1 (allowing for more starters historically)
    logger.debug(f"[{filename}] Rows after anomalous value removal: {len(df_clean)}")

    # Remove exact duplicates based on all columns
    df_clean = df_clean.drop_duplicates()
    logger.debug(f"[{filename}] Rows after dropping exact duplicates: {len(df_clean)}")
    
    rows_removed = original_rows - len(df_clean)
    if rows_removed > 0:
        logger.info(f"[{filename}] Removed {rows_removed} rows during cleaning ({rows_removed/original_rows*100:.1f}% of original).")
    
    return df_clean

def add_consolidation_metadata(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """
    Adds useful metadata for consolidation.
    Uses 'DriverNumber' and 'EventName'.
    
    Args:
        df (pd.DataFrame): Original DataFrame.
        filename (str): Name of the file for logging.
        
    Returns:
        pd.DataFrame: DataFrame with additional metadata.
    """
    df_meta = df.copy()
    cols_added = 0

    id_col_year = 'Year'
    id_col_event = 'EventName'
    id_col_driver = 'DriverNumber'
    id_col_lap = 'LapNumber'

    # Ensure necessary columns are strings for concatenation, handling potential NAs from Int64
    str_cols_for_id = {}
    if id_col_year in df_meta.columns: str_cols_for_id[id_col_year] = df_meta[id_col_year].astype(str)
    if id_col_event in df_meta.columns: str_cols_for_id[id_col_event] = df_meta[id_col_event].astype(str).str.replace(' ', '_').str.upper()
    if id_col_driver in df_meta.columns: str_cols_for_id[id_col_driver] = df_meta[id_col_driver].astype(str)
    if id_col_lap in df_meta.columns: str_cols_for_id[id_col_lap] = df_meta[id_col_lap].astype(str)

    # Unique RaceID
    if id_col_year in str_cols_for_id and id_col_event in str_cols_for_id:
        df_meta['RaceID'] = str_cols_for_id[id_col_year] + "_" + str_cols_for_id[id_col_event]
        cols_added +=1
    
    # Unique DriverRaceID
    if id_col_year in str_cols_for_id and id_col_driver in str_cols_for_id and id_col_event in str_cols_for_id:
        df_meta['DriverRaceID'] = str_cols_for_id[id_col_year] + "_" + \
                                  str_cols_for_id[id_col_driver] + "_" + \
                                  str_cols_for_id[id_col_event]
        cols_added +=1

    # Unique GlobalLapID
    if id_col_year in str_cols_for_id and id_col_driver in str_cols_for_id and \
       id_col_event in str_cols_for_id and id_col_lap in str_cols_for_id:
        df_meta['GlobalLapID'] = str_cols_for_id[id_col_year] + "_" + \
                                 str_cols_for_id[id_col_driver] + "_" + \
                                 str_cols_for_id[id_col_event] + "_" + \
                                 str_cols_for_id[id_col_lap]
        cols_added +=1
    
    if cols_added > 0:
        logger.info(f"[{filename}] Added {cols_added} metadata columns.")
    else:
        logger.warning(f"[{filename}] Could not add metadata columns. Check for missing: Year, EventName, DriverNumber, LapNumber.")

    return df_meta

def consolidate_data(input_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """
    Main function to consolidate all parquet files.
    
    Args:
        input_dir (Path): Directory with input parquet files.
        output_dir (Path): Directory for consolidated output.
        
    Returns:
        Dict[str, Any]: Consolidation statistics.
    """
    logger.info("=== Starting F1 Dataset Consolidation ===")
    
    parquet_files = get_parquet_files(input_dir)
    
    if not parquet_files:
        logger.error(f"No parquet files found in {input_dir}. Aborting consolidation.")
        raise FileNotFoundError(f"No parquet files to process in {input_dir}.")
        
    dataframes: List[pd.DataFrame] = []
    file_stats_list: List[Dict[str, Any]] = []
    errors_list: List[str] = []
    
    for i, file_path in enumerate(parquet_files):
        logger.info(f"Processing file {i+1}/{len(parquet_files)}: {file_path.name}")
        try:
            df = pd.read_parquet(file_path)
            logger.debug(f"[{file_path.name}] Read {len(df)} rows, {len(df.columns)} columns.")
            
            stats = validate_dataframe(df, file_path.name)
            file_stats_list.append(stats)
            
            if stats['issues']:
                 logger.warning(f"[{file_path.name}] Validation issues: {stats['issues']}")

            df_clean = clean_dataframe(df, file_path.name)
            
            if df_clean.empty:
                logger.warning(f"[{file_path.name}] DataFrame became empty after cleaning. Skipping this file.")
                errors_list.append(f"File {file_path.name} resulted in empty DataFrame after cleaning.")
                continue

            df_final = add_consolidation_metadata(df_clean, file_path.name)
            dataframes.append(df_final)
            logger.info(f"  (+) Successfully processed {file_path.name}: {len(df_final)} rows, {len(df_final.columns)} columns.")
            
        except Exception as e:
            error_msg = f"Error processing {file_path.name}: {e}"
            logger.error(error_msg, exc_info=True)
            errors_list.append(error_msg)
            continue # Skip to next file
            
    if not dataframes:
        logger.error("No DataFrames were successfully processed. Cannot consolidate.")
        # Create an empty stats dictionary or handle as appropriate
        return {
            'input_files_processed': 0,
            'input_files_errors': len(errors_list),
            'total_rows': 0,
            'total_columns': 0,
            'errors': errors_list,
            'file_statistics': file_stats_list,
            'output_file_size_mb': 0,
            'memory_usage_mb': 0,
            'unique_years': [],
            'unique_drivers': 0,
            'unique_races': 0,
        }

    logger.info(f"Concatenating {len(dataframes)} DataFrames...")
    try:
        consolidated_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Consolidated dataset: {len(consolidated_df)} rows, {len(consolidated_df.columns)} columns.")
    except Exception as e:
        logger.error(f"Error during DataFrame concatenation: {e}", exc_info=True)
        raise
        
    logger.info("Final validation of the consolidated dataset...")
    final_stats = validate_dataframe(consolidated_df, "consolidated_dataset")
    
    output_file = output_dir / "dataset.parquet"
    logger.info(f"Saving consolidated dataset to: {output_file}")
    try:
        consolidated_df.to_parquet(output_file, index=False, compression='snappy')
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"  (+) Dataset saved: {file_size_mb:.2f} MB")
    except Exception as e:
        logger.error(f"Error during saving of consolidated dataset: {e}", exc_info=True)
        raise

    consolidation_summary_stats = {
        'input_files_processed': len(dataframes),
        'input_files_errors': len(errors_list),
        'total_rows': len(consolidated_df),
        'total_columns': len(consolidated_df.columns),
        'unique_years': sorted(consolidated_df['Year'].unique().tolist()) if 'Year' in consolidated_df.columns and consolidated_df['Year'].notna().any() else [],
        'unique_drivers': consolidated_df['DriverNumber'].nunique() if 'DriverNumber' in consolidated_df.columns else 0,
        'unique_races': consolidated_df['RaceID'].nunique() if 'RaceID' in consolidated_df.columns else 0,
        'memory_usage_mb': final_stats['memory_usage_mb'],
        'output_file_size_mb': file_size_mb,
        'errors': errors_list,
        'file_statistics': file_stats_list
    }
    
    return consolidation_summary_stats

def generate_summary_report(stats: Dict[str, Any], output_dir: Path) -> None:
    """
    Generates a summary report of the consolidation process.
    
    Args:
        stats (Dict[str, Any]): Consolidation statistics.
        output_dir (Path): Directory to save the report.
    """
    report_file = output_dir / "consolidation_report.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("F1 DATASET CONSOLIDATION REPORT (F1/local/)\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("GENERAL SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Input Parquet Files Processed Successfully: {stats.get('input_files_processed', 0)}\n")
        f.write(f"Input Parquet Files with Errors/Skipped: {stats.get('input_files_errors', 0)}\n")
        f.write(f"Total Rows in Consolidated Dataset: {stats.get('total_rows', 0):,}\n")
        f.write(f"Total Columns in Consolidated Dataset: {stats.get('total_columns', 0)}\n")
        
        unique_years_list = stats.get('unique_years', [])
        if unique_years_list: # Check if list is not empty
             f.write(f"Years Covered: {', '.join(map(str, unique_years_list))}\n")
        else:
             f.write("Years Covered: N/A (No data or Year column missing/empty)\n")

        f.write(f"Unique Drivers (DriverNumber): {stats.get('unique_drivers', 0)}\n")
        f.write(f"Unique Races (RaceID): {stats.get('unique_races', 0)}\n")
        f.write(f"Consolidated DataFrame Memory Usage: {stats.get('memory_usage_mb', 0):.2f} MB\n")
        f.write(f"Output Parquet File Size: {stats.get('output_file_size_mb', 0):.2f} MB\n\n")
        
        errors_list = stats.get('errors', [])
        if errors_list:
            f.write("ERRORS/WARNINGS DURING PROCESSING\n")
            f.write("-" * 20 + "\n")
            for error in errors_list:
                f.write(f"- {error}\n")
            f.write("\n")
        
        file_stats_list = stats.get('file_statistics', [])
        if file_stats_list:
            f.write("STATISTICS PER INPUT FILE\n")
            f.write("-" * 20 + "\n")
            for file_stat in file_stats_list:
                f.write(f"\nFile: {file_stat.get('filename', 'N/A')}\n")
                f.write(f"  Rows Read: {file_stat.get('n_rows', 0):,}\n")
                f.write(f"  Columns Read: {file_stat.get('n_columns', 0)}\n")
                f.write(f"  Unique Drivers (DriverNumber): {file_stat.get('n_drivers', 0)}\n")
                # Laps might be LapNumber.nunique()
                f.write(f"  Unique Lap Numbers: {file_stat.get('n_laps', 0)}\n") 
                f.write(f"  Total Missing Values: {file_stat.get('missing_values', 0):,}\n")
                
                event_names_list = file_stat.get('event_names', [])
                if event_names_list:
                    f.write(f"  Event Names: {', '.join(map(str, event_names_list))}\n")
                else:
                    f.write(f"  Event Names: N/A\n")

                file_issues = file_stat.get('issues', [])
                if file_issues:
                    f.write(f"  Validation Issues: {'; '.join(file_issues)}\n")
    
    logger.info(f"Consolidation summary report saved to: {report_file}")

def main():
    """Main script function."""
    logger.info("--- Script execution started: Data Consolidation ---")
    try:
        _, input_dir, output_dir = setup_paths()
        
        # Suppress pandas PerformanceWarning for operations like concat on many small DFs
        warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
        
        consolidation_stats = consolidate_data(input_dir, output_dir)
        generate_summary_report(consolidation_stats, output_dir)
        
        logger.info("=" * 60)
        if consolidation_stats.get('total_rows', 0) > 0 :
            logger.info("CONSOLIDATION COMPLETED SUCCESSFULLY!")
            logger.info(f"Final consolidated dataset: {consolidation_stats['total_rows']:,} rows.")
            logger.info(f"Output file: {output_dir / 'dataset.parquet'}")
        elif consolidation_stats.get('input_files_errors',0) > 0 and consolidation_stats.get('input_files_processed',0) == 0 :
             logger.error("CONSOLIDATION FAILED. No files were processed successfully.")
        else:
            logger.warning("CONSOLIDATION COMPLETED, BUT THE RESULTING DATASET IS EMPTY or NO FILES PROCESSED.")
            logger.warning(f"Please check logs at {LOG_FILE} and report at {output_dir / 'consolidation_report.txt'}")

        logger.info("=" * 60)
        
    except FileNotFoundError as fnf_error:
        logger.error(f"CRITICAL FILE NOT FOUND ERROR: {fnf_error}", exc_info=True)
    except Exception as e:
        logger.error(f"CRITICAL UNHANDLED ERROR in main: {e}", exc_info=True)
        # No raise here to allow logging to complete if possible
    finally:
        logger.info("--- Script execution finished: Data Consolidation ---")

if __name__ == "__main__":
    main()

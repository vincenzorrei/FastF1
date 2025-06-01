"""
Data Consolidation Script for F1/colab/
========================================

This script consolidates all individual F1 race parquet files from the
Google Drive `raw_data` directory (configured via colab_path_config.yaml)
into a single dataset for further processing.

Functionality:
- Reads all .parquet files from the configured Google Drive `raw_data` path.
- Validates and cleans data.
- Adds consolidation metadata (RaceID, DriverRaceID, GlobalLapID).
- Saves the unified dataset to the configured Google Drive `processed_data` path.
- Generates a summary report to the configured Google Drive `logs` or `artifacts` path.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any
import warnings
import yaml # For loading colab_path_config

# --- Colab Path Configuration ---
def load_colab_paths():
    """Loads and resolves paths from F1/colab/configs/colab_path_config.yaml."""
    path_config_file = Path(__file__).parent.parent / "configs" / "colab_path_config.yaml"
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
if COLAB_PATHS and COLAB_PATHS.get('data_consolidation_log_file'):
    LOG_FILE = COLAB_PATHS['data_consolidation_log_file']
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
else:
    # Fallback if paths are not loaded correctly, though script should ideally halt earlier
    fallback_log_dir = Path(__file__).resolve().parent.parent.parent / "logs_colab_fallback"
    fallback_log_dir.mkdir(parents=True, exist_ok=True)
    LOG_FILE = fallback_log_dir / "colab_data_consolidation_fallback.log"
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

def get_parquet_files(input_dir: Path) -> List[Path]:
    """
    Finds all .parquet files in the input directory.
    """
    parquet_files = sorted(list(input_dir.glob("**/*.parquet")))
    if not parquet_files:
        logger.warning(f"No .parquet files found recursively in {input_dir}")
    else:
        logger.info(f"Found {len(parquet_files)} parquet files in {input_dir}.")
    return parquet_files

def validate_dataframe(df: pd.DataFrame, filename: str) -> Dict[str, Any]:
    """
    Validates a DataFrame and collects statistics.
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
    essential_columns = ['Year', 'DriverNumber', 'LapNumber', 'Position', 'LapTime', 'EventName']
    missing_cols = [col for col in essential_columns if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing essential columns: {missing_cols}")
    if 'LapNumber' in df.columns and (df['LapNumber'] <= 0).any():
        issues.append("LapNumber values <= 0 found.")
    if 'Position' in df.columns and (df['Position'] <= 0).any():
        issues.append("Position values <= 0 found.")
    id_cols = ['Year', 'EventName', 'DriverNumber', 'LapNumber']
    if all(col in df.columns for col in id_cols):
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
    """
    df_clean = df.copy()
    original_rows = len(df_clean)
    key_cols_for_na_check = ['LapNumber', 'DriverNumber', 'Year', 'EventName', 'LapTime']
    existing_key_cols = [col for col in key_cols_for_na_check if col in df_clean.columns]
    if existing_key_cols:
        df_clean = df_clean.dropna(subset=existing_key_cols)
    if 'LapNumber' in df_clean.columns:
        df_clean['LapNumber'] = pd.to_numeric(df_clean['LapNumber'], errors='coerce').astype('Int64')
    if 'DriverNumber' in df_clean.columns:
        df_clean['DriverNumber'] = pd.to_numeric(df_clean['DriverNumber'], errors='coerce').astype('Int64')
    if 'Position' in df_clean.columns:
        df_clean['Position'] = pd.to_numeric(df_clean['Position'], errors='coerce').astype('Int64')
    if 'Year' in df_clean.columns:
        df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce').astype('Int64')
    if 'LapTime' in df_clean.columns and not pd.api.types.is_timedelta64_dtype(df_clean['LapTime']):
        try:
            df_clean['LapTime'] = pd.to_timedelta(df_clean['LapTime'], errors='coerce')
        except Exception as e:
            logger.warning(f"[{filename}] Could not convert LapTime to timedelta: {e}.")
    df_clean = df_clean.dropna(subset=[col for col in ['LapNumber', 'DriverNumber', 'Year'] if col in df_clean.columns])
    if 'LapNumber' in df_clean.columns:
        df_clean = df_clean[df_clean['LapNumber'] > 0]
        df_clean = df_clean[df_clean['LapNumber'] <= 100]
    if 'Position' in df_clean.columns:
        df_clean = df_clean[df_clean['Position'] > 0]
        df_clean = df_clean[df_clean['Position'] <= 30]
    df_clean = df_clean.drop_duplicates()
    rows_removed = original_rows - len(df_clean)
    if rows_removed > 0:
        logger.info(f"[{filename}] Removed {rows_removed} rows during cleaning ({rows_removed/original_rows*100:.1f}% of original).")
    return df_clean

def add_consolidation_metadata(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """
    Adds useful metadata for consolidation.
    """
    df_meta = df.copy()
    cols_added = 0
    id_col_year = 'Year'
    id_col_event = 'EventName'
    id_col_driver = 'DriverNumber'
    id_col_lap = 'LapNumber'
    str_cols_for_id = {}
    if id_col_year in df_meta.columns: str_cols_for_id[id_col_year] = df_meta[id_col_year].astype(str)
    if id_col_event in df_meta.columns: str_cols_for_id[id_col_event] = df_meta[id_col_event].astype(str).str.replace(' ', '_').str.upper()
    if id_col_driver in df_meta.columns: str_cols_for_id[id_col_driver] = df_meta[id_col_driver].astype(str)
    if id_col_lap in df_meta.columns: str_cols_for_id[id_col_lap] = df_meta[id_col_lap].astype(str)
    if id_col_year in str_cols_for_id and id_col_event in str_cols_for_id:
        df_meta['RaceID'] = str_cols_for_id[id_col_year] + "_" + str_cols_for_id[id_col_event]
        cols_added +=1
    if id_col_year in str_cols_for_id and id_col_driver in str_cols_for_id and id_col_event in str_cols_for_id:
        df_meta['DriverRaceID'] = str_cols_for_id[id_col_year] + "_" + str_cols_for_id[id_col_driver] + "_" + str_cols_for_id[id_col_event]
        cols_added +=1
    if id_col_year in str_cols_for_id and id_col_driver in str_cols_for_id and id_col_event in str_cols_for_id and id_col_lap in str_cols_for_id:
        df_meta['GlobalLapID'] = str_cols_for_id[id_col_year] + "_" + str_cols_for_id[id_col_driver] + "_" + str_cols_for_id[id_col_event] + "_" + str_cols_for_id[id_col_lap]
        cols_added +=1
    if cols_added > 0:
        logger.info(f"[{filename}] Added {cols_added} metadata columns.")
    else:
        logger.warning(f"[{filename}] Could not add metadata columns. Check for missing: Year, EventName, DriverNumber, LapNumber.")
    return df_meta

def consolidate_data(input_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """
    Main function to consolidate all parquet files.
    """
    logger.info("=== Starting F1 Dataset Consolidation (Colab) ===")
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
            continue
    if not dataframes:
        logger.error("No DataFrames were successfully processed. Cannot consolidate.")
        return {'input_files_processed': 0, 'input_files_errors': len(errors_list), 'total_rows': 0, 'errors': errors_list}
    logger.info(f"Concatenating {len(dataframes)} DataFrames...")
    consolidated_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Consolidated dataset: {len(consolidated_df)} rows, {len(consolidated_df.columns)} columns.")
    final_stats = validate_dataframe(consolidated_df, "consolidated_dataset")
    output_file = output_dir / "dataset.parquet"
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output_dir exists
    logger.info(f"Saving consolidated dataset to: {output_file}")
    consolidated_df.to_parquet(output_file, index=False, compression='snappy')
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"  (+) Dataset saved: {file_size_mb:.2f} MB")
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

def generate_summary_report(stats: Dict[str, Any], report_file: Path) -> None:
    """
    Generates a summary report of the consolidation process.
    """
    report_file.parent.mkdir(parents=True, exist_ok=True) # Ensure report directory exists
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("F1 DATASET CONSOLIDATION REPORT (F1/colab/)\n")
        f.write("=" * 60 + "\n\n")
        f.write("GENERAL SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Input Parquet Files Processed Successfully: {stats.get('input_files_processed', 0)}\n")
        f.write(f"Input Parquet Files with Errors/Skipped: {stats.get('input_files_errors', 0)}\n")
        f.write(f"Total Rows in Consolidated Dataset: {stats.get('total_rows', 0):,}\n")
        f.write(f"Total Columns in Consolidated Dataset: {stats.get('total_columns', 0)}\n")
        unique_years_list = stats.get('unique_years', [])
        if unique_years_list:
             f.write(f"Years Covered: {', '.join(map(str, unique_years_list))}\n")
        else:
             f.write("Years Covered: N/A\n")
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
    """Main script function for Colab."""
    logger.info("--- Script execution started: Data Consolidation (Colab) ---")
    if not COLAB_PATHS:
        logger.error("CRITICAL: Colab paths configuration not loaded. Halting.")
        return

    try:
        input_dir = COLAB_PATHS['raw_data_output_directory']
        output_dir = COLAB_PATHS['processed_data_directory']
        report_file_path = COLAB_PATHS.get('logs_directory', output_dir) / "colab_consolidation_report.txt" # Save report in logs or output_dir

        if not input_dir.exists():
            logger.error(f"Input directory for raw data not found: {input_dir}")
            raise FileNotFoundError(f"Input directory for raw data not found: {input_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        report_file_path.parent.mkdir(parents=True, exist_ok=True)

        warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
        
        consolidation_stats = consolidate_data(input_dir, output_dir)
        generate_summary_report(consolidation_stats, report_file_path)
        
        logger.info("=" * 60)
        if consolidation_stats.get('total_rows', 0) > 0 :
            logger.info("COLAB CONSOLIDATION COMPLETED SUCCESSFULLY!")
            logger.info(f"Final consolidated dataset: {consolidation_stats['total_rows']:,} rows.")
            logger.info(f"Output file: {output_dir / 'dataset.parquet'}")
        elif consolidation_stats.get('input_files_errors',0) > 0 and consolidation_stats.get('input_files_processed',0) == 0 :
             logger.error("COLAB CONSOLIDATION FAILED. No files were processed successfully.")
        else:
            logger.warning("COLAB CONSOLIDATION COMPLETED, BUT THE RESULTING DATASET IS EMPTY or NO FILES PROCESSED.")
        logger.info(f"Please check logs at {LOG_FILE} and report at {report_file_path}")
        logger.info("=" * 60)
        
    except FileNotFoundError as fnf_error:
        logger.error(f"CRITICAL FILE NOT FOUND ERROR: {fnf_error}", exc_info=True)
    except Exception as e:
        logger.error(f"CRITICAL UNHANDLED ERROR in main (Colab): {e}", exc_info=True)
    finally:
        logger.info("--- Script execution finished: Data Consolidation (Colab) ---")

if __name__ == "__main__":
    if COLAB_PATHS:
        main()
    else:
        print("CRITICAL ERROR: Colab path configuration could not be loaded at script start.")
        print("Please ensure 'F1/colab/configs/colab_path_config.yaml' is correctly set up,")
        print("especially the 'base_project_drive_path'.")

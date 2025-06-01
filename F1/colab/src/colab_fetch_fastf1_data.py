import fastf1 as ff1
import pandas as pd
import yaml
import os
import time
from datetime import datetime
import requests
from pathlib import Path # Added for path manipulation

# --- Colab Path Configuration ---
# This function will load paths from the colab_path_config.yaml
# It's a simplified version of the example in colab_path_config.yaml
def load_colab_paths():
    """Loads and resolves paths from F1/colab/configs/colab_path_config.yaml."""
    # Path to the colab_path_config.yaml relative to this script's expected location
    # Assumes this script is in F1/colab/src/
    # So, ../configs/colab_path_config.yaml
    path_config_file = Path(__file__).parent.parent / "configs" / "colab_path_config.yaml"
    
    if not path_config_file.exists():
        print(f"ERROR: Colab path config file not found at {path_config_file}")
        print("Please ensure 'F1/colab/configs/colab_path_config.yaml' exists and is correctly structured.")
        print("And that 'base_project_drive_path' within it points to your 'FASTF1' project root on Google Drive.")
        return None

    with open(path_config_file, 'r') as f:
        raw_config = yaml.safe_load(f)

    base_drive_path_str = raw_config.get('base_project_drive_path')
    if not base_drive_path_str or "path/to/your/FASTF1" in base_drive_path_str:
        print("ERROR: 'base_project_drive_path' in 'colab_path_config.yaml' is not configured or is still a placeholder.")
        print(f"Current value: {base_drive_path_str}")
        print("Please update it to point to the root of your 'FASTF1' project directory on Google Drive.")
        return None
    
    base_drive_path = Path(base_drive_path_str)
    colab_root_str = raw_config.get('colab_root', 'F1/colab') # Default if missing

    resolved_paths = {}
    for key, value in raw_config.items():
        if isinstance(value, str):
            # Simplified formatting for Colab context
            # Assumes paths in YAML are either absolute or placeholders needing base_project_drive_path
            formatted_value = value.format(
                base_project_drive_path=str(base_drive_path), # Ensure it's a string for Path()
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

# --- Data Extraction Configuration Loading (for Colab) ---
# The main data extraction config (years, columns, etc.)
# is now also loaded using the path from COLAB_PATHS
def load_data_extraction_config():
    """Loads the data extraction YAML configuration file for Colab."""
    if not COLAB_PATHS or 'data_extraction_config_colab' not in COLAB_PATHS:
        print("ERROR: Colab paths not loaded or 'data_extraction_config_colab' missing in path config.")
        return None
    
    config_path = COLAB_PATHS['data_extraction_config_colab']
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Data extraction configuration loaded successfully from {config_path}")
        return config
    except FileNotFoundError:
        print(f"ERROR: Data extraction configuration file not found at {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"ERROR: Could not parse YAML configuration from {config_path}: {e}")
        return None

# --- Logging ---
def update_download_log(log_file_path, log_entry): # log_file_path is now a Path object
    """Appends a new entry to the CSV download log."""
    log_df = pd.DataFrame([log_entry])
    try:
        # Ensure parent directory exists
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        if not log_file_path.exists():
            log_df.to_csv(log_file_path, index=False)
        else:
            log_df.to_csv(log_file_path, mode='a', header=False, index=False)
    except Exception as e:
        print(f"ERROR: Could not write to log file {log_file_path}: {e}")

# --- Main Data Extraction Logic ---
def fetch_race_data(data_config, path_config, race_info): # Added path_config
    """
    Fetches and processes data for a single race event.
    Uses data_config for extraction parameters and path_config for file locations.
    """
    year = race_info['year']
    event_name = race_info['event_name']
    session_type = race_info['session_type'] # Should always be 'R'

    download_log_file = path_config['download_log_file'] # Get from path_config

    log_entry = {
        "Year": year,
        "EventName": event_name,
        "SessionType": session_type,
        "DownloadTimestamp": datetime.now().isoformat(),
        "Status": "Initiated",
        "NaNPercentage": None,
        "FilePath": None,
        "RetriesAttempted": 0,
        "Notes": ""
    }

    print(f"\nAttempting to fetch data for: {year} {event_name} - Session {session_type}")

    if session_type != 'R':
        log_entry["Status"] = "Skipped - NonRaceSession"
        log_entry["Notes"] = f"Session type is {session_type}, not 'R'."
        update_download_log(download_log_file, log_entry)
        print(log_entry["Notes"])
        return

    retries = 0
    max_retries = data_config.get('max_retries_on_failure', 3)
    success = False

    while retries <= max_retries and not success:
        log_entry["RetriesAttempted"] = retries
        try:
            print(f"Attempt {retries + 1}/{max_retries + 1} for {year} {event_name}")
            
            session = ff1.get_session(year, event_name, session_type)
            session.load(laps=True, telemetry=False, weather=True, messages=False)
            print(f"Session loaded for {year} {event_name}")

            laps_df = session.laps
            if laps_df is None or laps_df.empty:
                log_entry["Status"] = "Failed - NoLapData"
                log_entry["Notes"] = "No lap data found in the session."
                print(log_entry["Notes"])
                raise ff1.core.DataNotLoadedError("No lap data in session")

            laps_df['Year'] = session.event['EventDate'].year
            laps_df['EventName'] = session.event['EventName']
            laps_df['Country'] = session.event['Country']
            laps_df['CircuitName'] = session.event['Location']

            laps_with_weather_df = laps_df.copy()
            weather_data = session.weather_data
            weather_cols_to_add = data_config.get('weather_columns_to_extract', ['AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp', 'WindDirection', 'WindSpeed'])


            if weather_data is not None and not weather_data.empty:
                if 'LapStartTime' in laps_with_weather_df.columns and pd.api.types.is_datetime64_any_dtype(laps_with_weather_df['LapStartTime']):
                    time_key_laps = 'LapStartTime'
                elif 'Time' in laps_with_weather_df.columns and pd.api.types.is_timedelta64_dtype(laps_with_weather_df['Time']):
                    session_start_time = session.date
                    laps_with_weather_df['AbsoluteLapTime'] = session_start_time + laps_with_weather_df['Time']
                    time_key_laps = 'AbsoluteLapTime'
                else:
                    print(f"WARNING: Suitable time column for weather merge not found in laps_df for {year} {event_name}.")
                    for col in weather_cols_to_add: 
                        laps_with_weather_df[col] = pd.NA
                    weather_data = None

                if weather_data is not None and 'Time' in weather_data.columns:
                    if pd.api.types.is_timedelta64_dtype(weather_data['Time']):
                        session_start_time = session.date
                        weather_data['Time'] = session_start_time + weather_data['Time']
                    
                    if pd.api.types.is_datetime64_any_dtype(weather_data['Time']):
                        laps_with_weather_df = laps_with_weather_df.sort_values(by=time_key_laps)
                        weather_data_sorted = weather_data.sort_values(by='Time')
                        weather_cols_for_merge = [col for col in weather_cols_to_add if col in weather_data_sorted.columns]
                        
                        if not weather_cols_for_merge:
                            print(f"WARNING: None of the desired weather columns ({weather_cols_to_add}) found in session.weather_data for {year} {event_name}.")
                            for col in weather_cols_to_add:
                                laps_with_weather_df[col] = pd.NA
                        else:
                            weather_data_subset = weather_data_sorted[weather_cols_for_merge + ['Time']].copy()
                            laps_with_weather_df = pd.merge_asof(
                                left=laps_with_weather_df,
                                right=weather_data_subset,
                                left_on=time_key_laps,
                                right_on='Time',
                                direction='nearest',
                                suffixes=('', '_weather')
                            )
                    if time_key_laps == 'AbsoluteLapTime':
                        laps_with_weather_df = laps_with_weather_df.drop(columns=['AbsoluteLapTime'])
                    if 'Time_weather' in laps_with_weather_df.columns:
                         laps_with_weather_df = laps_with_weather_df.drop(columns=['Time_weather'])
                else:
                    if weather_data is not None:
                         print(f"WARNING: session.weather_data 'Time' column is missing or not datetime for {year} {event_name}.")
                    for col in weather_cols_to_add:
                        laps_with_weather_df[col] = pd.NA
            else:
                print(f"WARNING: No session.weather_data available for {year} {event_name}. Weather columns will be NaN.")
                for col in weather_cols_to_add:
                    laps_with_weather_df[col] = pd.NA

            columns_to_extract = data_config.get('columns_to_extract', [])
            column_mapping = data_config.get('column_mapping', {'TeamName': 'Team'}) # Get mapping from config
            
            final_columns = []
            missing_cols_in_source = []
            
            for col_config_name in columns_to_extract:
                actual_col_name = column_mapping.get(col_config_name, col_config_name)
                if actual_col_name in laps_with_weather_df.columns:
                    final_columns.append(actual_col_name)
                else:
                    if col_config_name in laps_with_weather_df.columns:
                         final_columns.append(col_config_name)
                    else:
                        missing_cols_in_source.append(col_config_name)

            if missing_cols_in_source:
                print(f"WARNING: Configured columns not found in source data for {year} {event_name}: {', '.join(missing_cols_in_source)}")
            
            if not final_columns:
                log_entry["Status"] = "Failed - NoColumnsExtracted"
                log_entry["Notes"] = "None of the specified columns_to_extract could be found in the source data."
                print(log_entry["Notes"])
                raise ValueError(log_entry["Notes"])

            processed_df = laps_with_weather_df[final_columns].copy()
            
            type_conversions = data_config.get('type_conversions', {})
            for col, dtype in type_conversions.items():
                if col in processed_df.columns:
                    try:
                        if dtype == 'Int64':
                             processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').astype('Int64')
                        else:
                            processed_df[col] = processed_df[col].astype(dtype)
                    except Exception as e_type:
                        print(f"WARNING: Could not convert column '{col}' to '{dtype}' for {year} {event_name}: {e_type}")


            critical_columns_for_nan_check = data_config.get('critical_columns_for_nan_check', [])
            actual_critical_columns = [col for col in critical_columns_for_nan_check if col in processed_df.columns]
            
            if not critical_columns_for_nan_check:
                print("WARNING: 'critical_columns_for_nan_check' is empty in config. Using all columns for NaN check.")
                nan_check_columns = processed_df.columns.tolist()
            elif not actual_critical_columns:
                print(f"WARNING: None of the specified 'critical_columns_for_nan_check' were found. Using all columns for NaN check.")
                nan_check_columns = processed_df.columns.tolist()
            else:
                nan_check_columns = actual_critical_columns
            
            if not nan_check_columns:
                 nan_percentage = 0.0
            else:
                nan_percentage = processed_df[nan_check_columns].isnull().mean().mean()

            log_entry["NaNPercentage"] = round(nan_percentage, 4)
            print(f"NaN percentage for check columns ({', '.join(nan_check_columns)}): {nan_percentage:.2%}")

            if nan_percentage > data_config.get('max_nan_percentage_threshold', 0.05):
                log_entry["Status"] = "Incomplete - HighNaN"
                log_entry["Notes"] = f"NaN percentage ({nan_percentage:.2%}) exceeds threshold."
                print(log_entry["Notes"])
                raise ValueError(log_entry["Notes"])

            safe_event_name = event_name.replace(" ", "_").replace("/", "_")
            # Use raw_data_output_directory from path_config
            output_dir = path_config['raw_data_output_directory'] / str(year) / safe_event_name
            output_dir.mkdir(parents=True, exist_ok=True) # Use Path.mkdir
            
            file_event_name = session.event.get('OfficialEventName', session.event['EventName']).replace(' ', '_').replace('/', '_')
            file_name = f"{file_event_name}_{session_type}_data.parquet"
            file_path = output_dir / file_name # Use Path object for joining

            print(f"DEBUG: DataFrame info for {file_path}:")
            processed_df.info(verbose=True, show_counts=True)
            print(f"DEBUG: DataFrame head for {file_path}:")
            print(processed_df.head())
            
            processed_df.to_parquet(file_path, index=False)
            log_entry["FilePath"] = str(file_path) # Store as string in log
            log_entry["Status"] = "Success"
            log_entry["Notes"] = f"Data saved to {file_path}"
            print(log_entry["Notes"])
            success = True

        except ff1.core.DataNotLoadedError as e:
            log_entry["Status"] = "Failed - NoLapData"
            log_entry["Notes"] = str(e)
            print(f"ERROR: DataNotLoadedError for {year} {event_name}: {e}")
            retries = max_retries + 1 
        except requests.exceptions.HTTPError as e:
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
                log_entry["Status"] = "Failed - RateLimit"
                log_entry["Notes"] = f"Rate limit exceeded: {str(e)}"
            else:
                log_entry["Status"] = "Failed - HTTPError"
                status_code_info = f", Status Code: {e.response.status_code}" if hasattr(e, 'response') and e.response is not None else ""
                log_entry["Notes"] = f"HTTP error: {str(e)}{status_code_info}"
            print(f"ERROR: HTTP error for {year} {event_name}: {e}")
        except ConnectionError as e: # More generic, requests.exceptions.ConnectionError is better
            log_entry["Status"] = "Failed - ConnectionError"
            log_entry["Notes"] = str(e)
            print(f"ERROR: Connection error for {year} {event_name}: {e}")
        except ValueError as e:
            print(f"ERROR: Value error during processing for {year} {event_name}: {e}")
        except Exception as e:
            log_entry["Status"] = "Failed - UnknownError"
            log_entry["Notes"] = f"An unexpected error: {type(e).__name__} - {str(e)}"
            print(f"ERROR: An unexpected error occurred for {year} {event_name}: {e}")
        
        if success:
            break

        retries += 1
        if retries <= max_retries:
            delay = data_config.get('request_delay_seconds', 5)
            print(f"Retrying ({retries}/{max_retries}) for {year} {event_name}... waiting {delay}s")
            time.sleep(delay)
        else:
            print(f"Max retries reached for {year} {event_name}. Final status: {log_entry['Status']}")
            if log_entry["Status"] not in ["Success", "Skipped - NonRaceSession", "Failed - NoLapData"] and not log_entry["Status"].startswith("Failed - MaxRetriesReached"):
                if log_entry["Status"] == "Initiated" or "Failed -" in log_entry["Status"]:
                    log_entry["Status"] = "Failed - MaxRetriesReached"
                    current_notes = log_entry.get("Notes", "")
                    if "Max retries reached" not in current_notes:
                        log_entry["Notes"] = (current_notes + "; Max retries reached.").strip("; ")
    
    update_download_log(download_log_file, log_entry)


def main():
    """Main function to orchestrate data extraction for Colab."""
    if not COLAB_PATHS:
        print("Halting execution: Colab paths could not be loaded. Please check 'F1/colab/configs/colab_path_config.yaml'.")
        return

    data_config = load_data_extraction_config()
    if not data_config:
        print("Halting execution: Data extraction configuration could not be loaded.")
        return

    # Setup FastF1 cache using path from COLAB_PATHS
    cache_path = COLAB_PATHS.get('fastf1_cache_path_drive') # Use the Drive path
    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)
        ff1.Cache.enable_cache(cache_path)
        print(f"FastF1 cache enabled at: {cache_path} (on Google Drive)")
    else:
        print("WARNING: 'fastf1_cache_path_drive' not specified in colab_path_config.yaml. FastF1 Cache is disabled.")

    # Ensure output and log directories exist (using Path objects from COLAB_PATHS)
    raw_data_dir = COLAB_PATHS.get('raw_data_output_directory')
    if raw_data_dir:
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Ensured raw data output directory exists: {raw_data_dir}")
    else:
        print("ERROR: 'raw_data_output_directory' not in path_config. Cannot proceed.")
        return
        
    download_log_file = COLAB_PATHS.get('download_log_file')
    if download_log_file:
        download_log_file.parent.mkdir(parents=True, exist_ok=True) # Ensure log directory exists
        print(f"Ensured download log file directory exists: {download_log_file.parent}")
    else:
        print("ERROR: 'download_log_file' not specified in path_config. Logging will fail.")
        return

    years_to_process = data_config.get('years_to_fetch', [])
    if not years_to_process:
        print("No years specified in 'years_to_fetch' in the data extraction configuration.")
        return

    all_potential_races = []
    print(f"Fetching event schedules for years: {years_to_process} (Order as in config: {years_to_process})") 
    for year_val in years_to_process: 
        try:
            schedule = ff1.get_event_schedule(year_val, include_testing=False)
            for index, event in schedule.iterrows():
                if "grand prix" in event['EventName'].lower(): # Basic filter for GP events
                    all_potential_races.append({
                        'year': int(year_val),
                        'event_name': event['EventName'], 
                        'session_type': 'R'
                    })
        except Exception as e:
            print(f"ERROR: Could not fetch event schedule for year {year_val}: {e}")

    if not all_potential_races:
        print("No Grand Prix races found for the specified years based on 'grand prix' in EventName.")
        return

    processed_races_set = set()
    if download_log_file.exists(): # Check if log file exists
        try:
            log_df = pd.read_csv(download_log_file)
            if not log_df.empty:
                if all(col in log_df.columns for col in ['Year', 'EventName', 'SessionType', 'Status']):
                    successful_downloads = log_df[log_df['Status'] == 'Success']
                    for _, row in successful_downloads.iterrows():
                        processed_races_set.add((int(row['Year']), row['EventName'], row['SessionType']))
                    print(f"Found {len(processed_races_set)} successfully processed race(s) in the log: {download_log_file}")
                else:
                    print(f"WARNING: Log file {download_log_file} is missing expected columns. Cannot determine processed races.")
        except pd.errors.EmptyDataError:
            print(f"Log file {download_log_file} is empty. No previously successful downloads to skip.")
        except Exception as e:
            print(f"Error reading or parsing log file {download_log_file}: {e}. Proceeding without skipping.")

    races_to_actually_fetch = [
        race for race in all_potential_races
        if (race['year'], race['event_name'], race['session_type']) not in processed_races_set
    ]

    if not races_to_actually_fetch:
        print("All potential races already processed successfully or no new races to process.")
        return
    
    print(f"Starting data extraction for {len(races_to_actually_fetch)} race(s) out of {len(all_potential_races)} potential races...")
    
    for race_info in races_to_actually_fetch:
        fetch_race_data(data_config, COLAB_PATHS, race_info) # Pass COLAB_PATHS
        time.sleep(data_config.get('request_delay_seconds_between_events', 2)) # Renamed for clarity

if __name__ == "__main__":
    # This check is important for Colab. If COLAB_PATHS is None, main() will print errors and return.
    if COLAB_PATHS:
        main()
    else:
        print("CRITICAL ERROR: Colab path configuration could not be loaded.")
        print("Please ensure 'F1/colab/configs/colab_path_config.yaml' is correctly set up,")
        print("especially the 'base_project_drive_path' pointing to your 'FASTF1' project on Google Drive.")
        print("This script cannot run without valid paths.")

import fastf1 as ff1
import pandas as pd
import yaml
import os
import time
from datetime import datetime
import requests

# --- Configuration Loading ---
CONFIG_PATH = "F1/local/configs/data_extraction_config.yaml"

def load_config(config_path):
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded successfully from {config_path}")
        return config
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"ERROR: Could not parse YAML configuration: {e}")
        return None

# --- Logging ---
def update_download_log(log_file_path, log_entry):
    """Appends a new entry to the CSV download log."""
    log_df = pd.DataFrame([log_entry])
    try:
        if not os.path.exists(log_file_path):
            log_df.to_csv(log_file_path, index=False)
        else:
            log_df.to_csv(log_file_path, mode='a', header=False, index=False)
    except Exception as e:
        print(f"ERROR: Could not write to log file {log_file_path}: {e}")

# --- Main Data Extraction Logic ---
def fetch_race_data(config, race_info):
    """
    Fetches and processes data for a single race event.
    """
    year = race_info['year']
    event_name = race_info['event_name']
    session_type = race_info['session_type'] # Should always be 'R'

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
        update_download_log(config['download_log_file'], log_entry)
        print(log_entry["Notes"])
        return

    retries = 0
    max_retries = config.get('max_retries_on_failure', 3)
    success = False

    while retries <= max_retries and not success:
        log_entry["RetriesAttempted"] = retries
        try:
            print(f"Attempt {retries + 1}/{max_retries + 1} for {year} {event_name}")
            
            # 1. Load session
            session = ff1.get_session(year, event_name, session_type)
            session.load(laps=True, telemetry=False, weather=True, messages=False)
            print(f"Session loaded for {year} {event_name}")

            # 3. Extract laps and weather data
            laps_df = session.laps
            if laps_df is None or laps_df.empty:
                log_entry["Status"] = "Failed - NoLapData"
                log_entry["Notes"] = "No lap data found in the session."
                print(log_entry["Notes"])
                # This is a non-retryable error for this attempt, but the loop might retry.
                # However, if a session truly has no laps, retrying won't help.
                # We will log this attempt and the outer loop will decide to retry or fail.
                raise ff1.core.DataNotLoadedError("No lap data in session") # Raise specific error

            # Add session metadata to laps_df
            laps_df['Year'] = session.event['EventDate'].year
            laps_df['EventName'] = session.event['EventName']
            laps_df['Country'] = session.event['Country']
            laps_df['CircuitName'] = session.event['Location']

            laps_with_weather_df = laps_df.copy()
            
            # More robust weather data handling using session.weather_data and merge_asof
            weather_data = session.weather_data # This is a DataFrame
            
            weather_cols_to_add = ['AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp', 'WindDirection', 'WindSpeed']

            if weather_data is not None and not weather_data.empty:
                # Ensure 'Time' columns are datetime objects for merge_asof
                # Lap times are relative to session start, weather times are absolute.
                # We need a common reference, usually session start time.
                # FastF1 laps DataFrame has 'Time' (timedelta from session start) and 'LapStartTime' (datetime)
                # session.weather_data also has 'Time' (datetime)

                # Use 'LapStartTime' from laps_df if available and it's datetime
                # If laps_df['Time'] is timedelta, convert it to absolute time using session start
                if 'LapStartTime' in laps_with_weather_df.columns and pd.api.types.is_datetime64_any_dtype(laps_with_weather_df['LapStartTime']):
                    time_key_laps = 'LapStartTime'
                elif 'Time' in laps_with_weather_df.columns and pd.api.types.is_timedelta64_dtype(laps_with_weather_df['Time']):
                    session_start_time = session.date # This is the session's start datetime
                    laps_with_weather_df['AbsoluteLapTime'] = session_start_time + laps_with_weather_df['Time']
                    time_key_laps = 'AbsoluteLapTime'
                else:
                    print(f"WARNING: Suitable time column for weather merge not found in laps_df for {year} {event_name}.")
                    for col in weather_cols_to_add: # Add empty columns if no weather merge possible
                        laps_with_weather_df[col] = pd.NA
                    weather_data = None # Skip merge attempt

                if weather_data is not None and 'Time' in weather_data.columns:
                    # Ensure weather_data['Time'] is also absolute datetime
                    if pd.api.types.is_timedelta64_dtype(weather_data['Time']):
                        session_start_time = session.date
                        weather_data['Time'] = session_start_time + weather_data['Time']
                    
                    if pd.api.types.is_datetime64_any_dtype(weather_data['Time']):
                        # Sort by time for merge_asof
                        laps_with_weather_df = laps_with_weather_df.sort_values(by=time_key_laps)
                        weather_data_sorted = weather_data.sort_values(by='Time')
                        
                        # Select relevant weather columns to merge that are ACTUALLY in weather_data_sorted
                        weather_cols_for_merge = [col for col in weather_cols_to_add if col in weather_data_sorted.columns]
                        
                        if not weather_cols_for_merge:
                            print(f"WARNING: None of the desired weather columns found in session.weather_data for {year} {event_name}.")
                            for col in weather_cols_to_add: # Add empty columns if no weather merge possible
                                laps_with_weather_df[col] = pd.NA
                        else:
                            weather_data_subset = weather_data_sorted[weather_cols_for_merge + ['Time']].copy()
                            laps_with_weather_df = pd.merge_asof(
                        left=laps_with_weather_df,
                        right=weather_data_subset,
                        left_on=time_key_laps,
                        right_on='Time', # Weather data's time column
                        direction='nearest', # Find the nearest weather reading
                        suffixes=('', '_weather') # Avoid duplicate 'Time' column if not dropped
                    )
                    # Drop the temporary AbsoluteLapTime if created, and weather's Time column
                    if time_key_laps == 'AbsoluteLapTime':
                        laps_with_weather_df = laps_with_weather_df.drop(columns=['AbsoluteLapTime'])
                    if 'Time_weather' in laps_with_weather_df.columns: # if suffixes added a Time_weather
                         laps_with_weather_df = laps_with_weather_df.drop(columns=['Time_weather'])

                else:
                    if weather_data is not None: # weather_data exists but 'Time' column is missing or wrong type
                         print(f"WARNING: session.weather_data 'Time' column is missing or not datetime for {year} {event_name}.")
                    for col in weather_cols_to_add: # Add empty columns if no weather merge possible
                        laps_with_weather_df[col] = pd.NA
            else:
                print(f"WARNING: No session.weather_data available for {year} {event_name}. Weather columns will be NaN.")
                for col in weather_cols_to_add:
                    laps_with_weather_df[col] = pd.NA

            # 4. Select columns
            columns_to_extract = config.get('columns_to_extract', [])
            column_mapping = {
                'TeamName': 'Team',
            }
            
            final_columns = []
            missing_cols_in_source = [] # Columns from config not found in DataFrame
            
            # Build the list of columns we can actually get from the DataFrame
            for col_config_name in columns_to_extract:
                actual_col_name = column_mapping.get(col_config_name, col_config_name)
                if actual_col_name in laps_with_weather_df.columns:
                    final_columns.append(actual_col_name)
                else:
                    # Also check original config name, in case it was a direct match
                    if col_config_name in laps_with_weather_df.columns:
                         final_columns.append(col_config_name)
                    else:
                        missing_cols_in_source.append(col_config_name)

            if missing_cols_in_source:
                print(f"WARNING: Configured columns not found in source data for {year} {event_name}: {', '.join(missing_cols_in_source)}")
            
            if not final_columns: # If NO columns could be resolved (empty list)
                log_entry["Status"] = "Failed - NoColumnsExtracted"
                log_entry["Notes"] = "None of the specified columns_to_extract could be found in the source data."
                print(log_entry["Notes"])
                raise ValueError(log_entry["Notes"]) # Raise error to be caught by general Exception

            processed_df = laps_with_weather_df[final_columns].copy()
            
            # --- Type Conversion before saving to Parquet ---
            if 'DriverNumber' in processed_df.columns:
                processed_df['DriverNumber'] = processed_df['DriverNumber'].astype(str)
            if 'TrackStatus' in processed_df.columns:
                processed_df['TrackStatus'] = processed_df['TrackStatus'].astype(str)
            if 'LapNumber' in processed_df.columns:
                # Convert to float first to handle any non-integer strings, then to Int64
                processed_df['LapNumber'] = pd.to_numeric(processed_df['LapNumber'], errors='coerce').astype('Int64')
            if 'Position' in processed_df.columns:
                processed_df['Position'] = pd.to_numeric(processed_df['Position'], errors='coerce').astype('Int64')
            # Ensure boolean columns are bool
            if 'FreshTyre' in processed_df.columns:
                processed_df['FreshTyre'] = processed_df['FreshTyre'].astype(bool)
            if 'IsAccurate' in processed_df.columns:
                processed_df['IsAccurate'] = processed_df['IsAccurate'].astype(bool)
            # --- End Type Conversion ---

            # 5. NaN Percentage Check
            # 5. NaN Percentage Check
            # Use critical_columns_for_nan_check from config
            critical_columns_for_nan_check = config.get('critical_columns_for_nan_check', [])
            
            actual_critical_columns = [col for col in critical_columns_for_nan_check if col in processed_df.columns]
            
            if not critical_columns_for_nan_check: # If config list is empty
                print("WARNING: 'critical_columns_for_nan_check' is empty in config. Using all columns for NaN check.")
                nan_check_columns = processed_df.columns.tolist()
            elif not actual_critical_columns: # If config list is not empty, but none were found in df
                print(f"WARNING: None of the specified 'critical_columns_for_nan_check' were found in the dataframe. Columns available: {processed_df.columns.tolist()}. Using all columns for NaN check.")
                nan_check_columns = processed_df.columns.tolist()
            else:
                nan_check_columns = actual_critical_columns
            
            if not nan_check_columns: # Should not happen if processed_df has columns
                 print("ERROR: No columns available in processed_df for NaN check. Skipping NaN check.")
                 # This implies an earlier issue, but as a safeguard:
                 nan_percentage = 0.0 # Assume good if no columns to check (or handle as error)
            else:
                nan_percentage = processed_df[nan_check_columns].isnull().mean().mean()

            log_entry["NaNPercentage"] = round(nan_percentage, 4)
            print(f"NaN percentage for check columns ({', '.join(nan_check_columns)}): {nan_percentage:.2%}")

            if nan_percentage > config.get('max_nan_percentage_threshold', 0.05):
                log_entry["Status"] = "Incomplete - HighNaN"
                log_entry["Notes"] = f"NaN percentage ({nan_percentage:.2%}) exceeds threshold {config.get('max_nan_percentage_threshold', 0.05):.2%}."
                print(log_entry["Notes"])
                raise ValueError(log_entry["Notes"]) # Raise error

            # 6. Save to Parquet
            safe_event_name = event_name.replace(" ", "_").replace("/", "_")
            output_dir = os.path.join(config['raw_data_output_directory'], str(year), safe_event_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Use official event name for file to be safe
            file_event_name = session.event.get('OfficialEventName', session.event['EventName']).replace(' ', '_').replace('/', '_')
            file_name = f"{file_event_name}_{session_type}_data.parquet"
            file_path = os.path.join(output_dir, file_name)

            # --- DEBUGGING: Inspect DataFrame before saving ---
            print(f"DEBUG: DataFrame info for {file_path}:")
            processed_df.info(verbose=True, show_counts=True)
            print(f"DEBUG: DataFrame head for {file_path}:")
            print(processed_df.head())
            # --- END DEBUGGING ---
            
            processed_df.to_parquet(file_path, index=False)
            log_entry["FilePath"] = file_path
            log_entry["Status"] = "Success"
            log_entry["Notes"] = f"Data saved to {file_path}"
            print(log_entry["Notes"])
            success = True # Mark as successful for this attempt

        except ff1.core.DataNotLoadedError as e: # Specific error for no lap data
            log_entry["Status"] = "Failed - NoLapData"
            log_entry["Notes"] = str(e)
            print(f"ERROR: DataNotLoadedError for {year} {event_name}: {e}")
            # This is likely non-retryable if the session fundamentally lacks data.
            # We break the retry loop by setting retries to max_retries + 1
            retries = max_retries + 1 
        # Removed ff1.ErgastConnectionError as it's not a valid attribute
        # Ergast issues would likely be caught by requests.exceptions.HTTPError or ConnectionError
        except requests.exceptions.HTTPError as e:
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
                log_entry["Status"] = "Failed - RateLimit"
                log_entry["Notes"] = f"Rate limit exceeded: {str(e)}"
                print(f"ERROR: Rate limit exceeded for {year} {event_name}: {e}")
            else:
                log_entry["Status"] = "Failed - HTTPError"
                status_code_info = f", Status Code: {e.response.status_code}" if hasattr(e, 'response') and e.response is not None else ""
                log_entry["Notes"] = f"HTTP error: {str(e)}{status_code_info}"
                print(f"ERROR: HTTP error for {year} {event_name}: {e}")
        except ConnectionError as e:
            log_entry["Status"] = "Failed - ConnectionError"
            log_entry["Notes"] = str(e)
            print(f"ERROR: Connection error for {year} {event_name}: {e}")
        except ValueError as e: # Catch custom errors for NaN or No Columns
            # Status and Notes should already be set in log_entry by the raising code
            print(f"ERROR: Value error during processing for {year} {event_name}: {e}")
        except Exception as e:
            log_entry["Status"] = "Failed - UnknownError"
            log_entry["Notes"] = f"An unexpected error: {type(e).__name__} - {str(e)}"
            print(f"ERROR: An unexpected error occurred for {year} {event_name}: {e}")
        
        if success:
            break # Exit while loop if this attempt was successful

        retries += 1 # Increment retry counter if not successful
        if retries <= max_retries:
            print(f"Retrying ({retries}/{max_retries}) for {year} {event_name}... waiting {config.get('request_delay_seconds', 5)}s")
            time.sleep(config.get('request_delay_seconds', 5))
        else:
            print(f"Max retries reached for {year} {event_name}. Final status: {log_entry['Status']}")
            if log_entry["Status"] not in ["Success", "Skipped - NonRaceSession", "Failed - NoLapData"] and not log_entry["Status"].startswith("Failed - MaxRetriesReached"):
                 # If it's still "Initiated" or a retryable error status after all retries, mark as MaxRetriesReached
                if log_entry["Status"] == "Initiated" or "Failed -" in log_entry["Status"]: # Check if it's an error status that could be retried
                    log_entry["Status"] = "Failed - MaxRetriesReached"
                    current_notes = log_entry.get("Notes", "")
                    if "Max retries reached" not in current_notes: # Avoid duplicate message
                        log_entry["Notes"] = (current_notes + "; Max retries reached.").strip("; ")
    
    # Final log update for this race event, reflecting the outcome of all attempts
    update_download_log(config['download_log_file'], log_entry)


def main():
    """Main function to orchestrate data extraction."""
    config = load_config(CONFIG_PATH)
    if not config:
        return

    # Setup FastF1 cache
    cache_path = config.get('fastf1_cache_path')
    if cache_path:
        if not os.path.exists(cache_path):
            os.makedirs(cache_path, exist_ok=True)
            print(f"Created FastF1 cache directory: {cache_path}")
        ff1.Cache.enable_cache(cache_path)
        print(f"FastF1 cache enabled at: {cache_path}")
    else:
        print("WARNING: FastF1 cache path not specified in config. Cache is disabled.")

    # Ensure output and log directories exist
    raw_data_dir = config.get('raw_data_output_directory')
    if raw_data_dir and not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir, exist_ok=True)
        print(f"Created raw data output directory: {raw_data_dir}")

    log_file_path = config.get('download_log_file')
    if log_file_path:
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            print(f"Created log directory: {log_dir}")
    else:
        print("ERROR: 'download_log_file' not specified in config. Logging will fail.")
        return


    years_to_process = config.get('years_to_fetch', [])
    if not years_to_process:
        print("No years specified in 'years_to_fetch' in the configuration.")
        return

    all_potential_races = []
    # Process years in the order they are listed in the config
    print(f"Fetching event schedules for years: {years_to_process} (Order as in config: {years_to_process})") 
    for year_val in years_to_process: 
        try:
            schedule = ff1.get_event_schedule(year_val, include_testing=False)
            for index, event in schedule.iterrows():
                # Filter for events that are likely Grand Prix races
                # FastF1's Event object has an `is_race()` method, but schedule items are simple dicts/Series.
                # We rely on 'EventName' containing "Grand Prix" and session type 'R'.
                # Also, FastF1's `get_event_schedule` might return non-GP events.
                # A more robust filter could be `event.get('EventFormat') == 'conventional'` if that field exists and is reliable.
                # Or, check `event.get('Type') == 'Race'` if available from schedule.
                # For now, "Grand Prix" in name is a common heuristic.
                if "grand prix" in event['EventName'].lower():
                    all_potential_races.append({
                        'year': int(year_val), # Ensure year is int
                        'event_name': event['EventName'], 
                        'session_type': 'R' # We are only interested in Races
                    })
        except Exception as e:
            print(f"ERROR: Could not fetch event schedule for year {year_val}: {e}")
            # Optionally log this to a separate operational log or main log with a special status

    if not all_potential_races:
        print("No Grand Prix races found for the specified years based on 'grand prix' in EventName.")
        return

    # --- Resume Logic: Read log and filter out successfully processed races ---
    processed_races_set = set()
    if log_file_path and os.path.exists(log_file_path):
        try:
            log_df = pd.read_csv(log_file_path)
            if not log_df.empty:
                # Ensure columns exist before trying to filter
                if 'Year' in log_df.columns and 'EventName' in log_df.columns and \
                   'SessionType' in log_df.columns and 'Status' in log_df.columns:
                    successful_downloads = log_df[log_df['Status'] == 'Success']
                    for _, row in successful_downloads.iterrows():
                        processed_races_set.add((int(row['Year']), row['EventName'], row['SessionType']))
                    print(f"Found {len(processed_races_set)} successfully processed race(s) in the log.")
                else:
                    print("WARNING: Log file is missing expected columns (Year, EventName, SessionType, Status). Cannot determine processed races.")
        except pd.errors.EmptyDataError:
            print(f"Log file {log_file_path} is empty. No previously successful downloads to skip.")
        except Exception as e:
            print(f"Error reading or parsing log file {log_file_path}: {e}. Proceeding without skipping previously successful downloads.")

    races_to_actually_fetch = [
        race for race in all_potential_races
        if (race['year'], race['event_name'], race['session_type']) not in processed_races_set
    ]

    if not races_to_actually_fetch:
        if all_potential_races:
            print("All potential races already processed successfully according to the log or no new races to process.")
        else:
            print("No races to process after initial filtering and checking logs.")
        return
    
    print(f"Starting data extraction for {len(races_to_actually_fetch)} race(s) out of {len(all_potential_races)} potential races (after checking log)...")
    
    for race_info in races_to_actually_fetch:
        fetch_race_data(config, race_info)
        # Add delay between different event requests (after all retries for one event are done)
        # This delay is important to respect API rate limits when processing multiple events.
        time.sleep(config.get('request_delay_seconds', 1)) 

if __name__ == "__main__":
    main()

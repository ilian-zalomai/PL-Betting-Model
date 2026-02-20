import os
import requests
import pandas as pd

def fetch_and_merge_data():
    base_url = "https://www.football-data.co.uk/mmz4281/"
    data_folder = "data"
    
    # Create data folder if it doesn't exist
    os.makedirs(data_folder, exist_ok=True)
    
    all_seasons_df = []
    
    print("Starting data download...")

    # Iterate from 2003-2004 to 2025-2026
    for year in range(3, 26): # 03-04 to 25-26
        year_str_start = str(year).zfill(2)
        year_str_end = str(year + 1).zfill(2)
        season_code = f"{year_str_start}{year_str_end}"
        
        file_url = f"{base_url}{season_code}/E0.csv"
        file_path = os.path.join(data_folder, f"{season_code}.csv")
        
        print(f"Downloading {file_url} to {file_path}...")
        
        try:
            response = requests.get(file_url)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Successfully downloaded {season_code}.csv")
            
            # Read the downloaded CSV and append to the list
            df = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip')
            df['Season'] = f"20{year_str_start}-20{year_str_end}" # Add a Season column
            all_seasons_df.append(df)

        except requests.exceptions.HTTPError as errh:
            print(f"HTTP Error for {file_url}: {errh}")
        except requests.exceptions.ConnectionError as errc:
            print(f"Error Connecting for {file_url}: {errc}")
        except requests.exceptions.Timeout as errt:
            print(f"Timeout Error for {file_url}: {errt}")
        except requests.exceptions.RequestException as err:
            print(f"An unexpected error occurred for {file_url}: {err}")
        except pd.errors.EmptyDataError:
            print(f"Warning: {file_path} is empty or contains no data. Skipping.")
        except pd.errors.ParserError as pe:
            print(f"Parsing Error for {file_path}: {pe}. Skipping bad lines.")
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

    if all_seasons_df:
        print("Merging all season data...")
        merged_df = pd.concat(all_seasons_df, ignore_index=True)
        merged_df.to_csv("all_seasons.csv", index=False)
        print("Successfully merged all seasons into all_seasons.csv")
    else:
        print("No data was downloaded or processed to merge.")

if __name__ == "__main__":
    fetch_and_merge_data()

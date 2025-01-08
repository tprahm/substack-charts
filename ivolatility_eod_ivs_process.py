import requests
import os
import pandas as pd
from dotenv import load_dotenv
import time
import gzip
import shutil

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the .env file
api_key = os.getenv("IVOLATILITY_API_KEY")

if not api_key:
    raise ValueError("API key is not set in the .env file or environment variables.")

# Define the initial API endpoint
url = "https://restapi.ivolatility.com/equities/eod/ivs"

symbol = "TSLA"

# Define query parameters
params = {
    "apiKey": api_key,
    "symbol": symbol,
    "from": "2017-01-01",     # Start date
    "to": "2024-12-23",       # End date
    "OTMFrom": 0,            # Lower OTM range limit
    "OTMTo": 30,              # Upper OTM range limit
    "periodFrom": 7,         # Lower tenor range limit
    "periodTo": 360,         # Upper tenor range limit
    "region": "USA"           # Default region
}

# Make the HTTP GET request
try:
    response = requests.get(url, params=params)
    response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
    
    # Parse the JSON response
    data = response.json()
    print("Initial Response:", data)
    
    # Handle data directly returned
    if "data" in data and data["data"]:
        # Save directly to a CSV
        print("Data found directly in the response.")
        df = pd.DataFrame(data["data"])
        df.to_csv(f"{symbol}.csv", index=False)
        print(f"Data saved to '{symbol}.csv'.")
    elif data.get("status", {}).get("urlForDetails"):
        # Handle pending data with polling
        status_url = data["status"]["urlForDetails"]
        print(f"Polling data availability at: {status_url}")
        
        while True:
            status_response = requests.get(status_url)
            status_response.raise_for_status()
            status_data = status_response.json()
            print("Status Response:", status_data)

            if isinstance(status_data, list) and len(status_data) > 0:
                meta = status_data[0].get("meta", {})
                if meta.get("status") == "COMPLETE":
                    print("Data retrieval completed.")
                    file_info = status_data[0].get("data", [])[0]
                    download_url = file_info.get("urlForDownload")
                    
                    if download_url:
                        download_response = requests.get(download_url)
                        gz_filename = "temp_data.csv.gz"
                        with open(gz_filename, "wb") as f:
                            f.write(download_response.content)
                        print(f"Compressed file downloaded as '{gz_filename}'.")

                        with gzip.open(gz_filename, "rb") as gz_file:
                            with open(f"{symbol}.csv", "wb") as csv_file:
                                shutil.copyfileobj(gz_file, csv_file)
                        print(f"Decompressed file saved as '{symbol}.csv'.")
                        
                        os.remove(gz_filename)
                        break
                    else:
                        print("Download URL not found.")
                        break
                elif meta.get("status") == "FAILED":
                    print("Data retrieval failed.")
                    break
            else:
                print("Data still processing. Retrying in 10 seconds...")
                time.sleep(10)
    else:
        raise ValueError("No data or status URL found in the response.")

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
import os
import pandas as pd
import numpy as np
import requests
import argparse
from tqdm import tqdm

POSITIONSTACK_KEY = "cf5c69fc0ec557ecf2732bb17b4d04e2"

def main(rerun_coords=False):
    raw_data = pd.read_csv('raw_data.csv', sep=',', header=0)
    if rerun_coords:
        with open('coords.csv', 'w+') as f:
            f.write('Address,City,Lat,Lon\n')
            with tqdm(total=len(raw_data)) as pbar:
                for index, row in raw_data.iterrows():
                    lat, lon = get_gps_loc_from_address(row['Address'], row['City'])
                    f.write(f'{row["Address"]},{row["City"]},{lat},{lon}\n')
                    pbar.update(1)

    # Read the coordinates
    coords = pd.read_csv('coords.csv', sep=',', header=0)
    assert len(coords) == len(raw_data), "Number of coordinates does not match number of datapoints. Start the script with --rerun_coords to rerun the coordinate retrieval."
    
    # Combine the coordinates with the raw data
    raw_data = raw_data.merge(coords, on=['Address', 'City'])
    
def pre_process_data(raw_data: pd.DataFrame):
    # Drop Address
    raw_data = raw_data.drop(columns=['Address'], axis=1)
    
    

def get_gps_loc_from_address(street: str, city: str):
    url = "http://api.positionstack.com/v1/forward"
    params = {
        'access_key': POSITIONSTACK_KEY,
        'query': f'{street}, {city}, Netherlands',
        'limit': 1,
        'output': 'json'
    }
    response = requests.get(url, params=params)
    data = response.json()
    try:
        if data['data']:
            if data['data'][0]['confidence'] < 0.9:
                print(f'WARNING: Confidence for {street}, {city} is {data["data"][0]["confidence"]}')

            return data['data'][0]['latitude'], data['data'][0]['longitude']
        else:
            print(f'No data found for {street}, {city}')
            return 0, 0
    except Exception as e:
        print(f'Error for {street}, {city}: {e}')
        print(data)
        print("============")
        return 0, 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_coords', action='store_true')
    args = parser.parse_args()
    main(args.rerun_coords)


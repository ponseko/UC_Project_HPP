import os
import pandas as pd
import numpy as np
import requests
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split

GOOGLE_KEY = "AIzaSyCE6N00wuhnMqFitHkpDCgLlbl_1zrJIFo"

def main(rerun_coords, train_size):
    raw_data = pd.read_csv('raw_data.csv', sep=',', header=0)
    if rerun_coords:
        # Clear the log file
        with open('coords_log.txt', 'w+', encoding='utf-8') as f:
            f.write('')

        with open('coords.csv', 'w+', encoding='utf-8') as f:
            f.write('Address,City,Lat,Lon\n')
        
        with tqdm(total=len(raw_data)) as pbar:
            for index, row in raw_data.iterrows():
                lat, lon = get_gps_loc_from_address(row['Address'], row['City'])
                with open('coords.csv', 'a', encoding='utf-8') as f:
                    f.write(f'{row["Address"]},{row["City"]},{lat},{lon}\n')
                pbar.update(1)

    # Read the coordinates
    coords = pd.read_csv('coords.csv', sep=',', header=0)
    assert len(coords) == len(raw_data), "Number of coordinates does not match number of datapoints. Start the script with --rerun_coords to rerun the coordinate retrieval."
    
    # Combine the coordinates with the raw data
    raw_data = raw_data.merge(coords, on=['Address', 'City'])
    processed_data = pre_process_data(raw_data)
    train, test = train_test_split(processed_data, train_size=train_size, shuffle=True)

    y_train, X_train = train['Price'], train.drop('Price', axis=1)
    y_test, X_test = test['Price'], test.drop('Price', axis=1)


    X_train: pd.DataFrame
    tmp = X_train.to_numpy()
    print("=========")
    print(X_train.dtypes)
    print(tmp.dtype)
    print(tmp.shape)
    print("=========")

    np.save('X_train.npy', X_train.to_numpy())
    np.save('y_train.npy', y_train.to_numpy())
    np.save('X_test.npy', X_test.to_numpy())
    np.save('y_test.npy', y_test.to_numpy())

def _get_bedrooms(bedrooms: str):
    """Return the number of bedrooms from the rooms string:
    Example:
        input: '5 kamers (4 slaapkamers)'
        output: 4
    """
    if not 'slaapkamer' in bedrooms:
        return 0
    
    return int(bedrooms.split(' ')[2][1:])

def _get_rooms(rooms: str):
    """Return the number of rooms from the rooms string:
    Example:
        input: '5 kamers (4 slaapkamers)'
        output: 5
    """
    return int(rooms.split(' ')[0])

def _get_bathrooms(bathrooms: str):
    """Return the number of bathrooms from the bathrooms string:
    Example:
        input: '1 badkamer en 2 aparte toiletten'
        output: 1
    """
    return int(bathrooms.split(' ')[0])

def _get_toilets(toilets: str):
    """Return the number of toilets from the toilets string:
    Example:
        input: '1 badkamer en 2 aparte toiletten'
        output: 2
    """
    if 'toilet' not in toilets:
        return 0
    
    return int(toilets.split(' ')[-3])

def pre_process_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the string data in the raw data and return a dataframe with the processed data"""

    # Drop the rows with no price (Prijs op aanvraag)
    processed_data = raw_data[raw_data['Price'] != 'Prijs op aanvraag']
    processed_data = processed_data.dropna(subset=['Price'])

    # Print unique values in each column
    for col in processed_data.columns:
        print(f'Unique values in {col}:')
        print(len(processed_data[col].unique()))

    # Parse the price
    processed_data['Price'] = processed_data['Price'].apply(lambda x: int(x.replace('.', '').replace(' v.o.n.', '').replace('von', '')[2:])).astype(np.int32)

    # Parse the Lot size (m2)
    processed_data['Lot size (m2)'] = processed_data['Lot size (m2)'].apply(lambda x: int(x.replace('.', '')[:-3])).astype(np.int32)
    
    # Parse the Living space size (m2)
    processed_data['Living space size (m2)'] = processed_data['Living space size (m2)'].apply(lambda x: int(x.replace('.', '')[:-3])).astype(np.int32)

    # Convert build year to int
    processed_data['Build year'] = pd.to_numeric(processed_data['Build year'].str.replace(r'[^0-9]+', '', regex=True), errors='coerce')

    # Convert build type to int ('Bestaande bouw' = 0, 'Nieuwbouw' = 1)
    processed_data['Build type'] = processed_data['Build type'].apply(lambda x: 0 if x == 'Bestaande bouw' else 1).astype(np.int32)

    categorical_cols = ['House type', 'Roof', 'Position', 'Garden']
    processed_data.drop(categorical_cols, axis=1, inplace=True)
    # for col in categorical_cols:
    #     if do_onehot:
    #         # One-hot encode the categorical columns
    #         onehot = pd.get_dummies(processed_data[col], prefix=col)
    #         processed_data = processed_data.drop(col, axis=1)
    #         processed_data = processed_data.join(onehot)
    #     else:
    #         # Convert the categorical columns to integers
    #         processed_data[col] = processed_data[col].astype('category').cat.codes

    # Convert energy labels to oridnal int
    energy_labels = {
        'A++++': 0,
        'A+++': 1,
        'A++': 2,
        'A+': 3,
        'A': 4,
        'B': 5,
        'C': 6,
        'D': 7,
        'E': 8,
        'F': 9,
        'G': 10,
        'Niet verplicht': np.nan,
    }
    processed_data.replace({'Energy label': energy_labels}, inplace=True)
    processed_data['Energy label'] = pd.to_numeric(processed_data['Energy label'], errors='coerce')
    
    # Parse bathrooms and bedrooms
    processed_data['Bedrooms'] = processed_data['Rooms'].apply(_get_bedrooms).astype(np.int32)
    processed_data['Total Rooms'] = processed_data['Rooms'].apply(_get_rooms).astype(np.int32)
    processed_data.drop(['Rooms'], axis=1, inplace=True)

    processed_data['Bathrooms'] = processed_data['Toilet'].apply(_get_bathrooms).astype(np.int32)
    processed_data['Toilets'] = processed_data['Toilet'].apply(_get_toilets).astype(np.int32)
    processed_data.drop(['Toilet'], axis=1, inplace=True)

    # Floor
    processed_data['N_floors'] = processed_data['Floors'].apply(lambda x: int(x.split(' ')[0])).astype(np.int32)
    # processed_data['Has_basement'] = processed_data['Floors'].apply(lambda x: 1 if 'kelder' in x else 0).astype(np.int32)
    # processed_data['Has_loft'] = processed_data['Floors'].apply(lambda x: 1 if 'zolder' in x else 0).astype(np.int32)
    # processed_data['Has_attic'] = processed_data['Floors'].apply(lambda x: 1 if 'vliering' in x else 0).astype(np.int32)
    processed_data.drop(['Floors'], axis=1, inplace=True)

    # Parse estimated neighbourhood price per m2
    processed_data['Estimated neighbourhood price per m2'] = pd.to_numeric(processed_data['Estimated neighbourhood price per m2'].str.replace(r'[^0-9]+', '', regex=True), errors='coerce')

    # Drop Address
    processed_data.drop(columns=['Address', 'City'], axis=1, inplace=True)

    # Reassign the column order to have lat and lon at the beginning
    cols = processed_data.columns.tolist()
    cols.remove('Lat')
    cols.remove('Lon')
    cols = ['Lat', 'Lon'] + cols
    processed_data = processed_data[cols]
    processed_data['Price'] = np.log(processed_data['Price'])

    print("WITH NAN: ", len(processed_data))
    processed_data.dropna(inplace=True)
    print("WITHOUT NAN: ", len(processed_data))

    print(processed_data.head())
    print(processed_data.columns)
    return processed_data
    

def get_gps_loc_from_address(street: str, city: str):
    url = f'https://maps.googleapis.com/maps/api/geocode/json?key={GOOGLE_KEY}&address='
    url += '+'.join(f'{street} {city} Netherlands'.split())
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data['results']:
            lat = data['results'][0]['geometry']['location']['lat']
            lon = data['results'][0]['geometry']['location']['lng']
            return lat, lon
        else:
            with open('coords_log.txt', 'a', encoding='utf-8') as f:
                f.write(f'No results for {street}, {city}, Netherlands')
            return 0, 0
    except Exception as e:
        with open('coords_log.txt', 'a', encoding='utf-8') as f:
                f.write('=======\n')
                f.write(f'Error for {street}, {city}: {e}\n')
                # f.write(f'{data}\n')
                f.write('=======\n')
        return 0, 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_coords', action='store_true')
    parser.add_argument('--train_size', type=float, default=0.8)
    args = parser.parse_args()
    main(args.rerun_coords, args.train_size)

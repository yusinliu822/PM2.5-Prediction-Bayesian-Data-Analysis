import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from funcs.functions import load_data, site_information, format_publishTime, plt_coff_corr
from src.config import DIRECTORY, IMAGE_DIRECTORY, SITEID
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm
import os

def preprocessing():
    if not os.path.exists(IMAGE_DIRECTORY):
        os.makedirs(IMAGE_DIRECTORY)

    # read data
    df_pm25 = load_data()
    with open(DIRECTORY + 'label_dict.json', 'r') as f:
        label = json.load(f)

    # format site information
    df_pm25 = site_information(df_pm25)

    # format publish time
    tqdm.pandas(desc="Format publish time")
    df_pm25.PublishTime = df_pm25.PublishTime.progress_apply(format_publishTime)

    # Drop unnecessary columns
    unnecessary_cols = ['County', 'Latitude', 'Longitude', 'Status']
    df_pm25.drop(columns=[col for col in unnecessary_cols], inplace=True)
    print("Successfully drop columns:", [col for col in unnecessary_cols])

    # Deal with missing values

    # Drop rows with missing pm25 values
    df_pm25.dropna(subset=['PM2.5'], inplace=True)

    # Fill missing values with UNKNOWN in 'Pollutant'
    df_pm25.Pollutant.fillna('UNKNOWN', inplace=True)

    # Drop rows with '-' in any columns
    row_mask = df_pm25.apply(lambda row: '-' in row.values, axis=1)
    df_pm25.drop(df_pm25[row_mask].index, inplace=True)

    ## drop rows with nan value
    nan_column = ['WindSpeed', 'WindDirec']
    drop_cols = [col for col in df_pm25.columns if col not in nan_column]
    df_pm25.dropna(subset=drop_cols, inplace=True)

    # reset column order
    df_pm25 = df_pm25[label['column_order']]

    # set data type
    datatype = {key: np.dtype(value) for key, value in label['datatype'].items()}
    df_pm25 = df_pm25.astype(datatype)

    # change Pollutant column to one-hot encoding
    df_pm25.Pollutant = df_pm25.Pollutant.replace(label['pollutant_mapping'])
    df_pm25 = pd.get_dummies(df_pm25, columns=['Pollutant'])

    # store data
    df_pm25.to_csv(DIRECTORY + 'pm25_processed.csv', index=False)

    # Plot coefficient relationship heatmap
    plt_coff_corr(df_pm25)

    print("Pre-processing done")

def deal_with_site():

    if os.path.exists(f'{DIRECTORY}pm25_changed{SITEID}.json'):
        return pd.read_json(f'{DIRECTORY}pm25_changed{SITEID}.json')
    
    # read data
    df_pm25 = pd.read_csv(f'{DIRECTORY}pm25_processed.csv', low_memory=False)

    # drop duplicates and set SiteID
    df_pm25_part = df_pm25[df_pm25.SiteId == SITEID]
    df_pm25_part.drop_duplicates("PublishTime", inplace=True)

    # Set PublishTime to index
    df_pm25_part = df_pm25_part.set_index('PublishTime', inplace=False)
    df_pm25_part.index = pd.to_datetime(df_pm25_part.index)

    # choose the time
    df_pm25_part = df_pm25_part[df_pm25_part.index.strftime('%Y-%m-%d') < '2023-1-1']

    # setting future 1 hour, 2 hour, 3 hour pm2.5
    SHIFT_PERIODS = [1, 2, 3]
    for period in SHIFT_PERIODS:
        df_pm25_part[f'PM{period}hour'] = df_pm25_part['PM2.5'].shift(periods=-period)

    #use ffiil to fill the nan value
    df_pm25_part = df_pm25_part.fillna(method='ffill')

    # drop useless columns
    columns_to_drop = [col for col in df_pm25_part.columns if 'Pollutant' in col]
    df_pm25_part.drop(columns=columns_to_drop, inplace=True)
    df_pm25_part.drop(columns=['WindSpeed', 'WindDirec'], inplace=True)
    # df_pm25_part.drop(columns=['WindSpeed', 'WindDirec', 'Pollutant'], inplace=True) ## original
    df_pm25_part.to_json(f'{DIRECTORY}pm25_changed{SITEID}.json', indent=4)

    # Plot coefficient relationship heatmap
    # fig, ax = plt.subplots(figsize=(20, 8))
    # sns.heatmap(df_pm25_part.corr(), annot=True, ax=ax, cmap='coolwarm')
    # plt.savefig(f'images/heatmap_region{SITEID}_{current_time}.png')
    # plt.close()
    print(f"Finish preprocessing for site{SITEID}!")
    return df_pm25_part

if __name__ == '__main__':
    
    preprocessing()
    df_pm25_part = deal_with_site()
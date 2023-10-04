import os
from tqdm import tqdm
import pandas as pd
import pandas as np
import seaborn as sns
import matplotlib.pyplot as plt
from funcs.site_func import extract_site_name, format_site_information, map_siteId
import datetime
from config import DIRECTORY, IMAGE_DIRECTORY

def load_data():

    # Read data and concatenate
    NAMEBASE = 'EPA_OD_'
    DATAYEARS = ['2021', '2022']
    DESTINATION = DIRECTORY + 'pm25.csv'

    if os.path.exists(DESTINATION): 
        return pd.read_csv(DESTINATION)
    
    df_pm25 = pd.DataFrame()
    for yr in DATAYEARS:
        for month in range(1, 13):
            month = str(month).zfill(2)
            path = DIRECTORY + NAMEBASE + yr + month + '.csv'
            if not os.path.exists(path): break
            df = pd.read_csv(path, low_memory=False)
            if len(cols) == 0: cols = df.columns
            assert (cols == df.columns).all(), f'Columns do not match {yr}{month}\n{cols}\n{df.columns}'
            df_pm25 = pd.concat([df_pm25, df], ignore_index=True)

    df_pm25.to_csv(DESTINATION, index=False)
    print("Successfully load data")
    return df_pm25

def site_information(df):
    
    region_cols = ['SiteId', 'SiteName', 'County', 'Latitude', 'Longitude']
    df_region = df[region_cols].drop_duplicates().reset_index(drop=True).sort_values(by='SiteId').reset_index(drop=True)
    format_site_information(df_region)

    tqdm.pandas(desc="extract_site_name")
    df.SiteName = df.SiteName.progress_apply(extract_site_name)
    tqdm.pandas(desc="map_siteId")
    df.SiteId = df.SiteName.progress_apply(map_siteId)

    print("Successfully format site information")

    return df

def plt_coff_corr(df_pm25):
    # Plot coefficient relationship heatmap
    df_pm25.drop(columns=['SiteId', 'SiteName', 'PublishTime', 'WindSpeed', 'WindDirec'], inplace=True)

    # Plot coefficient relationship heatmap
    _ , ax = plt.subplots(figsize=(20, 8))
    sns.heatmap(df_pm25.corr(), annot=True, ax=ax, cmap='coolwarm')
    # plt.savefig(IMAGE_DIRECTORY + 'corr_heatmap.png')

def format_publishTime(time):
    time = time.replace('/', '-')
    return datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:00').strftime('%Y-%m-%d %H:00:00')
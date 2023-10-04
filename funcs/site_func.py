import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm
from src.config import DIRECTORY

DESTINATION = DIRECTORY + 'site_information.csv'

def extract_site_name(siteName):
    regex = r"([^\(\t]*)\((.*)\)"
    match = re.search(regex, siteName)
    if match and match.group(1) == "永和": #???
        return match.group(1)
    return match.group(2) if match else siteName

def format_site_information(df_region):

    if os.path.exists(DESTINATION):
        return pd.read_csv(DESTINATION)
    tqdm.pandas(desc="Format site information")
    df_region_map = df_region.copy()
    # format site name
    df_region_map['SiteName'] = df_region_map['SiteName'].apply(extract_site_name)
    # change SiteId == 0 to 85
    df_region_map.loc[df_region_map['SiteId'] == 0, 'SiteId'] = 85
    df_region_map.drop_duplicates(subset=['SiteName', 'County'], inplace=True)
    df_region_map.sort_values(by='SiteId', inplace=True, ignore_index=True)
    # reset SiteId > 85 to index + 8
    df_region_map.loc[df_region_map['SiteId'] > 85, 'SiteId'] = df_region_map.loc[df_region_map['SiteId'] > 85].index + 8
    df_region_map.to_csv(DESTINATION, index=False)

# map site information to data
def map_siteId(siteName):
    df_region_map = pd.read_csv(DESTINATION)
    site_info = df_region_map[df_region_map['SiteName'] == siteName]
    return site_info['SiteId'].values[0] if len(site_info) > 0 else np.nan






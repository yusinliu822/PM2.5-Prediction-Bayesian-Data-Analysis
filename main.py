from src.preprocessing import preprocessing, deal_with_site
from src.baseline import model
from src.config import DIRECTORY
import os

def main():

    if not os.path.exists(DIRECTORY + 'pm25_processed.csv'):
        preprocessing()
    df_pm25_part = deal_with_site()
    model(df_pm25_part)
    
    

if __name__ == '__main__':
    main()
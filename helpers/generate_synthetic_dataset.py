import os
import time
import pandas as pd
import datetime as dt
from tqdm import tqdm
from helper import time_taken

"""
import kaggle
kaggle.api.authenticate()
def download_dataset(dataset: str):
    save_to = 'sample_data/'
    kaggle.api.dataset_download_files(dataset, path=save_to,
                                      unzip=True)
"""


def process_tables(path_to_tables: str = 'sample_data/'):
    for table in tqdm(os.listdir('../sample_data/csvfiles')):
        df = pd.read_csv('../sample_data/csvfiles/{}'.format(table), low_memory=False)
        if 'timestamp' not in df.columns:
            df['timestamp'] = dt.datetime.now()
            os.remove('../sample_data/csvfiles/{}'.format(table))
            df.to_parquet('../sample_data/parquet/{}'.format(table.replace('.csv', '.parquet')))
            df.to_csv('../sample_data/csvfiles/{}'.format(table))


def main():
    """
    TODO: use kaggle's API to automate dataset downloading. Example dataset collected from:
    (https://www.kaggle.com/datasets/kabure/retail-bankingdemodata?select=completedloan.csv)
    """
    start = time.time()
    process_tables()
    print('Done in ', time_taken(start, time.time()))


if __name__ == "__main__":
    main()

import os
import time
import shutil
import pandas as pd
import datetime as dt
from datetime import timedelta
from random import randrange
from tqdm import tqdm
from helper import time_taken

"""
Dataset used: https://www.kaggle.com/datasets/kabure/retail-bankingdemodata?select=completedloan.csv
"""


def download_datasets(dataset_ref: str, save_to='sample_data/csv/'):
    dataset = None
    os.system('kaggle datasets download -d "{}"'.format(dataset_ref))
    for file in os.listdir():
        if file.endswith('.zip'):
            dataset = file.replace('.zip', '')
    os.mkdir(save_to+dataset)
    shutil.move(dataset + '.zip', save_to + dataset)

    os.chdir(save_to + dataset)
    os.system('unzip {}'.format(dataset + '.zip'))
    os.remove(dataset + '.zip')

    # removing large files for now
    for file in os.listdir():
        size = os.stat(file).st_size / 1000000  # convert size to MB
        if size > 5:
            os.remove(file)
    return dataset

def process_tables(dataset, path_to_tables: str = 'sample_data/csv/'):
    print('Adding random timestamps')
    os.mkdir('../../parquet/{}'.format(dataset))
    def get_random_timestamps(n_rows: int = 2, window_size: int = 10):
        random_timestamps = []
        date = dt.datetime.now()
        while n_rows:
            n_rows = n_rows - 1
            random_date = date - timedelta(days=randrange(window_size+1))
            random_date = random_date.replace(hour=randrange(24), minute=randrange(60), second=randrange(60))
            random_timestamps.append(random_date)
        return random_timestamps

    for table in tqdm(os.listdir()):
        df = pd.read_csv(table, low_memory=False)
        if 'event_timestamp' not in df.columns:
            df['event_timestamp'] = get_random_timestamps(df[df.columns[0]].count())
            df.to_parquet('../../parquet/{}/{}'.format(dataset, table.replace('.csv', '.parquet')))
            os.remove(table)
            df.to_csv(table)


def main():
    shutil.rmtree('sample_data/csv/retail-bankingdemodata')
    start = time.time()
    dataset = download_datasets('kabure/retail-bankingdemodata', 'sample_data/csv/')
    process_tables(dataset)
    print('Done in ', time_taken(start, time.time()))


if __name__ == "__main__":
    main()

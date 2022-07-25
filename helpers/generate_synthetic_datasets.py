import os
import random
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
    os.mkdir(save_to + dataset)
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
            random_date = date - timedelta(days=randrange(window_size + 1))
            random_date = random_date.replace(hour=randrange(24), minute=randrange(60), second=randrange(60))
            random_timestamps.append(random_date)
        return random_timestamps

    # add starting tables to dataset (account.csv and district.csv)
    def add_starting_datasets():
        for specific_table in ['completeddistrict.csv', 'completedacct.csv']:
            new_df = pd.read_csv(specific_table, low_memory=False)
            original_ids = new_df[new_df.columns[0]].tolist()
            if len(original_ids) >= 100:
                ids = random.sample(original_ids, int(len(original_ids) * 0.80))
                size = len(new_df) - len(ids) + 200
                ids.extend(random.sample(ids, size))
            else:
                ids = random.sample(original_ids, int(len(original_ids) * 0.60))
                size = 100 - len(ids)
                ids.extend(random.sample(original_ids, size))
            starting_table = pd.DataFrame(ids, columns=[new_df.columns[0]])
            # add custom name
            starting_table.to_csv(new_df.columns[0].replace('_id', '') + '.csv', index=False)
        client_ids = pd.read_csv('crm_call_center_logs.csv')['rand_client'].tolist()
        client_ids = [int(x) for x in client_ids if str(x) != 'nan']

        other_client_ids = pd.read_csv('completeddisposition.csv')['client_id']
        client_ids = client_ids + list(set(other_client_ids) - set(client_ids))
        client_ids = list(set(client_ids))
        pd.DataFrame(list(zip(client_ids)),
                     columns=['client_id']).to_csv('client.csv', index=False)

    # add synthetic features
    def add_synthetic_features():
        new_df = pd.read_csv('completeddistrict.csv', low_memory=False)
        rows = len(new_df)
        new_df['avg_sales'] = random.sample(range(500000, 1500000), rows)
        new_df['net_growth'] = random.sample(range(3000, 9000), rows)
        new_df['return'] = random.sample(range(100, 999), rows)
        new_df['n_customer_sat'] = [random.sample(range(1, 5), 1)[0] for i in range(rows)]
        new_df.to_csv('completeddistrict.csv', index=False)

    # add a use-case for multiple entities
    def add_table_with_multiple_entities():
        account_ids = pd.read_csv('account.csv')['account_id'].tolist()
        size = len(account_ids)
        client_ids = random.sample(pd.read_csv('client.csv')['client_id'].to_list() * 2, size)
        province = random.sample(['British Columbia', 'Quebec', 'Ontario', 'Newfoundland & Labrador', 'Alberta'] * 5000,
                                 size)
        new_df = pd.DataFrame(list(zip(client_ids, account_ids, province, ['Canada'] * size)),
                              columns=['client_id', 'account_id', 'province', 'country'])
        new_df.to_csv('month_summary.csv', index=False)

    add_starting_datasets()
    add_synthetic_features()
    add_table_with_multiple_entities()
    # convert to parquet

    for table in tqdm(os.listdir()):
        df = pd.read_csv(table, low_memory=False)
        if 'event_timestamp' not in df.columns:
            df['event_timestamp'] = get_random_timestamps(df[df.columns[0]].count())
            df.to_parquet('../../parquet/{}/{}'.format(dataset, table.replace('.csv', '.parquet')))
            os.remove(table)
            df.to_csv(table, index=False)


def main():
    if os.path.exists('sample_data/csv/retail-bankingdemodata'):
        shutil.rmtree('sample_data/csv/retail-bankingdemodata')
    if os.path.exists('sample_data/parquet/retail-bankingdemodata'):
        shutil.rmtree('sample_data/parquet/retail-bankingdemodata')
    start = time.time()
    dataset = download_datasets('kabure/retail-bankingdemodata', 'sample_data/csv/')
    process_tables(dataset)
    print('Done in ', time_taken(start, time.time()))


if __name__ == "__main__":
    main()

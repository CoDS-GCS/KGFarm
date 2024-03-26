import random

import numpy as np

from feature_discovery.src.api.template import *
from helpers.helper import connect_to_graphdb #connect_to_stardog
from tqdm import tqdm
import pandas as pd

def add_labels(features_df: pd.DataFrame, database: str):
    path_to_groundtruth = '../../../helpers/groundtruth/{}_pkfk.csv'.format(database)
    groundtruth = pd.read_csv(path_to_groundtruth)
    total_pairs = len(groundtruth)
    processed_groundtruth = []
    for k, v in groundtruth.to_dict('index').items():
        primary_table = v['PK - Table']
        primary_key = v['PK - ColName']
        foreign_table = v['FK - Table']
        foreign_key = v['FK - ColName']
        processed_groundtruth.append((foreign_table, foreign_key, primary_table, primary_key))
    count=0
    labels = []
    for k, v in features_df.to_dict('index').items():
        count=count+1
        if (v['Foreign_table'], v['Foreign_key'], v['Primary_table'], v['Primary_key']) in processed_groundtruth:
            labels.append(1)
            processed_groundtruth.remove((v['Foreign_table'], v['Foreign_key'], v['Primary_table'], v['Primary_key']))
        else:
            labels.append(0)

    print(database, ' has ',count,' pairs')
    print('Non-detected pairs: ',len(processed_groundtruth),'/',total_pairs)
    print('PK-FK Not detected: ', processed_groundtruth)
    features_df['Has_pk_fk_relation'] = labels
    return features_df

if __name__ == "__main__":
    database_list = ['SAP','tpcd','tpcds','Credit','Grants', 'TPC-H', 'financial', 'financial_std', 'financial_ijs','cs', 'SAT','tpcc']
    for database in database_list:
        conn = connect_to_graphdb('http://localhost:7200/repositories/'+database)
        F = get_content_similar_pairs(conn)
        df = add_labels(F, database)
        df.to_csv('025_'+database+'.csv')
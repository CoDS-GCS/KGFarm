import numpy as np

from feature_discovery.src.api.template import *
from helpers.helper import connect_to_stardog
from tqdm import tqdm
import pandas as pd
from difflib import SequenceMatcher
import re

def lcs(S,T):
    m = len(S)
    n = len(T)
    counter = [[0]*(n+1) for x in range(m+1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if S[i] == T[j]:
                c = counter[i][j] + 1
                counter[i+1][j+1] = c
                if c > longest:
                    lcs_set = set()
                    longest = c
                    lcs_set.add(S[i-c+1:i+1])
                elif c == longest:
                    lcs_set.add(S[i-c+1:i+1])

    return lcs_set

def generate_features(conn, ind: pd.DataFrame,dataset_name):
    def aggregate_features(ind_pairs: pd.DataFrame, features: list):
        # 1. join features, key = A & B
        # 2. if NAN i.e. edge didn't exist --> 0
        # 3. repeat for all features
        result = ind_pairs
        for feature in tqdm(features):
            result = pd.concat([result, feature], axis=1)
            result = result.loc[:, ~result.columns.duplicated()].fillna(0)

        return result

    def normalize(vector: list):
        max_value = max(vector)
        return list(map(lambda x: x / max_value, vector))

    def generate_F01():
        F1 = []
        A = ind['A'].tolist()
        F = get_distinct_dependent_values(conn)
        F_list = pd.Series(F[F['A'].isin(A)].values.tolist())
        for a in A:
            for b in F_list:
                if a == b[0]:
                    F1.append(b[2])
        f1 = ind
        f1['F1'] = F1
        return f1

    def generate_F02():
        A = list(zip(ind['A'].tolist(),ind['B'].tolist()))
        F = get_content_similarity(conn)
        F_list = pd.Series(F[F[['A', 'B']].agg(tuple, 1).isin(tuple(A))].values.tolist())
        F2 = []
        for a in A:
            found = False
            for b in F_list:
                if a[0] == b[0] and a[1]==b[1]:
                    F2.append(b[2])
                    found = True
                    break
            if found == False:
                F2.append(0)
        f2 = ind
        f2['F2'] = F2
        return f2

    def generate_F03():
        # TODO: F3 generation needs optimization
        # counts how often A appears as B
        count = []
        A = ind['A'].tolist()
        B = ind['B'].tolist()

        for a in A:
            presence = 0
            for b in B:
                if a == b:
                    presence = presence + 1
            count.append(presence)

        count = normalize(count)
        f3 = ind
        f3['F3'] = count
        return f3

    def generate_F04():
        A = ind['A'].tolist()
        freq = list(map(lambda x: A.count(x), A))
        freq = normalize(freq)
        f4 = ind
        f4['F4'] = freq
        return f4

    def generate_F05():
        # TODO: F5 generation needs optimization
        # counts how often B appears as A
        count = []
        A = ind['A'].to_list()
        B = ind['B'].tolist()

        for b in B:
            presence = 0
            for a in B:
                if a == b:
                    presence = presence + 1
            count.append(presence)
        count = normalize(count)
        f5 = ind
        f5['F5'] = count
        return f5

    # def generate_F06():
    #     A = list(zip(ind['Foreign_table'].tolist(), ind['Primary_table'].tolist()))
    #     F6 = []
    #     for a in A:
    #         v = lcs(a[0].lower(),a[1].lower())
    #         try:
    #             if len(max(v, key=len))>3:
    #                 score = len(max(v, key=len))/len(a[0])
    #             else:
    #                 score = 0
    #         except ValueError:
    #             score = 0
    #         print(a[0],a[1], v,score)
    #         F6.append(score)
    #
    #     f6 = ind
    #     f6['F6'] = F6
    #     return f6

    def generate_F06():
        A = list(zip(ind['Foreign_table'].tolist(), ind['Primary_table'].tolist()))
        F6 = []
        for a in A:
            if a[0].lower()==a[1].lower() or a[0].lower() in a[1].lower():
                score = 1
            else:
                score = 0
            #print(a[0],a[1],score)
            F6.append(score)

        f6 = ind
        f6['F6'] = F6
        return f6


    # def generate_F06():
    #     A = list(zip(ind['A'].tolist(),ind['B'].tolist()))
    #     F = get_column_name_similarity(conn)
    #     F_list = pd.Series(F[F[['A', 'B']].agg(tuple, 1).isin(tuple(A))].values.tolist())
    #     F6 = []
    #     for a in A:
    #         found = False
    #         for b in F_list:
    #             if a[0] == b[0] and a[1]==b[1]:
    #                 F6.append(b[2])
    #                 found = True
    #                 break
    #         if found == False:
    #             F6.append(0)
    #     f6 = ind
    #     f6['F6'] = F6
    #     return f6

    def generate_F08():
        A = list(zip(ind['A'].tolist(),ind['B'].tolist()))
        F = get_range(conn)
        F_list = pd.Series(F[F[['A', 'B']].agg(tuple, 1).isin(tuple(A))].values.tolist())
        F8 = []
        for a in A:
            found = False
            for b in F_list:
                if a[0] == b[0] and a[1]==b[1]:
                    F8.append(b[2])
                    found = True
                    break
            if found == False:
                F8.append(0)
        f8 = ind
        f8['F8'] = F8
        return f8

    def generate_F09():
        A = list(zip(ind['A'].tolist(),ind['B'].tolist()))
        F = get_typical_name_suffix(conn)
        F_list = pd.Series(F[F[['A', 'B']].agg(tuple, 1).isin(tuple(A))].values.tolist())
        F9 = []
        for a in A:
            found = False
            for b in F_list:
                if a[0] == b[0] and a[1]==b[1]:
                    F9.append(b[2])
                    found = True
                    break
            if found == False:
                F9.append(0)
        f9 = ind
        f9['F9'] = F9
        return f9

    def generate_F10():
        A = list(zip(ind['A'].tolist(),ind['B'].tolist()))
        F = get_table_size_ratio(conn)
        F_list = pd.Series(F[F[['A', 'B']].agg(tuple, 1).isin(tuple(A))].values.tolist())
        F10 = []
        for a in A:
            found = False
            for b in F_list:
                if a[0] == b[0] and a[1]==b[1]:
                    F10.append(b[2])
                    found = True
                    break
            if found == False:
                F10.append(0)
        f10 = ind
        f10['F10'] = F10
        return f10

    (aggregate_features(ind, [generate_F01(),
                                    generate_F02(),
                                    generate_F03(),
                                    generate_F04(),
                                    generate_F05(),
                                    generate_F06(),
                                    generate_F08(),
                                    generate_F09(),
                                    generate_F10()])[['Foreign_table', 'Foreign_key', 'Primary_table',
                                                      'Primary_key', 'F1', 'F2',
                                                      'F3', 'F4', 'F5',
                                                      'F6', 'F8',
                                                      'F9', 'F10']]).to_csv(dataset_name + '_features.csv')
    return aggregate_features(ind, [generate_F01(),
                                    generate_F02(),
                                    generate_F03(),
                                    generate_F04(),
                                    generate_F05(),
                                    generate_F06(),
                                    generate_F08(),
                                    generate_F09(),
                                    generate_F10()])[['Foreign_table', 'Foreign_key', 'Primary_table',
                                                      'Primary_key', 'F1', 'F2',
                                                      'F3', 'F4', 'F5',
                                                      'F6', 'F8',
                                                      'F9', 'F10']]


def add_labels(features_df: pd.DataFrame, database: str):
    path_to_groundtruth = '../../../helpers/groundtruth/{}_pkfk.csv'.format(database[:-3])
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


def export_csv(features_df: pd.DataFrame, save_as: str):
    features_df.to_csv(save_as + '_features.csv')
    return features_df


def generate(database_list: list, export_features: bool = True):
    # This implementation is as per http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.438.9288&rep=rep1&type=pdf
    all_features_df=pd.DataFrame()
    for database in database_list:
        # TODO: add a filter for Table_A != Table_B  while retrieving inclusion dependencies for all features
        conn = connect_to_stardog(port=5822, db=database, show_status=True)
        # Get all inclusion dependencies (IND) or content similarity
        # To get the INDs from the graph
        #ind_by_graph = get_INDs(conn)
        # To get the INDs via human in the loop
        # ind_by_HITL = pd.read_csv('../../../helpers/IND_Discovery/'+database+'.csv')
        # To get pairs by Content Similarity
        content_similar = get_content_similar_pairs(conn)
        # Generate features for these IND or content similar pairs
        generated_pairs = content_similar
        features_df = generate_features(conn, generated_pairs,database)
        # Add labels / target using true mappings
        features_df = add_labels(features_df, database)
        features_df.drop(columns=['Foreign_table', 'Foreign_key', 'Primary_table', 'Primary_key'], axis=1, inplace=True)
        all_features_df = pd.concat([features_df,all_features_df])

        # Export csv
    if export_features:
        export_csv(all_features_df, database_list[0])
        return all_features_df

# if __name__ == "__main__":
#     generate(database='TPC-H')

import pandas as pd

from helpers.helper import connect_to_stardog
from operations.template import *


def generate_features(conn, ind: pd.DataFrame):
    def aggregate_features(ind_pairs: pd.DataFrame, features: list):
        # 1. join features, key = A & B
        # 2. if NAN i.e. edge didn't exist --> 0
        # 3. repeat for all features
        # print(ind_pairs)

        result = ind_pairs
        for feature in features:
            result = pd.concat([result, feature], axis=1)
            result = result.loc[:, ~result.columns.duplicated()].dropna()
        return result

    def normalize(vector: list):
        max_value = max(vector)
        return list(map(lambda x: x / max_value, vector))

    def generate_F01():
        return get_distinct_dependent_values(conn)

    def generate_F02():
        A = list(zip(ind['A'].tolist(), ind['B'].tolist()))
        F = get_content_similarity(conn)
        F_list = pd.Series(F[F[['A', 'B']].agg(tuple, 1).isin(tuple(A))].values.tolist())
        F2 = []
        for a in A:
            found = False
            for b in F_list:
                if a[0] == b[0] and a[1] == b[1]:
                    F2.append(b[2])
                    found = True
                    break
            if not found:
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

        if len(count) != 0:
            count = normalize(count)
        f3 = ind
        f3['F3'] = count
        return f3

    def generate_F04():
        A = ind['A'].tolist()
        freq = list(map(lambda x: A.count(x), A))
        if len(freq) != 0:
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
        if len(count) != 0:
            count = normalize(count)
        f5 = ind
        f5['F5'] = count
        return f5

    def generate_F06():
        return get_column_name_similarity(conn)

    def generate_F08():
        A = list(zip(ind['A'].tolist(), ind['B'].tolist()))
        F = get_range(conn)
        F_list = pd.Series(F[F[['A', 'B']].agg(tuple, 1).isin(tuple(A))].values.tolist())
        F8 = []
        for a in A:
            found = False
            for b in F_list:
                if a[0] == b[0] and a[1] == b[1]:
                    F8.append(b[2])
                    found = True
                    break
            if not found:
                F8.append(0)
        f8 = ind
        f8['F8'] = F8
        return f8

    def generate_F09():
        A = list(zip(ind['A'].tolist(), ind['B'].tolist()))
        F = get_typical_name_suffix(conn)
        F_list = pd.Series(F[F[['A', 'B']].agg(tuple, 1).isin(tuple(A))].values.tolist())
        F9 = []
        for a in A:
            found = False
            for b in F_list:
                if a[0] == b[0] and a[1] == b[1]:
                    F9.append(b[2])
                    found = True
                    break
            if not found:
                F9.append(0)
        f9 = ind
        f9['F9'] = F9
        return f9

    def generate_F10():
        A = list(zip(ind['A'].tolist(), ind['B'].tolist()))
        F = get_table_size_ratio(conn)
        F_list = pd.Series(F[F[['A', 'B']].agg(tuple, 1).isin(tuple(A))].values.tolist())
        F10 = []
        for a in A:
            found = False
            for b in F_list:
                if a[0] == b[0] and a[1] == b[1]:
                    F10.append(b[2])
                    found = True
                    break
            if not found:
                F10.append(0)
        f10 = ind
        f10['F10'] = F10
        return f10
    return aggregate_features(ind, [generate_F01(),
                                    generate_F02(),
                                    generate_F03(),
                                    generate_F04(),
                                    generate_F05(),
                                    generate_F06(),
                                    generate_F08(),
                                    generate_F09(),
                                    generate_F10()])[['A', 'B', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F8',
                                                      'F9', 'F10']]


def add_labels(features_df: pd.DataFrame, database: str):
    path_to_groundtruth = '../../../helpers/groundtruth/{}_pkfk.csv'.format(database)
    groundtruth = pd.read_csv(path_to_groundtruth)
    processed_groundtruth = []
    for k, v in groundtruth.to_dict('index').items():
        primary_table = v['PK - Table']
        primary_key = v['PK - ColName']
        foreign_table = v['FK - Table']
        foreign_key = v['FK - ColName']
        processed_groundtruth.append((foreign_table, foreign_key, primary_table, primary_key))

    labels = []
    for k, v in features_df.to_dict('index').items():
        if (v['Foreign_table'], v['Foreign_key'], v['Primary_table'], v['Primary_key']) in processed_groundtruth:
            labels.append(1)
        else:
            labels.append(0)

    features_df['Has_pk_fk_relation'] = labels
    return features_df


def export_csv(features_df: pd.DataFrame, save_as: str):
    features_df.to_csv(save_as + '_features.csv')
    return features_df


def generate(database: str, export_features: bool = False):
    # This implementation is as per http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.438.9288&rep=rep1&type=pdf
    # TODO: add a filter for Table_A != Table_B  while retrieving inclusion dependencies for all features
    conn = connect_to_stardog(port=5822, db=database, show_status=True)
    # Get all inclusion dependencies (IND) initially
    ind = get_INDs(conn)
    # Generate features for these IND pairs
    features_df = generate_features(conn, ind)
    # Add labels / target using true mappings
    features_df = add_labels(features_df, database)
    features_df.drop(columns=['Foreign_table', 'Foreign_key', 'Primary_table', 'Primary_key'], axis=1, inplace=True)
    # Export csv
    if export_features:
        export_csv(features_df, database)
    return features_df

# if __name__ == "__main__":
#     generate(database='TPC-H')

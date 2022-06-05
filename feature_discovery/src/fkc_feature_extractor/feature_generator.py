from feature_discovery.src.api.template import *
from helpers.helper import connect_to_stardog
from tqdm import tqdm
import pandas as pd


def generate_features(conn, ind: pd.DataFrame):
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
        return get_distinct_dependent_values(conn)

    def generate_F02():
        return get_content_similarity(conn)

    def generate_F03():
        # TODO: F3 generation needs optimization
        # counts how often A appears as B
        count = []
        A = ind['A'].to_list()
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

    def generate_F06():
        return get_column_name_similarity(conn)

    def generate_F08():
        return get_range(conn)

    def generate_F09():
        return get_typical_name_suffix(conn)

    def generate_F10():
        return get_table_size_ratio(conn)

    print('generating features')
    return aggregate_features(ind, [generate_F01(),
                                    generate_F02(),
                                    generate_F03(),
                                    generate_F04(),
                                    generate_F05(),
                                    generate_F06(),
                                    generate_F08(),
                                    generate_F09(),
                                    generate_F10()])[['Foreign_table', 'Foreign_key', 'Primary_table',
                                                      'Primary_key', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F8',
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

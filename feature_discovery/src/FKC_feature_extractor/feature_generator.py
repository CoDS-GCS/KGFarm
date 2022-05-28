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
        return list(map(lambda x: x/max_value, vector))

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

    def generate_F09():
        return get_typical_name_suffix(conn)

    def generate_F10():
        return get_table_size_ratio(conn)

    generate_F04()
    return aggregate_features(ind, [generate_F01(),
                                    generate_F02(),
                                    generate_F03(),
                                    generate_F04(),
                                    generate_F05(),
                                    generate_F06(),
                                    generate_F09(),
                                    generate_F10()])[['A', 'B', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F9', 'F10']]


def main():
    # This implementation is as per http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.438.9288&rep=rep1&type=pdf
    conn = connect_to_stardog(port=5820, db='chembl', show_status=True)
    # Get all inclusion dependencies (IND) initially
    ind = get_INDs(conn)
    # Generate features for these IND pairs
    features_df = generate_features(conn, ind)
    # print(features_df)
    features_df.to_csv('features.csv')


if __name__ == "__main__":
    main()

from feature_discovery.src.api.template import *
from helpers.helper import connect_to_stardog
from tqdm import tqdm
import pandas as pd


def generate_features(conn, ind: pd.DataFrame):
    def aggregate_features(ind_pairs: pd.DataFrame, features: list):
        # 1. join features, key = A & B
        # 2. if NAN i.e. edge didn't existed --> 0
        # 3. repeat for all features
        result = ind_pairs
        for feature in tqdm(features):
            result = pd.concat([result, feature], axis=1)
            result = result.loc[:, ~result.columns.duplicated()].fillna(0)
        return result

    def generate_F01():
        return get_distinct_dependent_values(conn)

    def generate_F02():
        return get_content_similarity(conn)

    def generate_F06():
        return get_column_name_similarity(conn)

    def generate_F09():
        return get_typical_name_suffix(conn)

    def generate_F10():
        return get_table_size_ratio(conn)

    return aggregate_features(ind, [generate_F01(),
                                    generate_F02(),
                                    generate_F06(),
                                    generate_F09(),
                                    generate_F10()])


def main():
    # This implementation is as per http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.438.9288&rep=rep1&type=pdf
    conn = connect_to_stardog(port=5822, db='chembl', show_status=True)
    # Get all inclusion dependencies (IND) initially
    ind = get_INDs(conn)
    # Generate features for these IND pairs
    features_df = generate_features(conn, ind)
    features_df.to_csv('features.csv')


if __name__ == "__main__":
    main()

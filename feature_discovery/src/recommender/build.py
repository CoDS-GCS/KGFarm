import os
import json
import pandas as pd
from helpers.helper import connect_to_stardog
from operations.template import get_transformations_on_columns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


class Recommender:
    """A classifier that recommends type of feature transformation based on column (feature) embeddings"""

    def __init__(self, feature_type: str, port: int = 5820, database: str = 'recommender',
                 metadata: str = '../../../helpers/sample_data/metadata/profiles/',
                 show_connection_status: bool = False):
        self.feature_type = feature_type
        self.metadata = metadata
        self.config = connect_to_stardog(port, database, show_connection_status)
        self.profiles = dict()  # column_id -> {column embeddings, column datatype}

    def generate_modelling_data(self):

        def get_profiles():
            for datatype in os.listdir(self.metadata):
                for profile_json in os.listdir(self.metadata + '/' + datatype):
                    if profile_json.endswith('.json'):
                        with open(self.metadata + '/' + datatype + '/' + profile_json, 'r') as open_file:
                            yield json.load(open_file)

        # load profiles in self.profiles
        for profile in get_profiles():
            dtype = profile['dataType']
            if 'N' in dtype:
                self.profiles['http://kglids.org/resource/'+profile['column_id']] = {'dtype': dtype, 'deep_embedding': profile['deep_embedding']}
            elif 'T' in dtype:
                self.profiles['http://kglids.org/resource/'+profile['column_id']] = {'dtype': dtype, 'minhash': profile['minhash']}
        print('{} profiles loaded.'.format(len(self.profiles)))

        # get transformations applied on real columns
        transformations_on_columns = get_transformations_on_columns(self.config)

        # generate modelling data
        transformations_on_columns['embeddings'] = transformations_on_columns['Column_id'] \
        .apply(lambda x: self.profiles.get(x))

    def define(self):
        pass

    def fit(self):
        pass

    @staticmethod
    def evaluate():
        pass
        return 0.0

    def save(self, f1_score: float):
        pass


def build():
    recommender = Recommender(feature_type='all')
    recommender.generate_modelling_data()
    recommender.define()
    recommender.fit()
    recommender.save(recommender.evaluate())
    print('done.')


if __name__ == '__main__':
    build()

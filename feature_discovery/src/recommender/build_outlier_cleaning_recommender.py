import json
import os
import warnings
import random

import joblib
from urllib.parse import quote_plus
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold
from helpers.helper import connect_to_stardog
from operations.template import get_cleaning_on_columns, get_outlier_cleaning
from sklearn.pipeline import Pipeline


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
RANDOM_STATE = 7

class Recommender:
    """A classifier that recommends type of feature transformation based on column (feature) embeddings"""

    def __init__(self, port: int = 5820, database: str = 'kaggle-pipeline-outlier',
                 metadata: str = '../../../operations/storage/CoLR_embeddings',
                 show_connection_status: bool = False):
        self.database = database
        self.metadata = metadata
        self.config = connect_to_stardog(port, self.database, show_connection_status)
        self.profiles = dict()  # column_id -> {column embeddings, column datatype}
        self.transformations = set()
        self.encoder = LabelEncoder()
        self.classifier =  RandomForestClassifier(random_state=RANDOM_STATE)
        self.modeling_data = None

    def generate_modeling_data(self, save: bool = True):

        def get_profiles():
            for datatype in os.listdir(self.metadata):
                if datatype == '.DS_Store':
                    continue
                for profile_json in os.listdir(self.metadata + '/' + datatype):
                    if profile_json.endswith('.json'):
                        with open(self.metadata + '/' + datatype + '/' + profile_json, 'r') as open_file:
                            yield json.load(open_file)

        def generate_table_id(profile_path: str, table_name: str):
            profile_path = profile_path.split('/')
            table_name = profile_path[-1]
            dataset_name = profile_path[-3]
            table_id = f'http://kglids.org/resource/kaggle/{quote_plus(dataset_name)}/dataResource/{quote_plus(table_name)}'
            return table_id


        """"
        Average the embedding of the numerical columns.
        """
        table_set = set()
        # Separate the embeddings based on dtype
        numerical_embeddings = {}
        for profile in get_profiles():
            dtype = profile['data_type']
            table_key = generate_table_id(profile['path'], profile['table_name'])
            table_set.add(table_key)

            if profile['embedding'] is not None:
            # Outlier detection is only applicable to numerical values
                if dtype == 'int' or dtype == 'float':
                    if table_key not in numerical_embeddings:
                        numerical_embeddings[table_key] = []
                    numerical_embeddings[table_key].append(profile['embedding'])


        for table in table_set:
            if table in numerical_embeddings:
                self.profiles[table] = np.mean(numerical_embeddings[table], axis=0)


        self.modeling_data = get_outlier_cleaning(self.config)
        print('og',self.modeling_data)
        self.modeling_data = self.modeling_data.merge(pd.DataFrame(self.profiles.items(), columns=['Table', 'emb']),
                                                      on='Table')
        print('2nd',self.modeling_data)

        # Keep only the 'Embeddings' column and add a new column 'Column1' with all elements set to 1
        result = self.modeling_data[['emb']].copy()
        result['Outliers'] = 1
        print('result',result)
        # To balance the dataset, we will add as many rows of embeddings for tables without outliers
        num_rows = len(self.modeling_data)
        additional_profiles = random.sample(self.profiles.items(), num_rows*2)
        print('add',additional_profiles)
        # index_values = []
        # for t in additional_profiles:
        #     print('t:',t[1])
        #     index_values.append(t[1])
        # print('index v',index_values)
        # data = {'http://kglids.org/resource/library/sklearn/ensemble/IsolationForest': 0,
        #                    'http://kglids.org/resource/library/sklearn/neighbors/LocalOutlierFactor': 0,
        #                    'http://kglids.org/resource/library/sklearn/svm/OneClassSVM': 0, 'NoOutliers': 1}
        # index_values = [np.array(t[1]) for t in additional_profiles]
        #
        # additional_df = pd.DataFrame(data, index=index_values)
        # additional_df = pd.DataFrame({'http://kglids.org/resource/library/sklearn/ensemble/IsolationForest': 0,
        #                    'http://kglids.org/resource/library/sklearn/neighbors/LocalOutlierFactor': 0,
        #                    'http://kglids.org/resource/library/sklearn/svm/OneClassSVM': 0, 'NoOutliers': 1},
        #                   index=[t[1] for t in additional_profiles])
        additional_df = pd.DataFrame({'emb': [np.array(t[1]) for t in additional_profiles], 'Outliers': 0})

        # additional_df.set_index('Embeddings', inplace=True)
        print('add2',additional_df)
        self.modeling_data = pd.concat([result, additional_df])

        if save:
            self.modeling_data.to_csv('modeling_data.csv', index=True)



    def train_test_evaluate_pivot(self, test):
        print(self.modeling_data)
        X = np.array(self.modeling_data['emb'])
        y = np.array(self.modeling_data['Outliers'])
        # For y to be integers instead of floats
        # y = y * 100
        # y = np.around(y)
        # y = y.astype(int)
        print('X',X)
        print('y',y)

        if test == True:
            # splitting the data to training and testing data set
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
            # Create a linear regression model
            self.classifier = RandomForestClassifier()
            self.classifier.fit((list(X_train)), y_train)
            # Make predictions on the test data
            y_pred = self.classifier.predict(list(X_test))
            f1 = f1_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            print("F1:", f1, 'accuracy', accuracy)
            return f1

        else:
            # Create a linear regression model
            self.classifier = RandomForestClassifier(n_estimators=100)
            self.classifier.fit(X, y)

    def save(self, scores: str, export: bool = True):
        print(scores)
        if export:
            if not os.path.exists('out'):
                os.mkdir('out')
            joblib.dump(self.classifier, 'out/outlier_cleaning_{}.pkl'.format('classification'), compress=9)


def build():
    recommender = Recommender()
    recommender.generate_modeling_data()
    recommender.save(scores=recommender.train_test_evaluate_pivot(True))
    print('done.')


if __name__ == '__main__':
    build()

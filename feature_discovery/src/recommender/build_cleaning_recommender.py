import csv
import json
import os
import warnings
import joblib
from urllib.parse import quote_plus
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold
from helpers.helper import connect_to_stardog
from operations.template import get_cleaning_on_columns
from sklearn.pipeline import Pipeline


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
RANDOM_STATE = 7

class Recommender:
    """A classifier that recommends type of feature transformation based on column (feature) embeddings"""

    def __init__(self, port: int = 5822, database: str = 'Niki-kaggle-pipeline',
                 metadata: str = '../../../operations/storage/metadata_Niki-kaggle',
                 #metadata: str = '../../../operations/storage/CoLR_embeddings',
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

        def clean_col():
            cleaning_table = get_cleaning_on_columns(self.config)
            #print('ct',cleaning_table)

            def change_value(row):
                if row['Cleaning_Op'] == 'http://kglids.org/resource/library/pandas/DataFrame/fillna':
                    if row['text'].__contains__('median'):
                        row['Cleaning_Op'] = 'fill-median'
                    elif row['text'].__contains__('mode'):
                        row['Cleaning_Op'] = 'fill-mode'
                    elif row['text'].__contains__('mean'):
                        row['Cleaning_Op'] = 'fill-mean'
                    elif row['text'].__contains__('ffill'):
                        row['Cleaning_Op'] = 'fill-ffill'
                    elif row['text'].__contains__('backfill'):
                        row['Cleaning_Op'] = 'fill-backfill'
                    elif row['text'].__contains__('bfill'):
                        row['Cleaning_Op'] = 'fill-bfill'
                    elif row['text'].__contains__('pad'):
                        row['Cleaning_Op'] = 'fill-pad'
                    elif row['text'].__contains__('fillna(0') or row['text'].__contains__('fillna(value=0') or\
                            row['text'].__contains__('fillna(value="0') or row['text'].__contains__("fillna(value='0") or\
                            row['text'].lower().__contains__('fillna("none') or row['text'].lower().__contains__("fillna('none") or\
                            row['text'].lower().__contains__('fillna("null') or row['text'].lower().__contains__("fillna('null") or \
                            row['text'].lower().__contains__('fillna("missing') or row['text'].lower().__contains__("fillna('missing") or \
                            row['text'].lower().__contains__('fillna("unknown') or row['text'].lower().__contains__("fillna('unknown") or \
                            row['text'].__contains__("fillna(''"):
                        row['Cleaning_Op'] = 'fill-outlier'
                    else:
                        row['Cleaning_Op'] = 'To be dropped'
                elif row['Cleaning_Op'] == 'http://kglids.org/resource/library/sklearn/impute/SimpleImputer':
                    if row['text'].__contains__('median'):
                        row['Cleaning_Op'] = 'SimpleImputer-median'
                    elif row['text'].__contains__('most_frequent'):
                        row['Cleaning_Op'] = 'SimpleImputer-most_frequent'
                    elif row['text'].__contains__('mean'):
                        row['Cleaning_Op'] = 'SimpleImputer-mean'
                    elif row['text'].__contains__('constant'):
                        row['Cleaning_Op'] = 'SimpleImputer-constant'
                    else:
                        row['Cleaning_Op'] = 'To be dropped'
                elif row['Cleaning_Op'] == 'http://kglids.org/resource/library/pandas/DataFrame/dropna':
                    row['Cleaning_Op'] = 'To be dropped'
                return row

            # apply the function to each row of the DataFrame
            cleaning_table = cleaning_table.apply(change_value, axis=1)
            cleaning_table = cleaning_table[cleaning_table['Cleaning_Op'] != 'To be dropped']
            return cleaning_table

        """"
        Average the embedding of the columns with missing values for categorical columns 
        and numerical columns separately. Concatenate the results to obtain a 600,1 embedding
        for the entire table.
        """
        table_set = set()
        # Separate the embeddings based on dtype
        string_embeddings = {}
        numerical_embeddings = {}
        for profile in get_profiles():
            dtype = profile['data_type']
            missing_value = profile['missing_values_count']

            # if missing_value == 0:
            #     continue

            table_key = generate_table_id(profile['path'], profile['table_name'])
            table_set.add(table_key)

            if profile['embedding'] is not None:
                if dtype == 'named_entity':
                    if table_key not in string_embeddings:
                        string_embeddings[table_key] = []
                    string_embeddings[table_key].append(profile['embedding'])
                elif dtype == 'int' or dtype == 'float':
                    if table_key not in numerical_embeddings:
                        numerical_embeddings[table_key] = []
                    numerical_embeddings[table_key].append(profile['embedding'])


        for table in table_set:
            if table in string_embeddings:
                string_embeddings_avg = np.mean(string_embeddings[table], axis=0)
            else:
                string_embeddings_avg = np.zeros(300)
            if table in numerical_embeddings:
                numerical_embeddings_avg = np.mean(numerical_embeddings[table], axis=0)
            else:
                numerical_embeddings_avg = np.zeros(300)

            self.profiles[table] = np.concatenate((string_embeddings_avg, numerical_embeddings_avg))

        for key, value in self.profiles.items():
            if isinstance(value, np.ndarray):
                self.profiles[key] = value.tolist()
        file_path = "model.csv"

        with open(file_path, "w", newline='') as f:
            writer = csv.writer(f)

            # Write each key-value pair as a row in the CSV file
            for key, value in self.profiles.items():
                writer.writerow([key, value])

        # with open('model.json', "w") as f:
        #     json.dump(self.profiles, f)

        self.modeling_data = clean_col()

        self.modeling_data.set_index('Table', inplace=True)
        self.modeling_data = self.modeling_data.dropna()
        self.modeling_data = pd.crosstab(index=self.modeling_data.index, columns=self.modeling_data['Cleaning_Op'],
                                         values=self.modeling_data['count'], aggfunc='sum', normalize='index')
        self.modeling_data = self.modeling_data.fillna(0)
        self.modeling_data['Embeddings'] = self.modeling_data.index.map(self.profiles)
        self.modeling_data = self.modeling_data.dropna()
        self.modeling_data.set_index('Embeddings', inplace=True)

        self.modeling_data.dropna(inplace=True)
        if save:
            self.modeling_data.to_csv('modeling_data.csv', index=True)



    def train_test_evaluate_pivot(self, test):
        #print(self.modeling_data)
        X = np.array(self.modeling_data.index)
        y = np.array(self.modeling_data.values)
        # For y to be integers instead of floats
        y = y * 100
        y = np.around(y)
        y = y.astype(int)

        if test == True:
            # splitting the data to training and testing data set
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
            # Create a linear regression model
            self.classifier = MultiOutputRegressor(RandomForestRegressor(n_estimators=100))
            self.classifier.fit((list(X_train)), y_train)
            # Make predictions on the test data
            y_pred = self.classifier.predict(list(X_test))
            mse = mean_squared_error(y_test, y_pred)
            print("Mean Squared Error:", mse)
            return mse

        else:
            # Create a linear regression model
            self.classifier = MultiOutputRegressor(RandomForestRegressor(n_estimators=100))
            self.classifier.fit((list(X)), y)

    def save(self, scores: str, export: bool = True):
        print(scores)
        if export:
            if not os.path.exists('out'):
                os.mkdir('out')
            joblib.dump(self.classifier, 'out/cleaning_{}.pkl'.format('test'), compress=9)


def build():
    recommender = Recommender()
    recommender.generate_modeling_data()
    recommender.save(scores=recommender.train_test_evaluate_pivot(True))
    print('done.')


if __name__ == '__main__':
    build()

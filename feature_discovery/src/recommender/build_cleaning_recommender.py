import json
import os
import warnings
import joblib
from urllib.parse import quote_plus

import numpy
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
# sys.path.append(r"C:\Datasets")
from helpers.helper import connect_to_stardog
# from word_embeddings import WordEmbedding
from operations.template import get_transformations_on_columns, get_cleaning_on_columns

from sklearn.pipeline import Pipeline


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


class Recommender:
    """A classifier that recommends type of feature transformation based on column (feature) embeddings"""

    def __init__(self, port: int = 5820, database: str = 'Niki-kaggle-pipeline',
                 metadata: str = '../../../operations/storage/CoLR_embeddings',
                 show_connection_status: bool = False):
        # self.feature_type = 'named_entity'
        self.database = database
        self.metadata = metadata
        self.config = connect_to_stardog(port, self.database, show_connection_status)
        self.profiles = dict()  # column_id -> {column embeddings, column datatype}
        # self.word_embedding_model = WordEmbedding()
        self.transformations = set()
        self.encoder = LabelEncoder()
        self.classifier = None
        self.modeling_data = None

    def generate_modeling_data(self):

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

        def generate_column_id(profile_path: str, column_name: str):
            profile_path = profile_path.split('/')
            table_name = profile_path[-1]
            dataset_name = profile_path[-3]
            column_id = f'http://kglids.org/resource/kaggle/{quote_plus(dataset_name)}/dataResource/{quote_plus(table_name)}/{quote_plus(column_name)}'
            return column_id

        # load profiles (based on feature-type) in self.profiles
        for profile in get_profiles():
            dtype = profile['data_type']
            # if dtype in 'int' or dtype == 'float':
            #     self.profiles[generate_table_id(profile['path'], profile['column_name'])] = {'dtype': 'numerical',
            #                                                                            'embeddings': profile['embedding']}
            # elif dtype == 'string' or dtype == 'named_entity':
            #     self.profiles[generate_table_id(profile['path'], profile['column_name'])] = {'dtype': 'categorical',
            #                                                                            'embeddings': profile['embedding']}
            missing_value = profile['missing_values_count']
            if missing_value == 0:
                continue
            else:
                # dictionary having for key the table_id and for value a tuple of the number of columns with missing values and the sum of their embeddings
                table_key = generate_table_id(profile['path'], profile['table_name'])
                if profile['embedding'] is not None:
                    if table_key in self.profiles:
                        table_value_count = self.profiles[table_key][0]
                        ###TO DO: Add embeddings
                        seq = np.array([self.profiles[table_key][1],profile['embedding']])
                        table_value_embeddings = np.sum(seq, axis=0)
                        #print(self.profiles[table_key][1][0:5], profile['embedding'][0:5], table_value_embeddings[0:5])
                        self.profiles[table_key] = (table_value_count+1,table_value_embeddings)
                        #print('Adding to ',table_key)
                    else:
                        table_value_embeddings = profile['embedding']
                        self.profiles[table_key] = ( 1, table_value_embeddings)
                        #print('creating a key for ', table_key)

        #Go through every dictionary element and average embeddings
        for key, value in self.profiles.items():
            self.profiles[key] = numpy.array(value[1])/value[0]


        # print('p', self.profiles)
        # Transform dict to df
        # df = pd.DataFrame([self.profiles])
        # print('df', df[0:5])
        def generate_col():
            cleaning_table = get_cleaning_on_columns(self.config)
            print('ct',cleaning_table)

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
                    elif row['text'].__contains__('fillna(0'):
                        row['Cleaning_Op'] = 'fill-0'
                    elif row['text'].lower().__contains__('fillna("none') or row['text'].lower().__contains__("fillna('none"):
                        row['Cleaning_Op'] = 'fill-None'
                    elif row['text'].__contains__("fillna(''"):
                        row['Cleaning_Op'] = 'fill-empty string'
                    else:
                        row['Cleaning_Op'] = 'fill'
                elif row['Cleaning_Op'] == 'http://kglids.org/resource/library/sklearn/impute/SimpleImputer':
                    if row['text'].__contains__('median'):
                        row['Cleaning_Op'] = 'SimpleImputer-median'
                    elif row['text'].__contains__('most_frequent'):
                        row['Cleaning_Op'] = 'Simple_imputer-most_frequent'
                    elif row['text'].__contains__('mean'):
                        row['Cleaning_Op'] = 'Simple_imputer-mean'
                    elif row['text'].__contains__('constant'):
                        row['Cleaning_Op'] = 'Simple_imputer-constant'
                    else:
                        row['Cleaning_Op'] = 'Simple_imputer'
                return row

            # apply the function to each row of the DataFrame
            cleaning_table = cleaning_table.apply(change_value, axis=1)
                    #if cleaning_table[i][1] == '':
                # print(cleaning_table.to_string())
            print(cleaning_table)
            return cleaning_table#['Table']

        self.modeling_data = generate_col()
        # self.modeling_data.drop_duplicates(subset=['Table'], keep='first',inplace=True)
        self.modeling_data.set_index('Table', inplace=True)
        # self.modeling_data['Embeddings'] = self.modeling_data.index.map(self.profiles)
        self.modeling_data = self.modeling_data.dropna()
        # self.modeling_data['Embeddings'] = pd.array(self.modeling_data['Embeddings'].tolist())
        # self.modeling_data.set_index('Embeddings', inplace=True)

        print('11111111',self.modeling_data.to_string())
        # self.modeling_data = self.modeling_data.drop('Table', axis=1)
        # Drop the table column
        # self.modeling_data = self.modeling_data.drop('Table', axis=1)

        # print('22222222222', self.modeling_data.to_string())
        # Create the pivot table
        # print('table',self.modeling_data.index)
        self.modeling_data = pd.crosstab(index=self.modeling_data.index, columns=self.modeling_data['Cleaning_Op'], values=self.modeling_data['count'], aggfunc='sum', normalize='index')
        # self.modeling_data['http://kglids.org/resource/library/sklearn/impute/SimpleImputer'] = self.modeling_data['http://kglids.org/resource/library/sklearn/impute/SimpleImputer'].astype(bool)
        self.modeling_data = self.modeling_data.fillna(0)
        self.modeling_data['Embeddings'] = self.modeling_data.index.map(self.profiles)
        self.modeling_data = self.modeling_data.dropna()
        self.modeling_data.set_index('Embeddings', inplace=True)
        # print('333333333333', self.modeling_data)
        # print(self.modeling_data.index)
        # new_df = pd.get_dummies(self.modeling_data['Cleaning_Op'])
        # print(new_df.to_string())

        # self.modeling_data = self.modeling_data.astype(bool).astype(int)
        # print(self.modeling_data.to_string())

        # result = pd.concat([generate_col(), df], axis=1)
        #print(result[0:5])
        # df = pd.DataFrame(data=[self.profiles])
        # print('HHHHHHHHHHHH',df)
        # print(f'{len(self.profiles)} {self.feature_type} profiles loaded from {self.metadata}')
        #
        # # get transformations applied on real columns
        # transformations_on_columns = get_transformations_on_columns(self.config)
        # print(f'found {len(transformations_on_columns)} distinct feature-transformations by querying {self.database}')
        #
        # # associate embeddings and datatype for transformed columns
        # transformations_on_columns['Data_type'] = transformations_on_columns['Column_id'] \
        #     .apply(lambda x: self.profiles.get(x).get('dtype') if self.profiles.get(x) else None)
        # transformations_on_columns['Embeddings'] = transformations_on_columns['Column_id'] \
        #     .apply(lambda x: self.profiles.get(x).get('embeddings') if self.profiles.get(x) else None)
        # self.modeling_data = transformations_on_columns

        # add untransformed columns to modeling data
        """ 
        transformed_columns = set(list(self.modeling_data['Column_id']))
        for column, profile in self.profiles.items():
            if column not in transformed_columns:
                self.modeling_data = self.modeling_data.append({'Transformation': 'Negative', 'Column_id': column,
                                                                'Data_type': self.feature_type,
                                                                'Embeddings': profile['embeddings']}, ignore_index=True)
        """

        # add column word-embeddings to modeling data
        """ 
        self.modeling_data['Word_embedding'] = self.modeling_data['Column_id'].apply(
            lambda x: self.word_embedding_model.get_embeddings(x))
        """
        # self.modeling_data.dropna(how='any', inplace=True)
        # self.modeling_data.to_csv('modeling_data_{}.csv'.format(self.feature_type), index=False)

    def prepare(self, plot: bool = True, save: bool = True, balance: bool = True):

        def pre_process_and_clean():
            # Delete rows with Nan values
            self.modeling_data.dropna(inplace=True)

            # Drop the table column
            # self.modeling_data = self.modeling_data.drop('Table', axis=1)


            # for index, row in tqdm(self.modeling_data.to_dict('index').items()):
            #     """
            #     Remove record if:
            #     1. Embeddings / Word embeddings not found
            #     2. Embeddings are incorrect
            #     3. Any scaling technique is used for categorical features
            #     and
            #     Re-map transformations
            #     """
            #     if not row['Embeddings'] or \
            #             row['Embeddings'][:10] == [-1] * 10 or \
            #             ('Scaler' in row['Transformation'] and self.feature_type == 'categorical') or \
            #             row['Transformation'] not in transformation_mapping.get(self.feature_type):
            #         self.modeling_data.drop(index=index, inplace=True)
            #     else:
            #         self.modeling_data.loc[index, 'Transformation'] = transformation_mapping.get(self.feature_type).get(
            #             row['Transformation'])
            #
            # # self.modeling_data.to_csv('modeling_data_{}.csv'.format(self.feature_type), index=False)
            # self.modeling_data.drop(['Column_id', 'Data_type'], axis=1, inplace=True)

        def plot_class_distribution():
            plt.rcParams['figure.dpi'] = 300
            sns.set_style('dark')
            fig, ax = plt.subplots(figsize=(8.5, 5))
            sns.countplot(x='Cleaning_Op', data=self.modeling_data, palette="Greens_r",
                          order=self.modeling_data['Cleaning_Op'].value_counts().index)
            plt.grid(color='gray', linestyle='dashed', axis='y')
            plt.ylabel('columns')
            plt.xlabel(f'Cleaning_Op')
            ax.bar_label(ax.containers[0])

            def change_width(axis, new_value):
                for patch in axis.patches:
                    current_width = patch.get_width()
                    diff = current_width - new_value
                    patch.set_width(new_value)
                    patch.set_x(patch.get_x() + diff * .5)

            change_width(ax, .35)
            ax.tick_params(axis='x', labelsize=9, rotation=35)
            fig.tight_layout()
            plt.show()

        # def transform():
        #     # TODO: cache label-encoder else unify transformation mapping
        #     # convert transformations
        #     self.modeling_data['Transformation'] = self.encoder.fit_transform(self.modeling_data['Transformation'])
        #
        def balance_classes():
            transformation_statistics = self.modeling_data['Cleaning_Op'].value_counts()
            transformation_statistics = dict(transformation_statistics)
            # lowest_occurring_transformation = min(transformation_statistics, key=transformation_statistics.get)
            np.random.seed(1)
            for transformation_class in transformation_statistics.keys():
                if transformation_class == 'http://kglids.org/resource/library/pandas/DataFrame/dropna' or transformation_class == 'http://kglids.org/resource/library/pandas/DataFrame/fillna':
                    drop = np.random.choice(
                        self.modeling_data[self.modeling_data['Cleaning_Op'] == transformation_class].index,
                        size=transformation_statistics.get(transformation_class)-transformation_statistics.get('http://kglids.org/resource/library/sklearn/impute/SimpleImputer'), replace=False)
                    self.modeling_data.drop(drop, inplace=True)

        """
        def concatenate_embeddings(rows):
            embedding = []
            embedding.extend(rows['Embeddings'])
            embedding.extend(rows['Word_embedding'])
            return embedding
        """

        pre_process_and_clean()

        # if balance:
        #     balance_classes()
        #
        # print(f"Class distribution: {pd.DataFrame(self.modeling_data['Cleaning_Op'].value_counts())}")

        """
        concatenate embeddings (column + word-embeddings)
        self.modeling_data['Embeddings'] = self.modeling_data.apply(concatenate_embeddings, axis=1)
        self.modeling_data.drop('Word_embedding', axis=1, inplace=True)
        """

        # self.transformations = set(list(self.modeling_data['Transformation']))

        # if plot:
        #     plot_class_distribution()

        # transform()
        #
        # print(f'{self.feature_type}: {len(self.modeling_data)}')

        if save:
            self.modeling_data.to_csv('modeling_data.csv', index=False)

    def define(self):
        # if self.feature_type == 'categorical':
        #     self.classifier = MLPClassifier(max_iter=20, activation='relu', solver='adam', learning_rate='adaptive')
        #     hyperparameters = {'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        #                    'activation': ['tanh', 'relu'],
        #                    'solver': ['sgd', 'adam'],
        #                    'alpha': [0.0001, 0.05],
        #                    'learning_rate': ['constant', 'adaptive']}
        # else:
        self.classifier = LogisticRegression() #RandomForestClassifier()
        self.classifier = OneVsRestClassifier(self.classifier)
        hyperparameters = {}
        return hyperparameters

    def train_test_evaluate(self, test):#(self, parameters: dict, tune: bool = False):

        # def optimize():
        #     self.classifier = GridSearchCV(estimator=self.classifier, param_grid=parameters, cv=inner_cv)
        #
        # def evaluate():
        #     y_true = []
        #     y_pred = []
        #
        #     def score(y_true_label, y_pred_label):
        #         y_true.extend(y_true_label)
        #         y_pred.extend(y_pred_label)
        #
        #     cross_val_score(self.classifier, X, y, cv=outer_cv, scoring=make_scorer(score))
        #     return y_true, y_pred
        print(self.modeling_data)
        X = np.array(self.modeling_data.index)
        y = np.array(self.modeling_data.values)#loc[:, ['http://kglids.org/resource/library/pandas/DataFrame/interpolate','http://kglids.org/resource/library/pandas/DataFrame/fillna','http://kglids.org/resource/library/pandas/DataFrame/dropna','http://kglids.org/resource/library/sklearn/impute/SimpleImputer','http://kglids.org/resource/library/sklearn/impute/KNNImputer','http://kglids.org/resource/library/sklearn/impute/IterativeImputer']]
        # For y to be integers instead of floats
        print('ynf', y)
        y = y * 100
        y = np.around(y)
        y = y.astype(int)
        print('y', y)
        #print('X',X)

        # initializing TfidfVectorizer
        # vetorizar = TfidfVectorizer(max_features=3000, max_df=0.85)
        # # fitting the tf-idf on the given data
        # vetorizar.fit(X)
        if test == True:
            # splitting the data to training and testing data set
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

            # Create a linear regression model
            self.classifier = RandomForestRegressor(n_estimators=100)
            self.classifier.fit((list(X_train)), y_train)
            # Make predictions on the test data
            y_pred = self.classifier.predict(list(X_test))
            print(y_pred)
            mse = mean_squared_error(y_test, y_pred)
            print("Mean Squared Error:", mse)

        else:
            # Create a linear regression model
            self.classifier = RandomForestRegressor(n_estimators=100)
            self.classifier.fit((list(X)), y)


        # def flatten_array(X):
        #     return np.array([np.array(i).flatten() for i in X])
        #
        # # Define the pipeline
        # pipeline = Pipeline([
        #     ('vec', FunctionTransformer(flatten_array)),
        #     ('clf', MultiOutputClassifier(LinearRegression()))
        # ])

        # # Define the normalization function
        # normalize = lambda X: (X - np.min(X)) / (np.max(X) - np.min(X))
        #
        # # Create a FunctionTransformer with the normalization function
        # normalizer = FunctionTransformer(normalize)
        #
        # # Create a linear regression model
        # model = LinearRegression()
        #
        # # Create a pipeline with the normalizer and the model
        # pipeline = Pipeline([('normalizer', normalizer), ('model', model)])
        # Fit the model on the training data


        # accuracy = accuracy_score(y_test, y_pred)
        # precision = precision_score(y_test, y_pred, average='macro')
        # recall = recall_score(y_test, y_pred, average='macro')
        # f1 = f1_score(y_test, y_pred, average='macro')
        #
        # print("Accuracy: ", accuracy)
        # print("Precision: ", precision)
        # print("Recall: ", recall)
        # print("F1-score: ", f1)
        ### Works for classification
        # # define a function that flattens the array
        # def flatten_array(X):
        #     return np.array([np.array(i).flatten() for i in X])
        #
        # # Define the pipeline
        # pipeline = Pipeline([
        #     ('vec', FunctionTransformer(flatten_array)),
        #     ('clf', MultiOutputClassifier(LogisticRegression()))
        # ])
        #
        # # Fit the pipeline on the training data
        # pipeline.fit(X_train, y_train)
        #
        # # Predict the labels for the test data
        # y_pred = pipeline.predict(X_test)
        # print(y_pred)
        # hamming_loss_value = hamming_loss(y_test, y_pred)
        # print("Hamming loss:", hamming_loss_value)

        # print('x',X)
        # print('y',y)
        # cv = StratifiedKFold(n_splits=5)
        # y_pred = cross_val_predict(self.classifier, X, y, cv=cv)
        # accuracy = accuracy_score(y, y_pred)
        # precision = precision_score(y, y_pred, average='macro')
        # recall = recall_score(y, y_pred, average='macro')
        # f1 = f1_score(y, y_pred, average='macro')
        #
        # print("Accuracy: ", accuracy)
        # print("Precision: ", precision)
        # print("Recall: ", recall)
        # print("F1-score: ", f1)



        # Nested CV with parameter optimization
        # inner_cv = KFold(n_splits=5, shuffle=True, random_state=0) #StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        # outer_cv = KFold(n_splits=5, shuffle=True, random_state=0) #StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

        # if tune:
        #     optimize()  # Hyperparameter optimization (using nested CV to prevent leakage)
        # true, pred = evaluate()

        # self.classifier.fit(X, y)  # fitting is done after results are evaluated i.e. no info leakage

        # Average scores over all folds
        # classification_report(y_true=true, y_pred=pred)
        # confusion_matrix(y_true=true, y_pred=pred)
        # return classification_report(y_true=true, y_pred=pred)
                                     # labels=self.encoder.transform(list(self.transformations)),
                                     # target_names=list(self.transformations))

    def save(self, scores: str, export: bool = True):
        print(scores)
        if export:
            if not os.path.exists('out'):
                os.mkdir('out')
            joblib.dump(self.classifier, 'out/cleaning_{}.pkl'.format('Table_fill'), compress=9)
            # joblib.dump(self.encoder, 'out/encoder_{}.pkl'.format(self.feature_type), compress=9)
            # print('model saved transformation_recommender_{}.pkl'.format(self.feature_type))
            # joblib.dump(self.classifier, 'out/transformation_recommender_{}.pkl'.format(self.feature_type),
            #             compress=9)


transformation_mapping = {'categorical':
                              {'LabelEncoder': 'Ordinal encoding',
                               'LabelBinarizer': 'Nominal encoding',
                               'OrdinalEncoder': 'Ordinal encoding',
                               'OneHotEncoder': 'Nominal encoding',
                               'label_binarize': 'Nominal encoding'},
                          'numerical':
                              {'StandardScaler': 'Scaling',
                               'scale': 'Scaling',
                               'MinMaxScaler': 'Normalization',
                               'normalize': 'Normalization',
                               'Normalizer': 'Normalization',
                               'minmax_scale': 'Normalization',
                               'RobustScaler': 'Scaling concerning outliers',
                               'robust_scale': 'Scaling concerning outliers',
                               'PowerTransformer': 'Gaussian distribution',
                               'power_transform': 'Gaussian distribution',
                               'LabelEncoder': 'Ordinal encoding',
                               'OrdinalEncoder': 'Ordinal encoding',
                               'OneHotEncoder': 'Nominal encoding'}}


def build():
    # TODO: plot all metrics for both feature types
    # for feature_type in ['categorical', 'numerical']:
    recommender = Recommender()
    recommender.generate_modeling_data()
    recommender.prepare(plot=True, save=True, balance=True)
    recommender.save(scores=recommender.train_test_evaluate(False))#parameters=recommender.define(), tune=False),
                       #  export=True)
    print('done.')


if __name__ == '__main__':
    build()

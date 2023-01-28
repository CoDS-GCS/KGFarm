import json
import os
import warnings
import joblib
from urllib.parse import quote_plus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, make_scorer
from helpers.helper import connect_to_stardog
# from word_embeddings import WordEmbedding
from operations.template import get_transformations_on_columns

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


class Recommender:
    """A classifier that recommends type of feature transformation based on column (feature) embeddings"""

    def __init__(self, feature_type: str, port: int = 5820, database: str = 'kgfarm_recommender',
                 metadata: str = '../../../operations/storage/CoLR_embeddings/',
                 show_connection_status: bool = False):
        self.feature_type = feature_type
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

        def generate_column_id(profile_path: str, column_name: str):
            profile_path = profile_path.split('/')
            table_name = profile_path[-1]
            dataset_name = profile_path[-3]
            column_id = f'http://kglids.org/resource/kaggle/{quote_plus(dataset_name)}/dataResource/{quote_plus(table_name)}/{quote_plus(column_name)}'
            return column_id

        # load profiles (based on feature-type) in self.profiles
        for profile in get_profiles():
            dtype = profile['data_type']
            if dtype in 'int' or dtype == 'float':
                self.profiles[generate_column_id(profile['path'], profile['column_name'])] = {'dtype': 'numerical',
                                                                                       'embeddings': profile['embedding']}
            elif dtype == 'string' or dtype == 'named_entity':
                self.profiles[generate_column_id(profile['path'], profile['column_name'])] = {'dtype': 'categorical',
                                                                                       'embeddings': profile['embedding']}

        print(f'{len(self.profiles)} {self.feature_type} profiles loaded from {self.metadata}')

        # get transformations applied on real columns
        transformations_on_columns = get_transformations_on_columns(self.config)
        print(f'found {len(transformations_on_columns)} distinct feature-transformations by querying {self.database}')

        # associate embeddings and datatype for transformed columns
        transformations_on_columns['Data_type'] = transformations_on_columns['Column_id'] \
            .apply(lambda x: self.profiles.get(x).get('dtype') if self.profiles.get(x) else None)
        transformations_on_columns['Embeddings'] = transformations_on_columns['Column_id'] \
            .apply(lambda x: self.profiles.get(x).get('embeddings') if self.profiles.get(x) else None)
        self.modeling_data = transformations_on_columns

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
            for index, row in tqdm(self.modeling_data.to_dict('index').items()):
                """
                Remove record if:
                1. Embeddings / Word embeddings not found
                2. Embeddings are incorrect
                3. Any scaling technique is used for categorical features
                and 
                Re-map transformations
                """
                if not row['Embeddings'] or \
                        row['Embeddings'][:10] == [-1] * 10 or \
                        ('Scaler' in row['Transformation'] and self.feature_type == 'categorical') or \
                        row['Transformation'] not in transformation_mapping.get(self.feature_type):
                    self.modeling_data.drop(index=index, inplace=True)
                else:
                    self.modeling_data.loc[index, 'Transformation'] = transformation_mapping.get(self.feature_type).get(
                        row['Transformation'])

            # self.modeling_data.to_csv('modeling_data_{}.csv'.format(self.feature_type), index=False)
            self.modeling_data.drop(['Column_id', 'Data_type'], axis=1, inplace=True)

        def plot_class_distribution():
            plt.rcParams['figure.dpi'] = 300
            sns.set_style('dark')
            fig, ax = plt.subplots(figsize=(8.5, 5))
            sns.countplot(x='Transformation', data=self.modeling_data, palette="Greens_r",
                          order=self.modeling_data['Transformation'].value_counts().index)
            plt.grid(color='gray', linestyle='dashed', axis='y')
            plt.ylabel('columns')
            plt.xlabel(f'{self.feature_type} transformations')
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

        def transform():
            # TODO: cache label-encoder else unify transformation mapping
            # convert transformations
            self.modeling_data['Transformation'] = self.encoder.fit_transform(self.modeling_data['Transformation'])

        def balance_classes():
            transformation_statistics = self.modeling_data['Transformation'].value_counts()
            transformation_statistics = dict(transformation_statistics)
            # lowest_occurring_transformation = min(transformation_statistics, key=transformation_statistics.get)
            np.random.seed(1)
            for transformation_class in transformation_statistics.keys():
                if transformation_class == 'Scaling' or transformation_class == 'Ordinal encoding':
                    drop = np.random.choice(
                        self.modeling_data[self.modeling_data['Transformation'] == transformation_class].index,
                        size=transformation_statistics.get(transformation_class)-transformation_statistics.get('Normalization'), replace=False)
                    self.modeling_data.drop(drop, inplace=True)

        """
        def concatenate_embeddings(rows):
            embedding = []
            embedding.extend(rows['Embeddings'])
            embedding.extend(rows['Word_embedding'])
            return embedding
        """

        pre_process_and_clean()

        if balance and self.feature_type == 'numerical':
            balance_classes()

        print(f"Class distribution: {pd.DataFrame(self.modeling_data['Transformation'].value_counts())}")

        """
        concatenate embeddings (column + word-embeddings)
        self.modeling_data['Embeddings'] = self.modeling_data.apply(concatenate_embeddings, axis=1)
        self.modeling_data.drop('Word_embedding', axis=1, inplace=True)
        """

        self.transformations = set(list(self.modeling_data['Transformation']))

        if plot:
            plot_class_distribution()

        transform()

        print(f'{self.feature_type}: {len(self.modeling_data)}')

        if save:
            self.modeling_data.to_csv('modeling_data_{}.csv'.format(self.feature_type), index=False)

    def define(self):
        if self.feature_type == 'categorical':
            self.classifier = MLPClassifier(max_iter=20, activation='relu', solver='adam', learning_rate='adaptive')
            hyperparameters = {'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
                           'activation': ['tanh', 'relu'],
                           'solver': ['sgd', 'adam'],
                           'alpha': [0.0001, 0.05],
                           'learning_rate': ['constant', 'adaptive']}
        else:
            self.classifier = RandomForestClassifier()
            hyperparameters = {}
        return hyperparameters

    def train_test_evaluate(self, parameters: dict, tune: bool = False):

        def optimize():
            self.classifier = GridSearchCV(estimator=self.classifier, param_grid=parameters, cv=inner_cv)

        def evaluate():
            y_true = []
            y_pred = []

            def score(y_true_label, y_pred_label):
                y_true.extend(y_true_label)
                y_pred.extend(y_pred_label)

            cross_val_score(self.classifier, X, y, cv=outer_cv, scoring=make_scorer(score))
            return y_true, y_pred

        X = list(self.modeling_data['Embeddings'])
        y = self.modeling_data['Transformation']

        # Nested CV with parameter optimization
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

        if tune:
            optimize()  # Hyperparameter optimization (using nested CV to prevent leakage)
        true, pred = evaluate()

        self.classifier.fit(X, y)  # fitting is done after results are evaluated i.e. no info leakage

        # Average scores over all folds
        return classification_report(y_true=true, y_pred=pred,
                                     labels=self.encoder.transform(list(self.transformations)),
                                     target_names=list(self.transformations))

    def save(self, scores: str, export: bool = True):
        print(scores)
        if export:
            if not os.path.exists('out'):
                os.mkdir('out')
            joblib.dump(self.encoder, 'out/encoder_{}.pkl'.format(self.feature_type), compress=9)
            print('model saved transformation_recommender_{}.pkl'.format(self.feature_type))
            joblib.dump(self.classifier, 'out/transformation_recommender_{}.pkl'.format(self.feature_type),
                        compress=9)


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
    for feature_type in ['categorical', 'numerical']:
        recommender = Recommender(feature_type=feature_type)
        recommender.generate_modeling_data()
        recommender.prepare(plot=True, save=True, balance=True)
        recommender.save(scores=recommender.train_test_evaluate(parameters=recommender.define(), tune=False),
                         export=True)
    print('done.')


if __name__ == '__main__':
    build()

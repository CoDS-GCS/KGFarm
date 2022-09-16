import os
import json
import joblib
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, make_scorer
from helpers.helper import connect_to_stardog
from operations.template import get_transformations_on_columns

pd.set_option('display.max_columns', None)


class Recommender:
    """A classifier that recommends type of feature transformation based on column (feature) embeddings"""

    def __init__(self, feature_type: str, port: int = 5820, database: str = 'recommender',
                 metadata: str = '../../../helpers/sample_data/metadata/profiles/',
                 show_connection_status: bool = False):
        self.feature_type = feature_type
        self.metadata = metadata
        self.config = connect_to_stardog(port, database, show_connection_status)
        self.profiles = dict()  # column_id -> {column embeddings, column datatype}
        self.transformations = set()
        self.encoder = LabelEncoder()
        self.classifier = None
        self.modeling_data = None

    def generate_modeling_data(self):

        def get_profiles():
            for datatype in os.listdir(self.metadata):
                for profile_json in os.listdir(self.metadata + '/' + datatype):
                    if profile_json.endswith('.json'):
                        with open(self.metadata + '/' + datatype + '/' + profile_json, 'r') as open_file:
                            yield json.load(open_file)

        # load profiles (based on feature-type) in self.profiles
        for profile in get_profiles():
            dtype = profile['dataType']
            if 'N' in dtype and self.feature_type == 'numeric':
                self.profiles['http://kglids.org/resource/' + profile['column_id']] = {'dtype': 'numeric',
                                                                                       'embeddings': profile[
                                                                                           'deep_embedding']}
            elif 'T' in dtype and self.feature_type == 'string':
                self.profiles['http://kglids.org/resource/' + profile['column_id']] = {'dtype': 'string',
                                                                                       'embeddings': profile['minhash']}
        print('{} {} profiles loaded.'.format(len(self.profiles), self.feature_type))

        # get transformations applied on real columns
        transformations_on_columns = get_transformations_on_columns(self.config)

        # associate embeddings and datatype for transformed columns
        transformations_on_columns['Data_type'] = transformations_on_columns['Column_id'] \
            .apply(lambda x: self.profiles.get(x).get('dtype') if self.profiles.get(x) else None)
        transformations_on_columns['Embeddings'] = transformations_on_columns['Column_id'] \
            .apply(lambda x: self.profiles.get(x).get('embeddings') if self.profiles.get(x) else None)
        self.modeling_data = transformations_on_columns

        # add untransformed columns to modeling data
        transformed_columns = set(list(self.modeling_data['Column_id']))
        for column, profile in tqdm(self.profiles.items()):
            if column not in transformed_columns:
                self.modeling_data = self.modeling_data.append({'Transformation': 'Negative', 'Column_id': column,
                'Data_type': self.feature_type, 'Embeddings': profile['embeddings']}, ignore_index=True)

    def prepare(self, plot: bool = True, save: bool = True):

        def clean():
            for index, row in self.modeling_data.to_dict('index').items():
                """
                Remove record if:
                1. Embeddings not found
                2. Embeddings are incorrect
                3. Sklearn.preprocessing.scale() is used
                4. Any scaling technique is used for string/categorical features
                """
                if not row['Embeddings'] or row['Transformation'] == 'scale'or row['Embeddings'][:10] == [-1]*10 or ('Scaler' in row['Transformation'] and self.feature_type == 'string'):
                    self.modeling_data.drop(index=index, inplace=True)
            self.modeling_data.drop(['Column_id', 'Data_type'], axis=1, inplace=True)

        def balance():  # take as many non-transformed samples as the most used of transformation technique
            transformation_statistics = self.modeling_data['Transformation'].value_counts()
            transformation_counts = list(transformation_statistics)
            transformation_counts.sort(reverse=True)
            n = transformation_statistics['Negative'] - transformation_counts[1]
            drop = np.random.choice(self.modeling_data[self.modeling_data['Transformation'] == 'Negative'].index,
                                    size=n, replace=False)
            self.modeling_data = self.modeling_data.drop(drop)

        def plot_class_distribution():
            plt.rcParams['figure.dpi'] = 200
            sns.set_style('dark')
            fig, ax = plt.subplots(figsize=(7, 3.5))
            sns.countplot(x='Transformation', data=self.modeling_data, palette="Greens_r",
                          order=self.modeling_data['Transformation'].value_counts().index)
            plt.grid(color='gray', linestyle='dashed', axis='y')
            plt.ylabel('No. of columns')
            plt.xlabel('Transformation type (for {} features)'.format(self.feature_type))
            ax.bar_label(ax.containers[0])

            def change_width(axis, new_value):
                for patch in axis.patches:
                    current_width = patch.get_width()
                    diff = current_width - new_value
                    patch.set_width(new_value)
                    patch.set_x(patch.get_x() + diff * .5)

            change_width(ax, .65)
            fig.tight_layout()
            plt.show()

        def transform():
            # TODO: cache label-encoder else unify transformation mapping
            # convert transformations
            self.modeling_data['Transformation'] = self.encoder.fit_transform(self.modeling_data['Transformation'])

        clean()
        balance()
        self.transformations = set(list(self.modeling_data['Transformation']))
        if plot:
            plot_class_distribution()
        transform()

        if save:
            self.modeling_data.to_csv('modeling_data_{}.csv'.format(self.feature_type), index=False)

    def define(self):
        self.classifier = RandomForestClassifier()
        hyperparameters = {'n_estimators': [100, 200]}
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
        inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
        outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

        if tune:
            optimize()  # Hyperparameter optimization (using nested CV to prevent leakage)
        true, pred = evaluate()

        self.classifier.fit(X, y)  # fitting is done after results are evaluated i.e. no info leakage

        # Average scores over all folds
        return classification_report(y_true=true, y_pred=pred,
                                     labels=self.encoder.transform(list(self.transformations)),
                                     target_names=list(self.transformations))

    def save(self, scores: str, cache: bool = True):
        print(scores)
        if cache:
            if not os.path.exists('cache'):
                os.mkdir('cache')
            joblib.dump(self.encoder, 'cache/encoder_{}.pkl'.format(self.feature_type), compress=9)
            print('model saved transformation_recommender_{}.pkl'.format(self.feature_type))
            joblib.dump(self.classifier, 'cache/transformation_recommender_{}.pkl'.format(self.feature_type), compress=9)


def build():
    # TODO: plot all metrics for both feature types
    for feature_type in ['string', 'numeric']:
        recommender = Recommender(feature_type=feature_type)
        recommender.generate_modeling_data()
        recommender.prepare(plot=False, save=False)
        recommender.save(scores=recommender.train_test_evaluate(parameters=recommender.define(), tune=False), cache=False)
    print('done.')


if __name__ == '__main__':
    build()

import os
import json
import warnings
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score,  classification_report
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
                self.profiles['http://kglids.org/resource/' + profile['column_id']] = {'dtype': 'numeric',
                                                                                       'embeddings': profile[
                                                                                           'deep_embedding']}
            elif 'T' in dtype:
                self.profiles['http://kglids.org/resource/' + profile['column_id']] = {'dtype': 'string',
                                                                                       'embeddings': profile['minhash']}
        print('{} profiles loaded.'.format(len(self.profiles)))

        # get transformations applied on real columns
        transformations_on_columns = get_transformations_on_columns(self.config)

        # generate modelling data (fetch data-type and embeddings)
        transformations_on_columns['Data_type'] = transformations_on_columns['Column_id'] \
            .apply(lambda x: self.profiles.get(x).get('dtype') if self.profiles.get(x) else None)
        transformations_on_columns['Embeddings'] = transformations_on_columns['Column_id'] \
            .apply(lambda x: self.profiles.get(x).get('embeddings') if self.profiles.get(x) else None)

        embeddings = list(transformations_on_columns['Embeddings'])
        embeddings = [e for e in embeddings if e is not None]

        print('{} embeddings found out of {}'.format(len(embeddings), len(transformations_on_columns)))

        # filter dataset based on feature data-type
        transformations_on_columns = transformations_on_columns. \
            loc[transformations_on_columns['Data_type'] == self.feature_type]
        print('{} {} samples found'.format(len(transformations_on_columns), self.feature_type))

        self.modeling_data = transformations_on_columns

        transformations_on_columns.to_csv('data.csv', index=False)

    def eda(self):

        def clean():
            self.modeling_data.drop(['Column_id', 'Data_type'], axis=1, inplace=True)
            for index, row in self.modeling_data.to_dict('index').items():
                if not row['Embeddings'] or row['Transformation'] == 'scale':
                    self.modeling_data.drop(index=index, inplace=True)

        def plot_class_distribution():
            plt.rcParams['figure.dpi'] = 200
            sns.set_style("dark")
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
        self.transformations = set(list(self.modeling_data['Transformation']))
        plot_class_distribution()
        transform()

        self.modeling_data.to_csv('data.csv', index=False)

    def define(self):
        self.classifier = RandomForestClassifier()

    def fit(self):
        X = list(self.modeling_data['Embeddings'])
        y = self.modeling_data['Transformation']

        # TODO: use cross-validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
        self.classifier.fit(X_train, y_train)
        return X_test, y_test

    def evaluate(self, testing_data: tuple):
        X_test = testing_data[0]
        y_test = testing_data[1]
        accuracy = self.classifier.score(X_test, y_test)
        y_pred = self.classifier.predict(X_test)

        # TODO: plot all metrics for both feature types
        precision = precision_score(y_true=y_test, y_pred=y_pred, average='macro')
        recall = recall_score(y_true=y_test, y_pred=y_pred, average='macro')
        f1 = f1_score(y_true=y_test, y_pred=y_pred, average='macro')
        print(classification_report(y_true=y_test, y_pred=y_pred,
                                    labels=self.encoder.transform(list(self.transformations)),
                                    target_names=list(self.transformations)))
        print('Accuracy: ', accuracy)
        print('Precision: ', precision)
        print('Recall: ', recall)
        print('F1:  ', f1)

        return accuracy, precision, recall, f1

    def save(self, scores: tuple):
        pass


def build():
    for feature_type in ['numeric', 'string']:
        recommender = Recommender(feature_type=feature_type)
        recommender.generate_modelling_data()
        recommender.eda()
        recommender.define()
        recommender.save(scores=recommender.evaluate(testing_data=recommender.fit()))
    print('done.')


if __name__ == '__main__':
    build()

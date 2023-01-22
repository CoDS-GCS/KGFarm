import json
import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from operations.template import get_features_and_targets
from helpers.helper import connect_to_stardog, generate_column_id, generate_table_id


class FeatureSelector:
    """A Feature selection model which takes a feature and target embedding as input and
     gives the selection confidence as output"""

    def __init__(self, port: int = 5820, database: str = 'kaggle',
                 metadata='../../storage/CoLR_embeddings/', show_connection_status: bool = True):
        self.config = connect_to_stardog(port, database, show_connection_status)
        self.metadata = metadata
        self.classifier = RandomForestClassifier()

        self.embeddings_table_to_column = {}  # {table_id: {column_id: embedding}
        self.modeling_data = None

    def load_embeddings_of_columns_per_table(self):
        for datatype in os.listdir(self.metadata):
            if datatype == 'int' or datatype == 'float':
                print(f'loading {datatype}-column profiles')
                for profile in tqdm(os.listdir(self.metadata + '/' + datatype)):
                    with open(self.metadata + '/' + datatype + '/' + profile, 'r') as open_file:
                        profile_info = json.load(open_file)
                        path = profile_info['path']
                        embedding = profile_info['embedding']
                        column_name = profile_info['column_name']
                        table_id = generate_table_id(profile_path=path)
                        column_id = generate_column_id(profile_path=path, column_name=column_name)

                        if table_id not in self.embeddings_table_to_column:
                            self.embeddings_table_to_column[table_id] = {column_id: embedding}
                        else:
                            table_info = self.embeddings_table_to_column.get(table_id)
                            table_info[column_id] = embedding
                            self.embeddings_table_to_column[table_id] = table_info

    def generate_modeling_data(self, n_samples, export: bool = False):
        """
        Modeling data:
        pipeline x: feature a, target - selected
        pipeline x: feature b, target - selected
        pipeline x: feature c, target - not selected
        """
        def merge_embeddings(feature: list, target: list):
            return feature + target

        def get_feature_embedding(feature: str):
            table = feature.rsplit('/', 1)[0]
            if table in self.embeddings_table_to_column:
                return self.embeddings_table_to_column.get(table).get(feature)
            else:
                return None

        self.modeling_data = get_features_and_targets(self.config, n_samples=n_samples)  # query pipeline graph to get features & target

        reformatted_modeling_data = []
        for index, row in self.modeling_data.to_dict('index').items():
            reformatted_modeling_data.append([row['Pipeline_id'], row['Selected_feature'], row['Target'], 'Selected'])
            reformatted_modeling_data.append([row['Pipeline_id'], row['Discarded_feature'], row['Target'], 'Discarded'])

        self.modeling_data = pd.DataFrame(reformatted_modeling_data, columns=['Pipeline', 'Feature', 'Target', 'Selection'])\
            .drop_duplicates().sort_values(by=['Pipeline', 'Target', 'Selection'])

        self.modeling_data['Feature_embedding'] = self.modeling_data['Feature'].apply(lambda x: get_feature_embedding(x))
        self.modeling_data['Target_embedding'] = self.modeling_data['Target'].apply(lambda x: get_feature_embedding(x))

        self.modeling_data.dropna(how='any', axis=0, inplace=True)

        if export:
            self.modeling_data.to_csv('/Users/shubhamvashisth/Downloads/Feature_selector_modeling_data.csv', index=False)
            print('modeling data save at /Users/shubhamvashisth/Downloads/Feature_selector_modeling_data.csv')

        self.modeling_data['Embeddings'] = self.modeling_data.apply(lambda x: merge_embeddings(x.Feature_embedding,
                                                                                      x.Target_embedding), axis=1)

    def visualize(self):
        plt.rcParams['figure.dpi'] = 300
        sns.set_style('dark')
        fig, ax = plt.subplots(figsize=(8.5, 5))
        sns.countplot(x='Selection', data=self.modeling_data, palette="Greens_r")
        plt.grid(color='gray', linestyle='dashed', axis='y')
        plt.ylabel('Number of features')
        plt.xlabel(f'Selection')
        plt.title(f'Number of selected vs discard features for {len(set(self.modeling_data["Pipeline"]))} pipelines')
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

    def train(self, export: bool = False):
        self.modeling_data['Selection'] = self.modeling_data['Selection'].apply(lambda x: 1 if x == 'Selected' else 0)
        X = list(self.modeling_data['Embeddings'])
        y = self.modeling_data['Selection']

        print(f'X: {np.shape(X)}\ny: {np.shape(y)}')

        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.20, random_state=1)

        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)
        print(classification_report(y_true=y_test, y_pred=y_pred))

        if export:
            self.classifier.fit(X, y)
            joblib.dump(self.classifier, 'out/feature_selector.pkl', compress=9)
            print('feature selector saved at out/feature_selector.pkl')


def build():
    selector = FeatureSelector(show_connection_status=False)
    selector.load_embeddings_of_columns_per_table()
    selector.generate_modeling_data(n_samples=None, export=True)
    selector.visualize()
    selector.train(export=True)


if __name__ == '__main__':
    build()

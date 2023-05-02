import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from urllib.parse import quote_plus
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from helpers.helper import connect_to_stardog
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from operations.template import get_scaling_transformations, get_unary_transformations
warnings.filterwarnings('ignore')
RANDOM_STATE = 7
np.random.seed(RANDOM_STATE)


class TransformationRecommender:
    def __init__(self):
        self.config = connect_to_stardog(port=5820, db='data_transformation', show_status=False)
        self.numeric_column_embeddings = dict()
        self.categorical_column_embeddings = dict()
        self.modeling_data_scaling: pd.DataFrame
        self.modeling_data_unary: pd.DataFrame
        self.model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE)
        self.label_encoder = LabelEncoder()

    def __load_column_embeddings(self, path_to_embeddings: str = '../../storage/CoLR_embeddings_data_transformation'):
        for data_type in os.listdir(path=path_to_embeddings):
            if data_type == '.DS_Store':
                continue
            for profile in tqdm(os.listdir(f'{path_to_embeddings}/{data_type}'),
                                desc=f'loading {data_type} embeddings'):
                with open(f'{path_to_embeddings}/{data_type}/{profile}') as json_file:
                    try:
                        profile_info = json.load(json_file)
                        column_path = profile_info.get('path')
                        table_name = list(filter(lambda x: x.endswith('.csv'), column_path.split('/')))[0]
                        dataset_name = column_path.split('/data/' + table_name)[0].split('/')[-1]
                        if data_type == 'numerical':
                            column_embedding = profile_info.get('embedding')
                            column_name = profile_info.get('column_name')
                            column_id = f'http://kglids.org/resource/kaggle/{quote_plus(dataset_name)}/dataResource/{quote_plus(table_name)}/{quote_plus(column_name)}'
                            self.numeric_column_embeddings[column_id] = column_embedding
                        else:
                            column_embedding = profile_info.get('minhash')
                            column_name = profile_info.get('columnName')
                            column_id = f'http://kglids.org/resource/kaggle/{quote_plus(dataset_name)}/dataResource/{quote_plus(table_name)}/{quote_plus(column_name)}'
                            self.categorical_column_embeddings[column_id] = column_embedding

                    except TypeError:
                        print(f'embeddings not loaded for {column_name} ({data_type})')

    def __query_and_map_transformations(self):
        print('querying transformations ', end='')
        scaling_transformation_info = get_scaling_transformations(config=self.config)
        unary_transformation_info = get_unary_transformations(config=self.config)

        scaling_transformation_info['Embedding'] = scaling_transformation_info['Transformed_column_id'].apply(
            lambda x: self.numeric_column_embeddings.get(x))
        scaling_transformation_info.dropna(inplace=True)
        unary_transformation_info['Embeddings'] = unary_transformation_info['Transformed_column_id'].apply(
            lambda x: self.numeric_column_embeddings.get(x))
        unary_transformation_info.dropna(inplace=True)
        print('done.')
        return scaling_transformation_info, unary_transformation_info

    def __generate_modeling_data(self, scaling_df: pd.DataFrame, unary_df: pd.DataFrame):

        def average_embeddings(embeddings: list):
            number_of_embeddings = len(embeddings)
            if number_of_embeddings == 1:
                return embeddings[0]
            else:
                avg_embeddings = [0] * 300
                for embedding in embeddings:
                    for i, e in enumerate(embedding):
                        avg_embeddings[i] = avg_embeddings[i] + e

                return [avg_embeddings[i]/number_of_embeddings for i in range(len(avg_embeddings))]

        print('modeling data ', end='')
        scaling_df['Table_id'] = scaling_df['Transformed_column_id'].apply(lambda x: os.path.dirname(x))
        scaling_df.sort_values(by=['Transformation', 'Table_id'], inplace=True)
        scaling_df.to_csv('/Users/shubhamvashisth/Desktop/scaling_modeling.csv', index=False)
        scaling_df.drop('Transformed_column_id', axis=1, inplace=True)

        scaling_transformation_column = []
        table_embedding_column = []

        # average embedding by grouping on table
        embeddings_per_table = []
        previous_table_id = list(scaling_df['Table_id'])[0]
        for index, embedding_info in scaling_df.to_dict('index').items():

            current_table_id = embedding_info.get('Table_id')
            column_embedding = embedding_info.get('Embedding')
            transformation = embedding_info.get('Transformation')

            if previous_table_id == current_table_id:
                embeddings_per_table.append(column_embedding)
            else:
                averaged_embeddings = average_embeddings(embeddings=embeddings_per_table)
                scaling_transformation_column.append(transformation)
                table_embedding_column.append(averaged_embeddings)
                embeddings_per_table = [column_embedding]
                previous_table_id = current_table_id

        self.modeling_data_scaling = pd.DataFrame({'Transformation': scaling_transformation_column, 'Embeddings': table_embedding_column})
        self.modeling_data_unary = unary_df.drop('Transformed_column_id', axis=1)
        self.modeling_data_scaling.to_csv('modeling_data_scaling.csv', index=False)
        self.modeling_data_unary.to_csv('modeling_data_unary.csv', index=False)
        print('done.')

    @staticmethod
    def __plot_transformation_distribution(modeling_data: pd.DataFrame, transformation_type: str):
        plt.rcParams['figure.dpi'] = 300
        sns.set_style('dark')
        fig, ax = plt.subplots(figsize=(8.5, 5))
        sns.countplot(x='Transformation', data=modeling_data, palette="Greens_r",
                      order=modeling_data['Transformation'].value_counts().index)
        plt.grid(color='gray', linestyle='dashed', axis='y')
        plt.ylabel('columns')
        plt.xlabel(f'{transformation_type} transformations')
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

    def __train_and_export_model(self):
        def pipeline(modeling_data: pd.DataFrame, export_desc: str):
            modeling_data['Transformation'] = self.label_encoder.fit_transform(modeling_data['Transformation'])
            X = modeling_data['Embeddings']
            y = modeling_data['Transformation']
            f1_per_fold = []
            acc_per_fold = []
            for fold, (train_index, test_index) in enumerate(StratifiedKFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True).split(X, y)):
                print(f'fold: {fold+1} ', end='')
                X_train, X_test = list(X.iloc[train_index]), list(X.iloc[test_index])
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                self.model.fit(X=X_train, y=y_train)
                y_out = self.model.predict(X=X_test)
                f1_per_fold.append(f1_score(y_true=y_test, y_pred=y_out, average='weighted'))
                acc_per_fold.append(self.model.score(X=X_test, y=y_test))
                print('done.')

            self.model.fit(X=list(X), y=y)
            model_name = f'out/{export_desc}_transformation_recommender.pkl'
            encoder_name =  f'out/{export_desc}_encoder.pkl'
            joblib.dump(self.model, model_name,compress=9)
            print('model saved at', model_name)
            joblib.dump(self.label_encoder, encoder_name,compress=9)
            print('encoder saved at', encoder_name)

            return f'{sum(f1_per_fold)/len(f1_per_fold):.3f}', f'{sum(acc_per_fold)/len(acc_per_fold):.3f}'

        """
        # filter transformation types
        self.modeling_data_scaling = self.modeling_data_scaling.loc[self.modeling_data_scaling['Transformation'].isin(['StandardScaler', 'MinMaxScaler', 'RobustScaler'])]
         self.modeling_data_unary = self.modeling_data_unary.loc[self.modeling_data_unary['Transformation'].isin(['log', 'sqrt'])]
        """

        f1, accuracy = pipeline(modeling_data=self.modeling_data_scaling, export_desc='scaling')
        print(f'Scaling - F1: {f1} | Accuracy: {accuracy}')
        f1, accuracy = pipeline(modeling_data=self.modeling_data_unary, export_desc='unary')
        print(f'Unary - F1: {f1} | Accuracy: {accuracy}')

    def build(self):
        self.__load_column_embeddings()
        scaling_transformation_info, unary_transformation_info = self.__query_and_map_transformations()
        self.__generate_modeling_data(scaling_df=scaling_transformation_info, unary_df=unary_transformation_info)
        """
        self.__plot_transformation_distribution(modeling_data=self.modeling_data_scaling, transformation_type='Scaling')
        self.__plot_transformation_distribution(modeling_data=self.modeling_data_unary, transformation_type='Unary')
        """
        self.__train_and_export_model()
        print('Done.')


if __name__ == '__main__':
    transformation_recommender = TransformationRecommender()
    transformation_recommender.build()

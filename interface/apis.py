import os
import sys
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from tqdm.notebook import tqdm
from sklearn.preprocessing import *
from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer, KNNImputer
from inference_manager.inference import InferenceManager

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


class KGFarm:
    def __init__(self):
        sys.path.insert(0, str(Path(os.getcwd()).parent.absolute()))
        print('KGFarm is running in HUMAN-IN-THE-LOOP mode!')
        self.recommender = InferenceManager()

    @staticmethod
    def load_titanic_dataset():
        return pd.read_csv(os.path.dirname(os.path.abspath(__file__))+'/titanic.csv')

    @staticmethod
    def separate_numerical_and_categorical_features(df: pd.DataFrame):
        categorical_features = set(df.select_dtypes(include=['object']).columns)
        numerical_features = set(df.columns) - categorical_features
        return list(numerical_features), list(categorical_features)

    def recommend_transformations(self, X: pd.DataFrame):
        return self.recommender.recommend_transformations(X=X)

    @staticmethod
    def apply_transformations(X: pd.DataFrame, recommendation: pd.Series):
        transformation = recommendation['Recommended_transformation']
        feature = recommendation['Feature']

        if transformation in {'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'QuantileTransformer',
                              'PowerTransformer'}:
            print(f'Applying {transformation} on {list(X.columns)}')
            if transformation == 'StandardScaler':
                scaler = StandardScaler()
            elif transformation == 'MinMaxScaler':
                scaler = MinMaxScaler()
            elif transformation == 'RobustScaler':
                scaler = RobustScaler()
            elif transformation == 'QuantileTransformer':
                scaler = QuantileTransformer()
            else:
                scaler = PowerTransformer()
            X[X.columns] = scaler.fit_transform(X=X[X.columns])
            return X, scaler

        elif transformation in {'Log', 'Sqrt', 'square'}:
            print(f'Applying {transformation} on {list(feature)}')
            if transformation == 'Log':
                def log_plus_const(x, const=0):
                    return np.log(x + np.abs(const) + 0.0001)

                for f in tqdm(feature):
                    min_neg_val = X[f].min()
                    unary_transformation_model = FunctionTransformer(func=log_plus_const,
                                                                     kw_args={'const': min_neg_val}, validate=True)
                    X[f] = unary_transformation_model.fit_transform(X=np.array(X[f]).reshape(-1, 1))

            elif transformation == 'Sqrt':
                def sqrt_plus_const(x, const=0):
                    return np.sqrt(x + np.abs(const) + 0.0001)

                for f in tqdm(feature):
                    min_neg_val = X[f].min()
                    unary_transformation_model = FunctionTransformer(func=sqrt_plus_const,
                                                                     kw_args={'const': min_neg_val}, validate=True)
                    X[f] = unary_transformation_model.fit_transform(X=np.array(X[f]).reshape(-1, 1))
            else:
                unary_transformation_model = FunctionTransformer(func=np.square, validate=True)
                X[feature] = unary_transformation_model.fit_transform(X=X[feature])
            return X, transformation

        elif transformation in {'OrdinalEncoder', 'OneHotEncoder'}:
            print(f'Applying {transformation} on {list(feature)}')
            if transformation == 'OrdinalEncoder':
                encoder = OrdinalEncoder()
                X[feature] = encoder.fit_transform(X=X[feature])
            else:
                encoder = OneHotEncoder(handle_unknown='ignore')
                one_hot_encoded_features = pd.DataFrame(encoder.fit_transform(X[feature]).toarray())
                X = X.join(one_hot_encoded_features)
                X = X.drop(feature, axis=1)
            return X, encoder

        else:
            raise ValueError(f'{transformation} not supported')

    @staticmethod
    def get_columns_to_be_cleaned(df: pd.DataFrame):
        for na_type in {'none', 'n/a', 'na', 'nan', 'missing', '?', '', ' '}:
            if na_type in {'?', '', ' '}:
                df.replace(na_type, np.nan, inplace=True)
            else:
                df.replace(r'(^)' + na_type + r'($)', np.nan, inplace=True, regex=True)

        columns = pd.DataFrame(df.isnull().sum())
        columns.columns = ['Missing values']
        columns['Feature'] = columns.index
        columns = columns[columns['Missing values'] > 0]
        columns.sort_values(by='Missing values', ascending=False, inplace=True)
        columns.reset_index(drop=True, inplace=True)
        return columns

    def recommend_cleaning_operations(self, df: pd.DataFrame, visualize_missing_data: bool = True):

        def plot_heat_map():
            plt.rcParams['figure.dpi'] = 300
            plt.figure(figsize=(15, 7))
            sns.heatmap(df.isnull(), yticklabels=False, cmap='Greens_r')
            plt.show()

        def plot_bar_graph(columns: pd.DataFrame):
            if len(columns) == 0:
                return
            sns.set_color_codes('pastel')
            plt.rcParams['figure.dpi'] = 300
            plt.figure(figsize=(6, 3))

            ax = sns.barplot(x="Feature", y="Missing values", data=columns,
                             palette='Greens_r', edgecolor='gainsboro')
            ax.bar_label(ax.containers[0], fontsize=6)

            def change_width(axis, new_value):
                for patch in axis.patches:
                    current_width = patch.get_width()
                    diff = current_width - new_value
                    patch.set_width(new_value)
                    patch.set_x(patch.get_x() + diff * .5)

            change_width(ax, .20)
            plt.grid(color='lightgray', axis='y')
            plt.ylabel('Missing value', fontsize=5.5)
            plt.xlabel('')
            ax.tick_params(axis='both', which='major', labelsize=5.5)
            ax.tick_params(axis='x', labelrotation=90, labelsize=5.5)
            plt.show()

        columns_to_be_cleaned = self.get_columns_to_be_cleaned(df=df)
        if visualize_missing_data:
            plot_heat_map()
            plot_bar_graph(columns=columns_to_be_cleaned)

        if len(columns_to_be_cleaned) == 0:
            print('nothing to clean')
            return

        cleaning_recommendation = self.recommender.get_cleaning_recommendation(df[columns_to_be_cleaned['Feature']])
        return cleaning_recommendation

    def clean(self, df: pd.DataFrame, recommendation: pd.Series, handle_outliers: bool = True):

        if handle_outliers:
            numerical_features, categorical_features = self.separate_numerical_and_categorical_features(df=df)
            if len(numerical_features) > 0 and self.recommender.check_for_outliers(df=df[numerical_features]):
                outlier_rows = LocalOutlierFactor(contamination=0.05).fit_predict(df[numerical_features])
                df[numerical_features][outlier_rows == -1] = 'none'
                df = pd.concat([df[categorical_features], df[numerical_features]], axis=1)

        columns_to_be_cleaned = list(self.get_columns_to_be_cleaned(df=df)['Feature'])

        if len(columns_to_be_cleaned) == 0:
            print('nothing to clean')
            return df

        uncleaned_numerical_features, uncleaned_categorical_features = \
            self.separate_numerical_and_categorical_features(df=df[columns_to_be_cleaned])

        if recommendation['Recommended_operation'] == 'Fill':
            print('cleaning by Fill')
            for feature in tqdm(columns_to_be_cleaned):
                if feature in uncleaned_numerical_features:
                    df[feature] = df[feature].fillna(df[feature].mean())
                else:
                    df[feature] = df[feature].fillna(df[feature].mode().values[0])

        elif recommendation['Recommended_operation'] == 'Interpolate':
            print('cleaning by Interpolation')
            for feature in tqdm(columns_to_be_cleaned):
                df[feature] = df[feature].interpolate()
                df[feature] = df[feature].interpolate(method='ffill')
                df[feature] = df[feature].interpolate(method='bfill')

        elif recommendation['Recommended_operation'] == 'Impute':
            print('cleaning by Imputation')
            for feature in tqdm(columns_to_be_cleaned):
                if feature in uncleaned_numerical_features:
                    df[feature] = KNNImputer().fit_transform(np.array(df[feature]).reshape(-1, 1))
                else:
                    df[feature] = SimpleImputer(strategy='most_frequent').fit_transform(np.array(df[feature]).reshape(-1, 1))

        return df

    def recommend_features_to_be_selected(self, task: str, X: pd.DataFrame, y: pd.Series,
                                          k: int = None):
        entity_df = pd.concat([X, y], axis=1)

        if k is None or len(entity_df) < k:
            k = len(entity_df)

        return self.recommender.get_feature_selection_score(task=task, entity_df=entity_df.sample(n=k, random_state=1),
                                                            dependent_variable=str(y.name))

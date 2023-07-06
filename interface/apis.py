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
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_regression
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

    @staticmethod
    def determine_ml_task(y: pd.Series):
        y_cardinality = len(y.unique())
        if y_cardinality == 2:
            return 'binary'
        elif y_cardinality > 2 and y_cardinality/len(y) < 0.5:
            return 'multiclass'
        else:
            return 'regression'

    @staticmethod
    def get_feature_dtypes(features: pd.DataFrame):
        feature_dtype_info = pd.DataFrame(features.dtypes)
        feature_dtype_info['Feature'] = feature_dtype_info.index
        feature_dtype_info = feature_dtype_info.rename(columns={0: 'Type'}).reset_index(drop=True)
        feature_dtype_info['Type'] = feature_dtype_info['Type'].apply(
            lambda x: 'categorical' if x == 'object' else 'numerical')
        numerical_features = list(feature_dtype_info.loc[feature_dtype_info['Type'] == 'numerical']['Feature'])
        categorical_features = list(feature_dtype_info.loc[feature_dtype_info['Type'] == 'categorical']['Feature'])
        return numerical_features, categorical_features

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

    def engineer_features(self, train_set: pd.DataFrame, test_set: pd.DataFrame, target: str, information_gain_thresh: float = 0.00, correlation_thresh: float = 0.90):
        pruning_info = {}
        df_size = train_set.memory_usage(deep=True).sum() / (1024 * 1024)
        task = self.determine_ml_task(y=pd.concat([train_set[target], test_set[target]]))
        numerical_features, categorical_features = self.get_feature_dtypes(features=train_set)

        def compute_feature_importance(train: pd.DataFrame, test: pd.DataFrame, theta1: float):
            nonlocal pruning_info
            X = train[numerical_features]
            if target in list(X.columns):
                X = X.drop(target, axis=1)
            y = train[target]

            if task == 'binary' or task == 'multiclass':
                feature_pruning_model = SelectKBest(mutual_info_classif, k='all')
            else:
                feature_pruning_model = SelectKBest(f_regression, k='all')
                theta1 = theta1 + 5

            feature_pruning_model.fit(X=X, y=y)

            for feature, score in zip(X.columns, feature_pruning_model.scores_):
                if score > theta1:
                    pruning_info[feature] = score

            pruning_info = {k: v for k, v in
                                 sorted(pruning_info.items(), key=lambda item: item[1], reverse=True)}

            if (df_size >= 5 and len(numerical_features) >= 100) or (df_size >= 20):
                pruned_features = list(pruning_info.keys())[:20]
            elif len(pruning_info) < int(0.1 * len(X.columns)) or len(pruning_info) <= 1:
                pruned_features = list(X.columns)
            else:
                pruned_features = list(pruning_info.keys())

            if target not in pruned_features:
                pruned_features.append(target)

            # TODO: fix this hack for categorical features
            if len(categorical_features) > 0:
                pruned_features.extend([c for c in categorical_features if c.lower() != 'name'])
            return train[pruned_features], test[pruned_features]

        def compute_feature_correlation(train: pd.DataFrame, test: pd.DataFrame, theta2: float):
            nonlocal pruning_info
            features = train[set(numerical_features).intersection(set(pruning_info.keys()))]
            if target in list(features.columns):
                features = train.drop(target, axis=1)
            features_to_discard = set()
            for feature_i, i in zip(features.columns, range(len(features))):
                for j in range(i):
                    if features.iloc[i, j] > theta2:
                        if pruning_info.get(features.columns[j]) > pruning_info.get(feature_i):
                            features_to_discard.add(feature_i)
                        else:
                            features_to_discard.add(features.columns[j])

            features_filtered_by_correlation = [feature for feature in features.columns if
                                                feature not in features_to_discard]

            if len(features_filtered_by_correlation) < int(0.1 * len(features)) or len(
                    features_filtered_by_correlation) <= 1:
                features_filtered_by_correlation = list(train.columns)
            else:
                features_filtered_by_correlation.append(target)

            return train[features_filtered_by_correlation], test[features_filtered_by_correlation]

        def apply_transform_fe(train: pd.DataFrame, test: pd.DataFrame):
            X = train.drop(target, axis=1)
            recommended_transformations = self.recommend_transformations(X=X)  # recommendations on train-set
            if isinstance(recommended_transformations, type(None)):
                print('data does not requires transformation')
                return train, test

            unary_categorical_transformations = recommended_transformations.loc[
                recommended_transformations['Transformation_type'] == 'categorical']
            unary_numerical_transformations = recommended_transformations.loc[
                recommended_transformations['Transformation_type'] == 'unary']
            scaling_transformation = recommended_transformations.loc[
                recommended_transformations['Transformation_type'] == 'scaling']

            if not unary_categorical_transformations.empty:
                print('transforming categorical features')
                for i, transformation_info in unary_categorical_transformations.to_dict('index').items():
                    transformation = transformation_info['Recommended_transformation']
                    features_to_transform = transformation_info['Feature']
                    print(f'Applying {transformation} on {features_to_transform}')
                    if transformation == 'OrdinalEncoder':
                        encoder = OrdinalEncoder()
                        encoder.fit(
                            X=pd.concat([train[features_to_transform], test[features_to_transform]]))
                        train[features_to_transform] = encoder.transform(train[features_to_transform])
                        test[features_to_transform] = encoder.transform(test[features_to_transform])
                    else:
                        encoder = OneHotEncoder(handle_unknown='ignore')
                        train.reset_index(drop=True, inplace=True)
                        one_hot_encoded_features = pd.DataFrame(
                            encoder.fit_transform(train[features_to_transform]).toarray())
                        train = pd.concat([train, one_hot_encoded_features], axis=1)
                        train = train.drop(features_to_transform, axis=1)
                        train.columns = train.columns.astype(str)
                        test.reset_index(drop=True, inplace=True)
                        one_hot_encoded_features = pd.DataFrame(
                            encoder.fit_transform(test[features_to_transform]).toarray())
                        test = pd.concat([test, one_hot_encoded_features], axis=1)
                        test = test.drop(features_to_transform, axis=1)
                        test.columns = test.columns.astype(str)

            if not scaling_transformation.empty:
                print('scaling features')
                for i, transformation_info in scaling_transformation.to_dict('index').items():
                    transformation = transformation_info['Recommended_transformation']
                    features_to_transform = list(train.drop(target, axis=1).columns)
                    if transformation in {'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'QuantileTransformer',
                                          'PowerTransformer'}:
                        print(f'Applying {transformation} on {features_to_transform}')
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
                        train[features_to_transform] = scaler.fit_transform(X=train[features_to_transform])
                        test[features_to_transform] = scaler.transform(X=test[features_to_transform])

            if not unary_numerical_transformations.empty:
                print('transforming numerical features using unary transformations')
                for i, transformation_info in unary_numerical_transformations.to_dict('index').items():
                    transformation = transformation_info['Recommended_transformation']
                    features_to_transform = transformation_info['Feature']
                    print(f'Applying {transformation} to {features_to_transform}')
                    if transformation == 'Log':
                        def log_plus_const(x, const=0):
                            return np.log(x + np.abs(const) + 0.0001)

                        for f in tqdm(features_to_transform):
                            min_neg_val = train[f].min()
                            unary_transformation_model = FunctionTransformer(func=log_plus_const,
                                                                             kw_args={'const': min_neg_val},
                                                                             validate=True)
                            train[f] = unary_transformation_model.fit_transform(
                                X=np.array(train[f]).reshape(-1, 1))

                        for f in tqdm(features_to_transform):
                            min_neg_val = test[f].min()
                            unary_transformation_model = FunctionTransformer(func=log_plus_const,
                                                                             kw_args={'const': min_neg_val},
                                                                             validate=True)
                            test[f] = unary_transformation_model.fit_transform(
                                X=np.array(test[f]).reshape(-1, 1))

                    elif transformation == 'Sqrt':
                        def sqrt_plus_const(x, const=0):
                            return np.sqrt(x + np.abs(const) + 0.0001)

                        for f in tqdm(features_to_transform):
                            min_neg_val = train[f].min()
                            unary_transformation_model = FunctionTransformer(func=sqrt_plus_const,
                                                                             kw_args={'const': min_neg_val},
                                                                             validate=True)
                            train[f] = unary_transformation_model.fit_transform(
                                X=np.array(train[f]).reshape(-1, 1))

                        for f in tqdm(features_to_transform):
                            min_neg_val = test[f].min()
                            unary_transformation_model = FunctionTransformer(func=sqrt_plus_const,
                                                                             kw_args={'const': min_neg_val},
                                                                             validate=True)
                            test[f] = unary_transformation_model.fit_transform(
                                X=np.array(test[f]).reshape(-1, 1))
                    else:
                        unary_transformation_model = FunctionTransformer(func=np.square, validate=True)
                        train[features_to_transform] = unary_transformation_model.fit_transform(
                            X=train[features_to_transform])
                        test[features_to_transform] = unary_transformation_model.transform(
                            test[features_to_transform])
            return train, test

        train_set, test_set = compute_feature_importance(train=train_set, test=test_set, theta1=information_gain_thresh)
        train_set, test_set = compute_feature_correlation(train=train_set, test=test_set, theta2=correlation_thresh)
        train_set, test_set = apply_transform_fe(train=train_set, test=test_set)
        return train_set, test_set

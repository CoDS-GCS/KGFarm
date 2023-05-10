import os
import sys
import time
import dask
import psutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import *
from dask.dataframe import from_pandas
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, r2_score, mean_squared_error
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_regression, f_classif
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor

spath = str(Path(os.getcwd()).resolve().parents[1])
sys.path.append(spath)
sys.path.append(f'{spath}operations')
from operations.api import KGFarm

pd.set_option('display.max_colwidth', 10)

RANDOM_STATE = 7
np.random.seed(RANDOM_STATE)


class EngineerFeatures:
    def __init__(self, path: str, information_gain_thresh: float, correlation_thresh: float, feature_selection_thresh: float):
        self.theta1 = information_gain_thresh
        self.theta2 = correlation_thresh
        self.theta3 = feature_selection_thresh
        self.path_to_dataset = path
        self.experiment_datasets_info = pd.read_csv('automl_datasets.csv')
        self.working_dir = os.getcwd()
        os.chdir('../../')
        self.kgfarm = KGFarm(show_connection_status=False)
        self.pruning_info = dict()
        self.models = None

    @staticmethod
    def __get_feature_dtypes(features: pd.DataFrame):
        feature_dtype_info = pd.DataFrame(features.dtypes)
        feature_dtype_info['Feature'] = feature_dtype_info.index
        feature_dtype_info = feature_dtype_info.rename(columns={0: 'Type'}).reset_index(drop=True)
        feature_dtype_info['Type'] = feature_dtype_info['Type'].apply(
            lambda x: 'categorical' if x == 'object' else 'numerical')
        numerical_features = list(feature_dtype_info.loc[feature_dtype_info['Type'] == 'numerical']['Feature'])
        categorical_features = list(feature_dtype_info.loc[feature_dtype_info['Type'] == 'categorical']['Feature'])
        return numerical_features, categorical_features

    @staticmethod
    def __separate_independent_and_dependent_variables(df: pd.DataFrame, target: str):
        independent_variables = [feature for feature in df.columns if feature != target]
        return df[independent_variables], df[target]

    def __compute_feature_importance(self, train_set: pd.DataFrame, test_set: pd.DataFrame, target: str, df_size: float,
                                     task: str, categorical_features: list, numerical_features: list):

        if len(categorical_features) > 0 and len(numerical_features) == 0:  # i.e. all categorical features
            return train_set, test_set

        else:
            X, y = self.__separate_independent_and_dependent_variables(df=train_set, target=target)
            X = X[numerical_features]  # consider numerical features for pruning
            X_parallelized = dask.dataframe.from_pandas(X, npartitions=os.cpu_count()-1).values

            if task == 'regression':
                feature_pruning_model = SelectKBest(f_regression, k='all')
                self.theta1 = self.theta1 + 5

            else:
                feature_pruning_model = SelectKBest(mutual_info_classif, k='all')

            feature_pruning_model.fit(X=X_parallelized, y=y)
            for feature, score in zip(X.columns, dask.compute(feature_pruning_model.scores_)[0]):
                if score > self.theta1:
                    self.pruning_info[feature] = score

            self.pruning_info = {k: v for k, v in
                                 sorted(self.pruning_info.items(), key=lambda item: item[1], reverse=True)}

            if (df_size >= 5 and len(numerical_features) >= 100) or (df_size >= 20):
                pruned_features = list(self.pruning_info.keys())[:20]
            elif len(self.pruning_info) < int(0.1*len(X.columns)) or len(self.pruning_info) <= 1:
                pruned_features = list(X.columns)
            else:
                pruned_features = list(self.pruning_info.keys())

            if target not in pruned_features:
                pruned_features.append(target)

            if len(categorical_features) > 0:
                pruned_features.extend(categorical_features)

            return train_set[pruned_features], test_set[pruned_features]

    def __compute_feature_correlation(self, train_set: pd.DataFrame, test_set: pd.DataFrame, target: str):
        """
        1. compute pairwise correlation
        2. drop features with the lowest feature importance
        """
        features = from_pandas(data=train_set.drop(target, axis=1), npartitions=os.cpu_count()-1).corr().compute()
        features_to_discard = set()
        for feature_i, i in zip(features.columns, range(len(features))):
            for j in range(i):
                if features.iloc[i, j] > self.theta2:
                    if self.pruning_info.get(features.columns[j]) > self.pruning_info.get(feature_i):
                        features_to_discard.add(feature_i)
                    else:
                        features_to_discard.add(features.columns[j])

        features_filtered_by_correlation = [feature for feature in features.columns if
                                            feature not in features_to_discard]

        if len(features_filtered_by_correlation) < int(0.1*len(features)) or len(features_filtered_by_correlation) <= 1:
            features_filtered_by_correlation = list(train_set.columns)
        else:
            features_filtered_by_correlation.append(target)
        return train_set[features_filtered_by_correlation], test_set[features_filtered_by_correlation]

    def __transform(self, train_set: pd.DataFrame, test_set: pd.DataFrame, target: str):
        X, _ = self.__separate_independent_and_dependent_variables(df=train_set, target=target)
        recommended_transformations = self.kgfarm.recommend_transformations(X=X)  # recommendations on train-set
        if isinstance(recommended_transformations, type(None)):
            print('data does not requires transformation')
            return train_set, test_set
        else:
            unary_categorical_transformations = recommended_transformations.loc[
                recommended_transformations['Transformation_type'].isin({'ordinal encoding', 'nominal encoding'})]
            unary_numerical_transformations = recommended_transformations.loc[
                recommended_transformations['Transformation_type'] == 'unary transformation']
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
                        encoder.fit(X=pd.concat([train_set[features_to_transform], test_set[features_to_transform]]))
                        train_set[features_to_transform] = encoder.transform(train_set[features_to_transform])
                        test_set[features_to_transform] = encoder.transform(test_set[features_to_transform])
                    else:
                        encoder = OneHotEncoder(handle_unknown='ignore')
                        encoder.fit(X=pd.concat([train_set[features_to_transform], test_set[features_to_transform]]))
                        one_hot_encoded_features_train_set = pd.DataFrame(encoder.transform(train_set[features_to_transform]).toarray())
                        train_set = train_set.join(one_hot_encoded_features_train_set)
                        train_set = train_set.drop(features_to_transform, axis=1)
                        one_hot_encoded_features_test_set = pd.DataFrame(encoder.transform(test_set[features_to_transform]).toarray())
                        test_set = test_set.join(one_hot_encoded_features_test_set)
                        test_set = test_set.drop(one_hot_encoded_features_test_set, axis=1)

                        train_set, encoder = self.kgfarm.apply_transformations(X=train_set[features_to_transform], recommendation=unary_categorical_transformations.iloc[i])
                        test_set[features_to_transform] = encoder.transform(test_set[features_to_transform])
                        one_hot_encoded_features = pd.DataFrame(
                            encoder.transform(test_set[features_to_transform]).toarray())
                        test_set = test_set.join(one_hot_encoded_features)
                        test_set = test_set.drop(features_to_transform, axis=1)

            if not scaling_transformation.empty:
                print('scaling features')
                for i, transformation_info in scaling_transformation.to_dict('index').items():
                    transformation = transformation_info['Recommended_transformation']
                    features_to_transform = list(train_set.drop(target, axis=1).columns)
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
                        train_set[features_to_transform] = scaler.fit_transform(X=train_set[features_to_transform])
                        test_set[features_to_transform] = scaler.transform(X=test_set[features_to_transform])

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
                            min_neg_val = train_set[f].min()
                            unary_transformation_model = FunctionTransformer(func=log_plus_const,
                                                                             kw_args={'const': min_neg_val},
                                                                             validate=True)
                            train_set[f] = unary_transformation_model.fit_transform(
                                X=np.array(train_set[f]).reshape(-1, 1))

                        for f in tqdm(features_to_transform):
                            min_neg_val = test_set[f].min()
                            unary_transformation_model = FunctionTransformer(func=log_plus_const,
                                                                             kw_args={'const': min_neg_val},
                                                                             validate=True)
                            test_set[f] = unary_transformation_model.fit_transform(
                                X=np.array(test_set[f]).reshape(-1, 1))

                    elif transformation == 'Sqrt':
                        def sqrt_plus_const(x, const=0):
                            return np.sqrt(x + np.abs(const) + 0.0001)

                        for f in tqdm(features_to_transform):
                            min_neg_val = train_set[f].min()
                            unary_transformation_model = FunctionTransformer(func=sqrt_plus_const,
                                                                             kw_args={'const': min_neg_val},
                                                                             validate=True)
                            train_set[f] = unary_transformation_model.fit_transform(
                                X=np.array(train_set[f]).reshape(-1, 1))

                        for f in tqdm(features_to_transform):
                            min_neg_val = test_set[f].min()
                            unary_transformation_model = FunctionTransformer(func=sqrt_plus_const,
                                                                             kw_args={'const': min_neg_val},
                                                                             validate=True)
                            test_set[f] = unary_transformation_model.fit_transform(
                                X=np.array(test_set[f]).reshape(-1, 1))
                    else:
                        unary_transformation_model = FunctionTransformer(func=np.square, validate=True)
                        train_set[features_to_transform] = unary_transformation_model.fit_transform(
                            X=train_set[features_to_transform])
                        test_set[features_to_transform] = unary_transformation_model.transform(
                            test_set[features_to_transform])
            return train_set, test_set

    def __select_features(self, train_set: pd.DataFrame, test_set: pd.DataFrame, target: str):
        X, y = self.__separate_independent_and_dependent_variables(df=train_set, target=target)
        filter_model = SelectKBest(f_classif, k='all')
        filter_model.fit(X=X, y=y)

        features_filtered = set()
        for feature, score in zip(X.columns, filter_model.scores_):
            if score > self.theta3:
                features_filtered.add(feature)

        if len(features_filtered) < int(0.1*len(X.columns)) or len(features_filtered) <= 1:
            features_filtered = set(X.columns)

        features_filtered.add(target)
        """
        try:
            feature_selection_recommendation = self.kgfarm.recommend_features_to_be_selected(
                entity_df=train_set[features_filtered], dependent_variable=target, task=task, n=300)
            feature_selection_recommendation = feature_selection_recommendation.loc[
                feature_selection_recommendation['Selection_score'] > 0.60]
        except AttributeError:
            return train_set[features_filtered], test_set[features_filtered]
        if len(feature_selection_recommendation) == 0:
            features_filtered_by_kgfarm = list(train_set.columns)
        else:
            features_filtered_by_kgfarm = feature_selection_recommendation['Feature'].tolist()
            features_filtered_by_kgfarm.append(target)
        return train_set[features_filtered_by_kgfarm], test_set[features_filtered_by_kgfarm]
        """
        return train_set[features_filtered], test_set[features_filtered]

    def __train_and_evaluate(self, train_set: pd.DataFrame, test_set: pd.DataFrame, target: str, task: str):
        X_train, y_train = self.__separate_independent_and_dependent_variables(df=train_set, target=target)
        X_test, y_test = self.__separate_independent_and_dependent_variables(df=test_set, target=target)
        scores_f1_r2 = []
        scores_acc_mse = []

        if task == 'regression':
            for model in self.models:
                model.fit(X=X_train, y=y_train)
                y_out = model.predict(X=X_test)
                scores_f1_r2.append(r2_score(y_true=y_test, y_pred=y_out))
                scores_acc_mse.append(np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_out)))

        else:
            if task == 'multi-class':
                f1_type = 'weighted'
            else:
                f1_type = 'binary'
            for model in self.models:
                model.fit(X=X_train, y=y_train)
                y_out = model.predict(X=X_test)
                scores_f1_r2.append(f1_score(y_true=y_test, y_pred=y_out, average=f1_type))
                scores_acc_mse.append((model.score(X=X_test, y=y_test)))

        return scores_f1_r2, scores_acc_mse

    def run(self):
        experiment_results = []
        os.chdir(self.working_dir)
        for dataset_count, dataset_info in tqdm(self.experiment_datasets_info.to_dict('index').items(),
                                                desc='Datasets processed'):
            self.pruning_info = dict()
            print(f'{dataset_info["Dataset"]} ({dataset_info["Task"]})')
            df = pd.read_csv(f'{self.path_to_dataset}{dataset_info["Dataset"]}/{dataset_info["Dataset"]}.csv')
            df_size = df.memory_usage(deep=True).sum() / (1024 * 1024)
            target = dataset_info['Target']
            numerical_features, categorical_features = self.__get_feature_dtypes(
                df.drop(dataset_info['Target'], axis=1))
            if dataset_info["Task"] != 'regression':
                result_f1_r2 = {'KNN': 0, 'RF': 0, 'NN': 0}
                result_acc_mse = {'KNN': 0, 'RF': 0, 'NN': 0}
                df[target] = LabelEncoder().fit_transform(df[target])
                self.models = [KNeighborsClassifier(),
                               RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE),
                               MLPClassifier(random_state=RANDOM_STATE)]
                folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

            else:
                result_f1_r2 = {'GB': 0, 'RF': 0, 'EN': 0}
                result_acc_mse = {'GB': 0, 'RF': 0, 'EN': 0}
                self.models = [GradientBoostingRegressor(random_state=RANDOM_STATE),
                               RandomForestRegressor(random_state=RANDOM_STATE),
                               ElasticNet(alpha=0.75, random_state=RANDOM_STATE)]
                folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

            f1_r2_per_dataset = list()
            acc_mse_per_dataset = list()
            n_rows = len(df)
            n_features = len(df.columns) - 1  # excluding target

            memory_before = psutil.Process().memory_info().rss
            start = time.time()
            fold = 1
            for train_index, test_index in folds.split(
                    df, df[target]):
                print(f'{dataset_info["Dataset"]} fold-{fold}')
                train_set = df.iloc[train_index]
                test_set = df.iloc[test_index]

                print(f'computing feature importance for {n_features} features', end=' ')

                train_set, test_set = self.__compute_feature_importance(train_set=train_set, test_set=test_set,
                                                                        target=target, df_size=df_size,
                                                                        task=dataset_info['Task'],
                                                                        categorical_features=categorical_features,
                                                                        numerical_features=numerical_features)
                print(f'done. | # features: {len(train_set.columns) - 1}')

                print(f'computing feature correlation b/w {len(train_set.columns) - 1} features', end=' ')
                train_set, test_set = self.__compute_feature_correlation(train_set=train_set, test_set=test_set,
                                                                         target=target)
                print(f'done. | # features: {len(train_set.columns) - 1}')

                print(f'recommending data transformations on {len(train_set.columns) - 1} features', end=' ')
                train_set, test_set = self.__transform(train_set=train_set, test_set=test_set, target=target)
                print(f'done.')

                print(f'computing feature importance for {len(train_set.columns) - 1} features', end=' ')
                train_set, test_set = self.__select_features(train_set=train_set, test_set=test_set, target=target)
                print(f'done. | # features: {len(train_set.columns) - 1}')

                print(f'training and evaluating models on {len(train_set.columns) - 1} features')
                f1_r2, acc_mse = self.__train_and_evaluate(train_set=train_set, test_set=test_set, target=target,
                                                           task=dataset_info['Task'])
                f1_r2_per_dataset.append(f1_r2)
                acc_mse_per_dataset.append(acc_mse)
                print('done.')

                print(f'{dataset_info["Dataset"]} fold-{fold} completed.\n', '-' * 100)
                fold = fold + 1

            time_taken = f'{(time.time() - start):.2f}'
            memory_usage = f'{abs(psutil.Process().memory_info().rss - memory_before) / (1024 * 1024):.2f}'

            # parse result as a dataframe
            for fold_result in f1_r2_per_dataset:
                for acc, model in zip(fold_result, result_f1_r2.keys()):
                    result_f1_r2[model] += acc
            for fold_result in acc_mse_per_dataset:
                for acc, model in zip(fold_result, result_acc_mse.keys()):
                    result_acc_mse[model] += acc

            result_f1_r2 = {k: f'{(v * 100 / 5):.2f}' for k, v in result_f1_r2.items()}
            result_acc_mse = {k: f'{(v * 100 / 5):.2f}' for k, v in result_acc_mse.items()}.values()
            result_df = pd.DataFrame(list(result_f1_r2.items()), columns=['ML Model', 'F1/R2: KGFarm'])
            result_df['Dataset'] = [dataset_info['Dataset']] * (len(self.models))
            result_df['Task'] = [dataset_info['Task']] * len(self.models)
            result_df['ACC/RMSE: KGFarm'] = result_acc_mse
            result_df['# Features'] = [n_features] * (len(self.models))
            result_df['# Rows'] = [n_rows] * (len(self.models))
            result_df['# Numerical Features'] = [len(numerical_features)] * (len(self.models))
            result_df['# Categorical Features'] = [len(categorical_features)] * (len(self.models))
            result_df['Time: KGFarm (in seconds)'] = [time_taken] * (len(self.models))
            result_df['Memory: KGFarm (in MB)'] = [memory_usage] * (len(self.models))
            experiment_results.append(result_df[['Dataset', 'Task', '# Features', '# Rows', '# Numerical Features',
                                                 '# Categorical Features', 'ML Model', 'F1/R2: KGFarm',
                                                 'ACC/RMSE: KGFarm', 'Time: KGFarm (in seconds)',
                                                 'Memory: KGFarm (in MB)']])
            pd.concat(experiment_results).to_csv('kgfarm_on_automl_datasets.csv', index=False)
            print(
                pd.concat(experiment_results)[['Dataset', 'ML Model', 'F1/R2: KGFarm', 'ACC/RMSE: KGFarm']].reset_index(
                    drop=True))
            print(f'{len(self.experiment_datasets_info) - dataset_count - 1} datasets left.')
        print('Done.')


if __name__ == '__main__':
    experiment = EngineerFeatures(path='../../../../../data/automl_datasets/', information_gain_thresh=0.00, correlation_thresh=0.90, feature_selection_thresh=1.00)
    experiment.run()

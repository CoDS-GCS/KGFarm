import os
import sys
import time
import psutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import f1_score, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.feature_selection import mutual_info_classif, f_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
spath = str(Path(os.getcwd()).resolve().parents[1])
sys.path.append(spath)
sys.path.append(f'{spath}operations')
from operations.api import KGFarm

RANDOM_STATE = 7
np.random.seed(RANDOM_STATE)


class EngineerFeatures:
    def __init__(self, path: str, theta1: float, theta2: float):
        self.theta1 = theta1
        self.theta2 = theta2
        self.path_to_dataset = path
        self.experiment_datasets_info = pd.read_csv('automl_datasets.csv')
        self.working_dir = os.getcwd()
        os.chdir('../../')
        self.kgfarm = KGFarm(show_connection_status=False)
        self.information_gain = dict()
        self.models = None

    @staticmethod
    def __separate_independent_and_dependent_variables(df: pd.DataFrame, target: str):
        independent_variables = [feature for feature in df.columns if feature != target]
        return df[independent_variables], df[target]

    def __compute_information_gain(self, train_set: pd.DataFrame, test_set: pd.DataFrame, target: str):
        information_gain_model = SelectKBest(mutual_info_classif, k='all')
        X, y = self.__separate_independent_and_dependent_variables(df=train_set, target=target)
        information_gain_model.fit(X=X, y=y)

        for feature, score in zip(X.columns, information_gain_model.scores_):
            if score > self.theta1:
                self.information_gain[feature] = score

        self.information_gain = {k: v for k, v in
                                 sorted(self.information_gain.items(), key=lambda item: item[1], reverse=True)}
        features_filtered_by_information_gain = list(self.information_gain.keys())
        features_filtered_by_information_gain.append(target)
        return train_set[features_filtered_by_information_gain], test_set[features_filtered_by_information_gain]

    def __compute_feature_correlation(self, train_set: pd.DataFrame, test_set: pd.DataFrame, target: str):
        features = train_set.drop(target, axis=1).corr()

        features_to_discard = set()
        for feature_i, i in zip(features.columns, range(len(features))):
            for j in range(i):
                if features.iloc[i, j] > self.theta2:
                    if self.information_gain.get(features.columns[j]) > self.information_gain.get(feature_i):
                        features_to_discard.add(feature_i)
                    else:
                        features_to_discard.add(features.columns[j])

        features_filtered_by_correlation = [feature for feature in features.columns if
                                            feature not in features_to_discard]
        features_filtered_by_correlation.append(target)
        return train_set[features_filtered_by_correlation], test_set[features_filtered_by_correlation]

    def __transform(self, train_set: pd.DataFrame, test_set: pd.DataFrame, target: str):
        X, _ = self.__separate_independent_and_dependent_variables(df=train_set, target=target)
        recommended_transformations = self.kgfarm.recommend_data_transformations(entity_df=X, show_query=False, show_insights=False)

        if isinstance(recommended_transformations, type(None)):
            print('data does not requires transformation')
            return train_set, test_set

        else:
            for n, recommendation in recommended_transformations.to_dict('index').items():
                train_set, _ = self.kgfarm.apply_transformation(transformation_info=recommended_transformations.iloc[n],
                                                                entity_df=train_set, output_message='min')
                test_set, _ = self.kgfarm.apply_transformation(transformation_info=recommended_transformations.iloc[n],
                                                               entity_df=test_set, output_message='min')

            return train_set, test_set

    def __select_features(self, train_set: pd.DataFrame, test_set: pd.DataFrame, target: str, task: str):
        X, y = self.__separate_independent_and_dependent_variables(df=train_set, target=target)
        filter_model = SelectKBest(f_classif, k='all')
        filter_model.fit(X=X, y=y)

        features_filtered_by_anova = set()
        for feature, score in zip(X.columns, filter_model.scores_):
            if score > 1.00:
                features_filtered_by_anova.add(feature)

        features_filtered_by_anova.add(target)
        try:
            feature_selection_recommendation = self.kgfarm.recommend_features_to_be_selected(
                entity_df=train_set[features_filtered_by_anova], dependent_variable=target, task=task, n=300)
            feature_selection_recommendation = feature_selection_recommendation.loc[
                feature_selection_recommendation['Selection_score'] > 0.60]
        except AttributeError:
            return train_set[features_filtered_by_anova], test_set[features_filtered_by_anova]

        features_filtered_by_kgfarm = feature_selection_recommendation['Feature'].tolist()
        features_filtered_by_kgfarm.append(target)
        return train_set[features_filtered_by_kgfarm], test_set[features_filtered_by_kgfarm]

    def __train_and_evaluate(self, train_set: pd.DataFrame, test_set: pd.DataFrame, target: str, task: str):
        X_train, y_train = self.__separate_independent_and_dependent_variables(df=train_set, target=target)
        X_test, y_test = self.__separate_independent_and_dependent_variables(df=test_set, target=target)
        scores = []

        if task == 'regression':
            for model in self.models:
                model.fit(X=X_train, y=y_train)
                y_out = model.predict(X=X_test)
                scores.append(r2_score(y_true=y_test, y_pred=y_out))

        else:
            for model in self.models:
                model.fit(X=X_train, y=y_train)
                y_out = model.predict(X=X_test)
                scores.append(f1_score(y_true=y_test, y_pred=y_out))

        return scores

    def run(self):
        experiment_results = []
        os.chdir(self.working_dir)
        for _, dataset_info in tqdm(self.experiment_datasets_info.to_dict('index').items(), desc='Datasets processed'):
            result = {'KNN': 0, 'RF': 0, 'NN': 0}
            self.information_gain = dict()
            print(f'{dataset_info["Dataset"]} ({dataset_info["Task"]})')
            df = pd.read_csv(f'{self.path_to_dataset}{dataset_info["Dataset"]}/{dataset_info["Dataset"]}.csv')
            target = dataset_info['Target']

            if dataset_info["Task"] != 'regression':
                df[target] = LabelEncoder().fit_transform(df[target])
                self.models = [KNeighborsClassifier(n_neighbors=7, weights='distance'),
                       RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE),
                       MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=200, learning_rate='adaptive', solver='adam', activation='relu', random_state=RANDOM_STATE)]
                folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

            else:
                self.models = [GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
                               RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
                               MLPRegressor(hidden_layer_sizes=(100, 50, 25), max_iter=200, learning_rate='adaptive',
                                             solver='adam', activation='relu', random_state=RANDOM_STATE)]
                folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

            scores_per_dataset = list()
            n_rows = len(df)
            n_features = len(df.columns) - 1  # excluding target

            memory_before = psutil.Process().memory_info().rss
            start = time.time()
            fold = 1
            # TODO: use k-fold for regression, stratified otherwise
            for train_index, test_index in folds.split(
                    df, df[target]):
                print(f'{dataset_info["Dataset"]} fold-{fold}')
                train_set = df.iloc[train_index]
                test_set = df.iloc[test_index]

                print(f'computing information gain for {n_features}')
                train_set, test_set = self.__compute_information_gain(train_set=train_set, test_set=test_set,
                                                                      target=target)
                print(f'information gain done, # features: {len(train_set.columns)-1}')

                print(f'computing feature correlation b/w {len(train_set.columns)-1} features')
                train_set, test_set = self.__compute_feature_correlation(train_set=train_set, test_set=test_set,
                                                                         target=target)
                print(f'correlation done, # features: {len(train_set.columns)-1}')

                print(f'recommending and applying transformation on {len(train_set.columns)-1} features')
                train_set, test_set = self.__transform(train_set=train_set, test_set=test_set, target=target)
                print(f'transformations done, # features: {len(train_set.columns)-1}')

                print(f'computing feature importance of {len(train_set.columns)-1} features')
                train_set, test_set = self.__select_features(train_set=train_set, test_set=test_set, target=target,
                                                             task=dataset_info['Task'])
                print(f'feature selection done, # features: {len(train_set.columns)-1}')

                print(f'training and evaluating models on {len(train_set.columns)-1} features')
                scores_per_dataset.append(
                    self.__train_and_evaluate(train_set=train_set, test_set=test_set, target=target, task=dataset_info['Task']))
                print('model training and evaluation done.')

                print(f'{dataset_info["Dataset"]} fold-{fold} completed.\n', '-'*100)
                fold = fold + 1

            time_taken = f'{(time.time()-start):.2f}'
            memory_usage = f'{abs(psutil.Process().memory_info().rss - memory_before)/(1024*1024):.2f}'

            # parse result as a dataframe
            for fold_result in scores_per_dataset:
                for acc, model in zip(fold_result, result.keys()):
                    result[model] += acc

            result = {k: f'{(v * 100 / 5):.2f}' for k, v in result.items()}
            result = pd.DataFrame(list(result.items()), columns=['Classifier', 'F1/R2: KGFarm'])
            result['Dataset'] = [dataset_info['Dataset']] * (len(self.models))
            result['Task'] = [dataset_info['Task']] * len(self.models)
            result['# Features'] = [n_features] * (len(self.models))
            result['# Rows'] = [n_rows] * (len(self.models))
            result['Time: KGFarm (in seconds)'] = [time_taken] * (len(self.models))
            result['Memory: KGFarm (in MB)'] = [memory_usage] * (len(self.models))
            experiment_results.append(result[['Dataset', 'Task', '# Features', '# Rows', 'Classifier', 'F1/R2: KGFarm',
                                              'Time: KGFarm (in seconds)', 'Memory: KGFarm (in MB)']])
            pd.concat(experiment_results).to_csv('kgfarm_on_automl_datasets.csv', index=False)
            print(pd.concat(experiment_results)[['Dataset', 'Task', 'Classifier', 'F1/R2: KGFarm', 'Time: KGFarm (in seconds)']])

        pd.concat(experiment_results).to_csv('kgfarm_vs_autolearn.csv', index=False)
        print('Done.')

# TODO: add accuracy
# TODO: change NN in regression to Elasticnet

if __name__ == '__main__':
    experiment = EngineerFeatures(path='../../../../../data/automl_datasets/', theta1=0.05, theta2=0.90)
    experiment.run()

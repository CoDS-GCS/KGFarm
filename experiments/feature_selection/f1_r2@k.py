import os
import sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.model_selection import cross_val_score, KFold
from operations.api import KGFarm

path_to_datasets = '/Users/shubhamvashisth/Desktop/benchmark_datasets'
K = [5, 10, 20]  # as described in the KGFarm feature selection experiment

datasets = []
tasks = []
base_f1 = []
data_points = []
n_features = []
base_times = []
filter_5 = []
kgfarm_5 = []
embedded_5 = []
wrapper_5 = []
filter_10 = []
kgfarm_10 = []
embedded_10 = []
wrapper_10 = []
filter_20 = []
kgfarm_20 = []
embedded_20 = []
wrapper_20 = []
kgfarm_times = []
filter_times = []
wrapper_times = []
embedded_times = []


def apply_filter_method():
    start_time = time.time()
    selector = SelectKBest(f_classif, k='all')
    selector.fit(X, y)
    f_test = pd.DataFrame({'Feature': independent_variables, 'F_test_score': selector.scores_}).\
        sort_values(by='F_test_score', ascending=False).reset_index(drop=True)
    filter_time = time.time()-start_time
    print(f'Filter (anova): {filter_time:.3f} seconds')
    filter_times.append(f'{filter_time:.3f}')
    return f_test


def apply_embedded_method():
    start_time = time.time()
    if task == 'regression':
        tree = DecisionTreeRegressor()
    else:
        tree = DecisionTreeClassifier()
    tree.fit(X, y)
    feature_importance = pd.DataFrame({'Feature': independent_variables, 'F_test_score': tree.feature_importances_}). \
        sort_values(by='F_test_score', ascending=False).reset_index(drop=True)
    embedded_time = time.time()-start_time
    print(f'Embedded (tree): {embedded_time:.3f} seconds')
    embedded_times.append(f'{embedded_time:.3f}')
    return feature_importance


def apply_kgfarm_method():
    os.chdir('../../')
    kgfarm = KGFarm(show_connection_status=False)
    start_time = time.time()
    kgfarm_fs_score = kgfarm.recommend_features_to_be_selected(task=task, entity_df=df, dependent_variable=target)
    kgfarm_time = time.time()-start_time
    print(f'KGFarm: {kgfarm_time:.3f} seconds')
    kgfarm_times.append(f'{kgfarm_time:.3f}')
    return kgfarm_fs_score


def apply_wrapper_method():
    if task == 'regression':
        tree = DecisionTreeRegressor()
    else:
        tree = DecisionTreeClassifier()
    start_time = time.time()
    selector = RFE(estimator=tree, n_features_to_select=None, step=1)
    selector.fit(X, y)
    rfe_scores = pd.DataFrame({'Feature': independent_variables, 'Ranking': selector.ranking_}). \
        sort_values(by='Ranking').reset_index(drop=True)
    wrapper_time = time.time()-start_time
    print(f'Wrapper (RFE): {wrapper_time:.3f} seconds')
    wrapper_times.append(f'{wrapper_time:.3f}')
    return rfe_scores


def calculate_score(feature_selection_info: pd.DataFrame, algorithm, scoring, cv, technique):
    for k in K:
        model_training_start_time = time.time()
        feature = list(feature_selection_info.head(k)['Feature'])
        f1 = np.mean(cross_val_score(algorithm, df[feature], y, cv=cv, scoring=scoring))
        print(f'K = {k} | {scoring.replace("_macro", "").upper()} = {f1:.3f} | Features: {feature} | Time for model training: {time.time()-model_training_start_time}')
        if technique == 'filter':
            if k == 5:
                filter_5.append(f'{f1:.3f}')
            elif k == 10:
                filter_10.append(f'{f1:.3f}')
            elif k == 20:
                filter_20.append(f'{f1:.3f}')
        elif technique == 'kgfarm':
            if k == 5:
                kgfarm_5.append(f'{f1:.3f}')
            elif k == 10:
                kgfarm_10.append(f'{f1:.3f}')
            elif k == 20:
                kgfarm_20.append(f'{f1:.3f}')
        elif technique == 'embedded':
            if k == 5:
                embedded_5.append(f'{f1:.3f}')
            elif k == 10:
                embedded_10.append(f'{f1:.3f}')
            elif k == 20:
                embedded_20.append(f'{f1:.3f}')
        else:
            if k == 5:
                wrapper_5.append(f'{f1:.3f}')
            elif k == 10:
                wrapper_10.append(f'{f1:.3f}')
            elif k == 20:
                wrapper_20.append(f'{f1:.3f}')


if __name__ == '__main__':
    for n, dataset_info in tqdm(pd.read_csv('datasets.csv').to_dict('index').items()):
        dataset_name = dataset_info['Feature selection dataset']
        task = dataset_info['Task']
        print(f'{n+1}. {dataset_name} ({task})')
        df = pd.read_csv(f'{path_to_datasets}/{dataset_name}/{dataset_name}.csv')
        target = dataset_info['Target']
        independent_variables = [feature for feature in df.columns if feature != target]
        X = df[independent_variables]
        y = df[target]

        if task == 'regression':
            model = RandomForestRegressor()
            metric = 'r2'
        elif task == 'multi-class' or task == 'binary':
            model = RandomForestClassifier(n_estimators=100)
            metric = 'f1_macro'
            y = LabelEncoder().fit_transform(y)
        else:
            ValueError(f'task must be either regression or classification')
            sys.exit()

        kf = KFold(n_splits=10)

        start = time.time()
        baseline = np.mean(cross_val_score(model, X, y, cv=kf, scoring=metric))
        base_time = time.time()-start
        print(f'Baseline ({len(independent_variables)} features, {len(y)} datapoints): {baseline:.3f}\ntime taken: {base_time:.3f} seconds')

        filter_scores = apply_filter_method()
        calculate_score(filter_scores, algorithm=model, scoring=metric, cv=kf, technique='filter')
        kgfarm_scores = apply_kgfarm_method()
        calculate_score(kgfarm_scores, algorithm=model, scoring=metric, cv=kf, technique='kgfarm')
        embedded_scores = apply_embedded_method()
        calculate_score(embedded_scores, algorithm=model, scoring=metric, cv=kf, technique='embedded')
        wrapper_scores = apply_wrapper_method()
        calculate_score(wrapper_scores, algorithm=model, scoring=metric, cv=kf, technique='wrapper')

        datasets.append(dataset_name)
        tasks.append(task)
        data_points.append(len(y))
        n_features.append(len(independent_variables))
        base_f1.append(f'{baseline:.3f}')
        base_times.append(f'{base_time:.3f}')

    results = pd.DataFrame({'Dataset': datasets, '# Features': n_features, '# Datapoints': data_points, 'Baseline': base_f1,
                       'KGFarm (K=5)': kgfarm_5, 'KGFarm (K=10)': kgfarm_10, 'KGFarm (K=20)': kgfarm_20,
                       'Filter (K=5)': filter_5, 'Filter (K=10)': filter_10, 'Filter (K=20)': filter_20,
                       'Embedded (K=5)': embedded_5, 'Embedded (K=10)': embedded_10, 'Embedded (K=20)': embedded_20,
                       'Wrapper (K=5)': wrapper_5, 'Wrapper (K=10)': wrapper_10, 'Wrapper (K=20)': wrapper_20,
                       'Baseline time': base_times, 'KGFarm time': kgfarm_times, 'Filter time': filter_times,
                       'Embedded time': embedded_times, 'Wrapper time': wrapper_times})

    results.to_csv('feature_selection_results.csv', index=True)
    print('Done.')

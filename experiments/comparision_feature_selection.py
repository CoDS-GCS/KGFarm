import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.model_selection import cross_val_score, KFold
from operations.api import KGFarm

task = 'classification'  # 'classification' or 'regression'
datasets = {'Epileptic Seizure Recognition.csv': 'y'}  # dataset: target variable
K = [5, 10, 20]  # as described in the KGFarm feature selection experiment


def apply_filter_method():
    start_time = time.time()
    selector = SelectKBest(f_classif, k='all')
    selector.fit(X, y)
    f_test = pd.DataFrame({'Feature': independent_variables, 'F_test_score': selector.scores_}).\
        sort_values(by='F_test_score', ascending=False).reset_index(drop=True)
    print(f'Filter (anova): {time.time()-start_time:.3f} seconds')
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
    print(f'Embedded (tree): {time.time() - start_time:.3f} seconds')
    return feature_importance


def apply_kgfarm_method():
    os.chdir('../')
    kgfarm = KGFarm(show_connection_status=False)
    start_time = time.time()
    kgfarm_fs_score = kgfarm.recommend_features_to_be_selected(entity_df=df, dependent_variable=dependent_variable)
    print(f'KGFarm: {time.time() - start_time:.3f} seconds')
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
    print(f'Wrapper (RFE): {time.time() - start_time:.3f} seconds')
    return rfe_scores


def calculate_score(feature_selection_info: pd.DataFrame, algorithm, scoring, cv):
    for k in K:
        model_training_start_time = time.time()
        feature = list(feature_selection_info.head(k)['Feature'])
        f1 = np.mean(cross_val_score(algorithm, df[feature], y, cv=cv, scoring=scoring))
        print(f'K = {k} | {scoring.replace("_macro", "").upper()} = {f1:.3f} | Features: {feature} | Time for model training: {time.time()-model_training_start_time}')


if __name__ == '__main__':
    for dataset, dependent_variable in datasets.items():
        print(dataset.strip('.csv'))
        path = f'data/{dataset}'
        df = pd.read_csv(path)

        if path == 'data/Epileptic Seizure Recognition.csv':
            df.drop('Unnamed', axis=1, inplace=True)

        independent_variables = [feature for feature in df.columns if feature != dependent_variable]

        X = df[independent_variables]
        y = df[dependent_variable]

        if task == 'regression':
            model = RandomForestRegressor()
            metric = 'r2'
        elif task == 'classification':
            model = RandomForestClassifier(n_estimators=100)
            metric = 'f1_macro'
        else:
            ValueError(f'task must be either regression or classification')
            sys.exit()

        kf = KFold(n_splits=10)
        start = time.time()
        baseline = np.mean(cross_val_score(model, X, y, cv=kf, scoring=metric))
        print(f'Baseline ({len(independent_variables)} features, {len(y)} datapoints): {baseline:.3f}\ntime taken: {time.time()-start:.3f} seconds')

        filter_scores = apply_filter_method()
        calculate_score(filter_scores, algorithm=model, scoring=metric, cv=kf)
        embedded_scores = apply_embedded_method()
        calculate_score(embedded_scores, algorithm=model, scoring=metric, cv=kf)
        kgfarm_scores = apply_kgfarm_method()
        calculate_score(kgfarm_scores, algorithm=model, scoring=metric, cv=kf)
        wrapper_scores = apply_wrapper_method()
        calculate_score(wrapper_scores, algorithm=model, scoring=metric, cv=kf)

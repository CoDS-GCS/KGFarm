import os
import time
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.model_selection import cross_val_score, KFold
from operations.api import KGFarm

datasets = [{'Epileptic Seizure Recognition.csv': 'y'}]
K = [5, 10, 20]
kf = KFold(n_splits=10)


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
    tree = RandomForestClassifier()
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
    start_time = time.time()
    selector = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=None, step=1)
    selector.fit(X, y)
    rfe_scores = pd.DataFrame({'Feature': independent_variables, 'Ranking': selector.ranking_}). \
        sort_values(by='Ranking').reset_index(drop=True)
    print(f'Wrapper (RFE): {time.time() - start_time:.3f} seconds')
    return rfe_scores


def calculate_f1(feature_selection_info: pd.DataFrame):
    for k in K:
        feature = list(feature_selection_info.head(k)['Feature'])
        f1 = np.mean(cross_val_score(RandomForestClassifier(n_estimators=100), df[feature], y, cv=kf, scoring='f1_macro'))
        print(f'K = {k} | F1 = {f1:.3f} | Features: {feature}')


for dataset in datasets:
    path = f'data/{list(dataset.keys())[0]}'
    df = pd.read_csv(path)

    if path == 'data/Epileptic Seizure Recognition.csv':
        df.drop('Unnamed', axis=1, inplace=True)

    dependent_variable = list(dataset.values())[0]
    independent_variables = [feature for feature in df.columns if feature != dependent_variable]

    X = df[independent_variables]
    y = df[dependent_variable]

    baseline = np.mean(cross_val_score(RandomForestClassifier(n_estimators=100), X, y, cv=kf, scoring='f1_macro'))
    print(f'Baseline ({len(independent_variables)} features): {baseline:.3f}')

    filter_scores = apply_filter_method()
    calculate_f1(filter_scores)
    embedded_scores = apply_embedded_method()
    calculate_f1(embedded_scores)
    kgfarm_scores = apply_kgfarm_method()
    calculate_f1(kgfarm_scores)
    wrapper_scores = apply_wrapper_method()
    calculate_f1(wrapper_scores)

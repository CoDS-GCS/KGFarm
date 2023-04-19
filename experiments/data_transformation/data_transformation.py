import os
import sys
import time
import psutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
spath = str(Path(os.getcwd()).resolve().parents[1])
sys.path.append(spath)
sys.path.append(f'{spath}operations')
from operations.api import KGFarm
cwd = os.getcwd()
os.chdir('../../')

kgfarm = KGFarm(show_connection_status=False)

RANDOM_STATE = 30
np.random.seed(RANDOM_STATE)

if __name__ == '__main__':
    os.chdir(cwd)
    datasets = []
    f1_scores = []
    timings = []
    memory = []

    for dataset_info in tqdm(pd.read_csv('datasets.csv').to_dict('index').values()):
        dataset = dataset_info['Dataset']
        df = pd.read_csv(f'../data/lfe_datasets/{dataset}.csv')

        memory_before = psutil.Process().memory_info().rss
        start = time.time()

        fold = 1
        f1_per_fold = []
        for train_index, test_index in StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE).split(
                df, df[df.columns[-1]]):
            print(f'{dataset_info["Dataset"]} fold-{fold}')
            train_set = df.iloc[train_index]
            test_set = df.iloc[test_index]

            # using KGFarm for recommending data transformations
            recommended_transformations = kgfarm.recommend_data_transformations(entity_df=df[df.columns[:-1]], show_query=False, show_insights=False)

            # applying transformation (if recommended)
            if isinstance(recommended_transformations, type(None)):
                print('data does not requires transformation')

            else:  # transformations are applied separately on train and test sets to avoid feature-leakage
                for n, recommendation in recommended_transformations.to_dict('index').items():
                    train_set, _ = kgfarm.apply_transformation(
                        transformation_info=recommended_transformations.iloc[n],
                        entity_df=train_set, output_message='min')
                    test_set, _ = kgfarm.apply_transformation(
                        transformation_info=recommended_transformations.iloc[n],
                        entity_df=test_set, output_message='min')

            X_train = train_set[train_set.columns[:-1]]
            y_train = train_set[train_set.columns[-1]]
            X_test = test_set[test_set.columns[:-1]]
            y_test = test_set[test_set.columns[-1]]

            # training and evaluating ML model
            model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE)
            model.fit(X=X_train, y=y_train)
            y_out = model.predict(X=X_test)
            f1 = f1_score(y_true=y_test, y_pred=y_out)
            f1_per_fold.append(f1)
            fold = fold + 1

        time_taken = f'{(time.time() - start):.2f}'
        f1_per_dataset = f'{np.mean(f1_per_fold):.3f}'
        f1_scores.append(f1_per_dataset)
        timings.append(time_taken)
        memory_usage = f'{abs(psutil.Process().memory_info().rss - memory_before) / (1024 * 1024):.2f}'
        memory.append(memory_usage)
        datasets.append(dataset)

        results_df = pd.DataFrame({'Dataset': datasets, 'F1: KGFarm': f1_scores, 'Time: KGFarm': timings,
                                'Memory: KGFarm': memory})
        results_df = results_df[['Dataset', 'F1: KGFarm', 'Time: KGFarm', 'Memory: KGFarm']]
        results_df.to_csv('kgfarm_on_lfe_datasets.csv', index=False)
        print(results_df)

    print('Done.')

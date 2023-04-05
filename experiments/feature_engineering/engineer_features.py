import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold


class EngineerFeatures:
    def __init__(self, path: str):
        self.path_to_dataset = path
        self.experiment_datasets_info = pd.read_csv('datasets.csv')
        self.information_gain = dict()

    @staticmethod
    def __compute_information_gain(train_set: pd.DataFrame, test_set: pd.DataFrame, target: str):
        # TODO: compute IG on features of train-set and pick the very same from test-set
        return train_set, test_set

    @staticmethod
    def __compute_feature_correlation(train_set: pd.DataFrame, test_set: pd.DataFrame, target: str):
        # TODO: drop target
        return train_set, test_set

    @staticmethod
    def __transform():
        # use KGFarm
        pass

    @staticmethod
    def __select_features():
        # use KGFarm
        pass

    def run(self):
        for _, dataset_info in tqdm(self.experiment_datasets_info.to_dict('index').items(), desc='Datasets processed'):
            df = pd.read_csv(f'{self.path_to_dataset}/{dataset_info["Dataset"]}.csv', header=None)
            target = dataset_info['Target']

            for train_index, test_index in StratifiedKFold(n_splits=5, shuffle=True).split(df, df[target]):
                train_set = df.iloc[train_index]
                test_set = df.iloc[test_index]
                train_set, test_set = self.__compute_information_gain(train_set=train_set, test_set=test_set,
                                                                      target=target)
                train_set, test_set = self.__compute_feature_correlation(train_set=train_set, test_set=test_set,
                                                                      target=target)


# TODO: get sonar with header
if __name__ == '__main__':
    experiment = EngineerFeatures(path='/Users/shubhamvashisth/Documents/projects/AutoLearn')
    experiment.run()

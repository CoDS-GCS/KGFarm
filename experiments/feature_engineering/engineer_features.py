import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, f_classif, SelectKBest
from operations.api import KGFarm


class EngineerFeatures:
    def __init__(self, path: str, theta1: float, theta2: float):
        self.theta1 = theta1
        self.theta2 = theta2
        self.path_to_dataset = path
        self.experiment_datasets_info = pd.read_csv('datasets.csv')
        self.working_dir = os.getcwd()
        os.chdir('../../')
        self.kgfarm = KGFarm(show_connection_status=False)
        self.information_gain = dict()

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
        # print(pd.DataFrame({'Feature': self.information_gain.keys(), 'Score': self.information_gain.values()}))
        features_filtered_by_information_gain = list(self.information_gain.keys())
        features_filtered_by_information_gain.append(target)
        return train_set[features_filtered_by_information_gain], test_set[features_filtered_by_information_gain]

    def __compute_feature_correlation(self, train_set: pd.DataFrame, test_set: pd.DataFrame, target: str):
        features = train_set.drop(target, axis=1).corr()

        features_to_discard = set()
        for feature_i, i in zip(features.columns, range(len(features))):
            for j in range(i):
                if features.iloc[i, j] > self.theta2:
                    # print(f' feature {features.columns[j]} & {feature_i} have high correlation than {self.theta2}')
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
        recommended_transformations = self.kgfarm.recommend_data_transformations(entity_df=X, show_metadata=False,
                                                                                 show_query=False, show_insights=False)

        for n, recommendation in recommended_transformations.to_dict('index').items():
            train_set, _ = self.kgfarm.apply_transformation(transformation_info=recommended_transformations.iloc[n], entity_df=train_set)
            test_set, _ = self.kgfarm.apply_transformation(transformation_info=recommended_transformations.iloc[n], entity_df=test_set)

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
        feature_selection_recommendation = self.kgfarm.recommend_features_to_be_selected(
            entity_df=train_set[features_filtered_by_anova], dependent_variable=target, task=task)
        feature_selection_recommendation = feature_selection_recommendation.loc[feature_selection_recommendation['Selection_score'] > 0.60]

        features_filtered_by_kgfarm = feature_selection_recommendation['Feature'].tolist()
        features_filtered_by_kgfarm.append(target)
        return train_set[features_filtered_by_kgfarm], test_set[features_filtered_by_kgfarm]

    def __train_and_evaluate(self, train_set: pd.DataFrame, test_set: pd.DataFrame, target: str):
        X_train, y_train = self.__separate_independent_and_dependent_variables(df=train_set, target=target)
        X_test, y_test = self.__separate_independent_and_dependent_variables(df=test_set, target=target)
        random_forest_classifier = RandomForestClassifier()
        random_forest_classifier.fit(X=X_train, y=y_train)
        y_pred = random_forest_classifier.predict(X=X_test)
        return f1_score(y_true=y_test, y_pred=y_pred, average='macro')
        # return random_forest_classifier.score(X=X_test, y=y_test)

    def run(self):
        os.chdir(self.working_dir)
        for _, dataset_info in tqdm(self.experiment_datasets_info.to_dict('index').items(), desc='Datasets processed'):
            df = pd.read_csv(f'{self.path_to_dataset}{dataset_info["Dataset"]}.csv')
            target = dataset_info['Target']
            scores = list()
            for train_index, test_index in StratifiedKFold(n_splits=5, shuffle=True, random_state=1).split(df,
                                                                                                           df[target]):
                train_set = df.iloc[train_index]
                test_set = df.iloc[test_index]
                train_set, test_set = self.__compute_information_gain(train_set=train_set, test_set=test_set,
                                                                      target=target)

                train_set, test_set = self.__compute_feature_correlation(train_set=train_set, test_set=test_set,
                                                                         target=target)

                train_set, test_set = self.__transform(train_set=train_set, test_set=test_set, target=target)

                self.__select_features(train_set=train_set, test_set=test_set, target=target, task=dataset_info['Task'])

                scores.append(self.__train_and_evaluate(train_set=train_set, test_set=test_set, target=target))
            print(f'F1 for {dataset_info["Dataset"]}: {sum(scores) / len(scores):.3f}')


if __name__ == '__main__':
    experiment = EngineerFeatures(path='../data/', theta1=0.00, theta2=0.90)
    experiment.run()

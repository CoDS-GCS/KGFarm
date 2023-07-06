import sys
import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
np.random.seed(7)
sys.path.append('../')
from interface.apis import KGFarm
kgfarm = KGFarm()


class TestAPIs(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestAPIs, self).__init__(*args, **kwargs)

    def test_data_cleaning_recommendation(self):
        df = kgfarm.load_titanic_dataset()
        cleaning_info = kgfarm.recommend_cleaning_operations(df=df, visualize_missing_data=False)
        expected_cleaning_operations = ['Fill', 'Interpolate', 'Impute']

        self.assertEqual(list(cleaning_info['Recommended_operation']),
                         expected_cleaning_operations)

    def test_data_transformation_recommendation(self):
        df = kgfarm.load_titanic_dataset()
        df = df.dropna()
        X = df.drop('Survived', axis=1)
        transformation_info = kgfarm.recommend_transformations(X=X)
        expected_transformations = ['OrdinalEncoder', 'StandardScaler', 'Log']

        self.assertEqual(list(transformation_info['Recommended_transformation']),
                         expected_transformations)

    def test_feature_engineering(self):
        df = kgfarm.load_titanic_dataset()
        df = df.dropna()
        X = df.drop('Survived', axis=1)
        y = df['Survived']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        train_set = pd.concat([X_train, y_train], axis=1)
        test_set = pd.concat([X_test, y_test], axis=1)
        train_set, test_set = kgfarm.engineer_features(train_set=train_set, test_set=test_set, target='Survived')
        self.assertEqual(len(train_set.columns), len(test_set.columns))

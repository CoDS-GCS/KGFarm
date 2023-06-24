import sys
import unittest
import numpy as np
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

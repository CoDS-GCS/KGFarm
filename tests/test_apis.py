import sys
import unittest
import numpy as np
np.random.seed(7)
sys.path.append('../')
from interface.apis import KGFarm


class TestAPIs(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestAPIs, self).__init__(*args, **kwargs)
        self.kgfarm = KGFarm()
        self.df = self.kgfarm.load_titanic_dataset()
        self.df = self.df.dropna()
        self.X = self.df.drop('Survived', axis=1)
        self.y = self.df['Survived']

    def test_data_transformation(self):
        transformation_info = self.kgfarm.recommend_transformations(X=self.X)
        expected_transformations = ['StandardScaler', 'Log', 'OrdinalEncoder']

        self.assertEqual(list(transformation_info['Recommended_transformation']),
                         expected_transformations)


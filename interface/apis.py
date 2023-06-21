import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm
from sklearn.preprocessing import *
from inference_manager.inference import InferenceManager

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


class KGFarm:
    def __init__(self):
        sys.path.insert(0, str(Path(os.getcwd()).parent.absolute()))
        print('KGFarm is running in HUMAN-IN-THE-LOOP mode!')
        self.recommender = InferenceManager()

    @staticmethod
    def load_titanic_dataset():
        return pd.read_csv(os.path.dirname(os.path.abspath(__file__))+'/titanic.csv')

    def recommend_transformations(self, X: pd.DataFrame):
        return self.recommender.recommend_transformations(X=X)

    @staticmethod
    def apply_transformations(X: pd.DataFrame, recommendation: pd.Series):
        transformation = recommendation['Recommended_transformation']
        feature = recommendation['Feature']

        if transformation in {'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'QuantileTransformer',
                              'PowerTransformer'}:
            print(f'Applying {transformation} on {list(X.columns)}')
            if transformation == 'StandardScaler':
                scaler = StandardScaler()
            elif transformation == 'MinMaxScaler':
                scaler = MinMaxScaler()
            elif transformation == 'RobustScaler':
                scaler = RobustScaler()
            elif transformation == 'QuantileTransformer':
                scaler = QuantileTransformer()
            else:
                scaler = PowerTransformer()
            X[X.columns] = scaler.fit_transform(X=X[X.columns])
            return X, scaler

        elif transformation in {'Log', 'Sqrt', 'square'}:
            print(f'Applying {transformation} on {list(feature)}')
            if transformation == 'Log':
                def log_plus_const(x, const=0):
                    return np.log(x + np.abs(const) + 0.0001)

                for f in tqdm(feature):
                    min_neg_val = X[f].min()
                    unary_transformation_model = FunctionTransformer(func=log_plus_const,
                                                                     kw_args={'const': min_neg_val}, validate=True)
                    X[f] = unary_transformation_model.fit_transform(X=np.array(X[f]).reshape(-1, 1))

            elif transformation == 'Sqrt':
                def sqrt_plus_const(x, const=0):
                    return np.sqrt(x + np.abs(const) + 0.0001)

                for f in tqdm(feature):
                    min_neg_val = X[f].min()
                    unary_transformation_model = FunctionTransformer(func=sqrt_plus_const,
                                                                     kw_args={'const': min_neg_val}, validate=True)
                    X[f] = unary_transformation_model.fit_transform(X=np.array(X[f]).reshape(-1, 1))
            else:
                unary_transformation_model = FunctionTransformer(func=np.square, validate=True)
                X[feature] = unary_transformation_model.fit_transform(X=X[feature])
            return X, transformation

        elif transformation in {'OrdinalEncoder', 'OneHotEncoder'}:
            print(f'Applying {transformation} on {list(feature)}')
            if transformation == 'OrdinalEncoder':
                encoder = OrdinalEncoder()
                X[feature] = encoder.fit_transform(X=X[feature])
            else:
                encoder = OneHotEncoder(handle_unknown='ignore')
                one_hot_encoded_features = pd.DataFrame(encoder.fit_transform(X[feature]).toarray())
                X = X.join(one_hot_encoded_features)
                X = X.drop(feature, axis=1)
            return X, encoder

        else:
            raise ValueError(f'{transformation} not supported')

    def recommend_features_to_be_selected(self, task: str, X: pd.DataFrame, y: pd.Series,
                                          k: int = None):
        entity_df = pd.concat([X, y], axis=1)

        if k is None or len(entity_df) < k:
            k = len(entity_df)

        return self.recommender.get_feature_selection_score(task=task, entity_df=entity_df.sample(n=k, random_state=1),
                                                            dependent_variable=str(y.name))

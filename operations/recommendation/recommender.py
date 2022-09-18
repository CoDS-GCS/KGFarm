import joblib
import torch
import bitstring
import numpy as np
import pandas as pd
from datasketch import MinHash
from tqdm import tqdm

from operations.recommendation.utils.column_embeddings import load_numeric_embedding_model


class Recommender:
    def __init__(self):
        self.numeric_transformation_recommender = joblib.load(
            'operations/recommendation/utils/models/transformation_recommender_numeric.pkl')
        self.categorical_transformation_recommender = joblib.load(
            'operations/recommendation/utils/models/transformation_recommender_string.pkl')
        self.numeric_encoder = joblib.load('operations/recommendation/utils/models/encoder_numeric.pkl')
        self.categorical_encoder = joblib.load('operations/recommendation/utils/models/encoder_string.pkl')
        self.numeric_embedding_model = load_numeric_embedding_model()
        self.categorical_embedding_model = MinHash(num_perm=512)

    def get_transformation_recommendations(self, entity_df: pd.DataFrame):
        transformation_info = {}
        numeric_column_embeddings = {}
        categorical_column_embeddings = {}

        def compute_embeddings():
            print('computing feature embeddings')

            def get_bin_repr(val):
                return [int(j) for j in bitstring.BitArray(float=float(val), length=32).bin]

            for column in tqdm(entity_df.columns):
                if pd.api.types.is_numeric_dtype(entity_df[column]):
                    bin_repr = entity_df[column].apply(get_bin_repr, convert_dtype=False).to_list()
                    bin_tensor = torch.FloatTensor(bin_repr).to('cpu')
                    with torch.no_grad():
                        embedding_tensor = self.numeric_embedding_model(bin_tensor).mean(axis=0)
                    numeric_column_embeddings[column] = embedding_tensor.tolist()
                else:
                    column_value = list(entity_df[column])
                    self.categorical_embedding_model = MinHash(num_perm=512)
                    for word in column_value:
                        if isinstance(word, str):
                            self.categorical_embedding_model.update(word.lower().encode('utf8'))
                    categorical_column_embeddings[column] = self.categorical_embedding_model.hashvalues.tolist()

        def classify_numeric_transformation():
            for column, embedding in numeric_column_embeddings.items():
                transformation_info[column] = self.numeric_encoder. \
                    inverse_transform(np.array(self.numeric_transformation_recommender. \
                                               predict(np.array(embedding).reshape(1, -1))[0]).reshape(1, -1))[0]

        def classify_categorical_transformation():
            for column, embedding in categorical_column_embeddings.items():
                transformation_info[column] = self.categorical_encoder. \
                    inverse_transform(np.array(self.categorical_transformation_recommender. \
                                               predict(np.array(embedding).reshape(1, -1))[0]).reshape(1, -1))[0]

        compute_embeddings()
        classify_numeric_transformation()
        classify_categorical_transformation()
        return transformation_info

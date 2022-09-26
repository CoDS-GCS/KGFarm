import torch
import joblib
import bitstring
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasketch import MinHash
from feature_discovery.src.recommender.word_embeddings import WordEmbedding
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
        self.word_embedding = WordEmbedding('feature_discovery/src/recommender/utils/glove_embeddings/glove.6B.100d.pickle')
        self.auto_insight_report = dict()

    def get_transformation_recommendations(self, entity_df: pd.DataFrame):
        self.auto_insight_report = {}
        transformation_info = {}
        numeric_column_embeddings = {}
        categorical_column_embeddings = {}
        word_embeddings = {}

        def compute_content_embeddings():
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

        def compute_word_embeddings():
            for column in entity_df.columns:
                tokens = self.word_embedding.tokenize(column)
                word_embeddings[column] = self.word_embedding.calculate_word_embeddings(tokens)

        def classify_numeric_transformation():
            for column, embedding in numeric_column_embeddings.items():
                embedding.extend(word_embeddings.get(column))
                predicted_transformation = self.numeric_encoder. \
                    inverse_transform(np.array(self.numeric_transformation_recommender. \
                                               predict(np.array(embedding).reshape(1, -1))[0]).reshape(1, -1))[0]
                transformation_info[column] = predicted_transformation
                if predicted_transformation == 'LabelEncoder' or predicted_transformation == 'OneHotEncoder':
                    self.auto_insight_report[column] = predicted_transformation

        def classify_categorical_transformation():
            for column, embedding in categorical_column_embeddings.items():
                embedding.extend(word_embeddings.get(column))
                predicted_transformation = self.categorical_encoder. \
                    inverse_transform(np.array(self.categorical_transformation_recommender. \
                                               predict(np.array(embedding).reshape(1, -1))[0]).reshape(1, -1))[0]
                transformation_info[column] = predicted_transformation

        def reformat(df):
            transformation_info_grouped = []
            feature = []
            transformation = None
            for row_number, value in df.to_dict('index').items():
                if transformation == value['Transformation']:
                    feature.append(value['Feature'])
                    if row_number == len(df) - 1:  # last row
                        row = df.to_dict('index').get(row_number - 1)
                        transformation_info_grouped.append({'Transformation': transformation,
                                                            'Package': row['Package'],
                                                            'Library': row['Library'],
                                                            'Feature': feature})
                else:
                    if row_number == 0:
                        transformation = value['Transformation']
                        feature = [value['Feature']]
                        continue
                    row = df.to_dict('index').get(row_number - 1)
                    transformation_info_grouped.append({'Transformation': transformation,
                                                        'Package': row['Package'],
                                                        'Library': row['Library'],
                                                        'Feature': feature})
                    transformation = value['Transformation']
                    feature = [value['Feature']]
                    if row_number == len(df) - 1:  # add if last transformation has single feature
                        transformation_info_grouped.append({'Transformation': transformation,
                                                            'Package': row['Package'],
                                                            'Library': row['Library'],
                                                            'Feature': feature})

            df = pd.DataFrame(transformation_info_grouped)
            return df

        def show_insights():
            # TODO: Add insights for robust scalar -> outlier
            print('• Insights about your entity_df:')
            insight_n = 1
            for column, transformation in self.auto_insight_report.items():
                print('\t{}. {} (a numeric column) looks like a categorical feature'.format(insight_n, column))
                insight_n = insight_n + 1

        compute_content_embeddings()
        compute_word_embeddings()
        classify_numeric_transformation()
        classify_categorical_transformation()
        transformation_info = pd.DataFrame.from_dict({'Feature': list(transformation_info.keys()),
                                                      'Transformation': list(transformation_info.values()),
                                                      'Package': 'preprocessing',
                                                      'Library': 'sklearn'})
        transformation_info = transformation_info[transformation_info['Transformation'] != 'Negative']
        transformation_info.sort_values(by='Transformation', inplace=True)
        transformation_info.reset_index(drop=True, inplace=True)
        if self.auto_insight_report:
            show_insights()
        return reformat(transformation_info)

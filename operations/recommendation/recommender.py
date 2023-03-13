import operator
import torch
import joblib
import bitstring
import numpy as np
import pandas as pd
from datasketch import MinHash
from collections import Counter
from pyspark.sql.functions import *
from sklearn.preprocessing import LabelEncoder
from operations.storage.embeddings import Embeddings
from operations.recommendation.utils.column_embeddings import load_numeric_embedding_model


class Recommender:
    def __init__(self):
        # KGFarm transformation recommendation
        self.numeric_transformation_recommender = joblib.load(
            'operations/recommendation/utils/models/transformation_recommender_numerical.pkl')
        self.categorical_transformation_recommender = joblib.load(
            'operations/recommendation/utils/models/transformation_recommender_categorical.pkl')
        self.numeric_encoder = joblib.load('operations/recommendation/utils/models/encoder_numerical.pkl')
        self.categorical_encoder = joblib.load('operations/recommendation/utils/models/encoder_categorical.pkl')
        self.numeric_embedding_model = load_numeric_embedding_model()
        self.categorical_embedding_model = MinHash(num_perm=512)
        # self.word_embedding = WordEmbedding(
        #     'feature_discovery/src/recommender/utils/glove_embeddings/glove.6B.100d.pickle')
        self.embeddings = Embeddings(
            'operations/storage/embedding_store/embeddings_120K.pickle')  # column embeddings, ~120k column profile embeddings, solves cold start
        self.auto_insight_report = dict()
        self.categorical_thresh = 0.60
        self.numerical_thresh = 0.50
        # KGFarm feature selector
        self.feature_selector = None

    def __compute_content_embeddings(self,
                                     entity_df: pd.DataFrame):  # DDE for numeric columns, Minhash for string columns
        numeric_column_embeddings = {}
        categorical_column_embeddings = {}

        def get_bin_repr(val):
            return [int(j) for j in bitstring.BitArray(float=float(val), length=32).bin]

        for column in entity_df.columns:
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
        return numeric_column_embeddings, categorical_column_embeddings

    # def __compute_word_embeddings(self, entity_df: pd.DataFrame):  # glove embeddings
    #     word_embeddings = {}
    #     for column in entity_df.columns:
    #         tokens = self.word_embedding.tokenize(column)
    #         word_embeddings[column] = self.word_embedding.calculate_word_embeddings(tokens)
    #     return word_embeddings

    def get_transformation_recommendations(self, entity_df: pd.DataFrame):
        self.auto_insight_report = {}
        transformation_info = {}

        def classify_numeric_transformation(numeric_column_embeddings: dict):  # , word_embeddings: dict):
            for column, embedding in numeric_column_embeddings.items():
                # embedding.extend(word_embeddings.get(column))
                probability = self.numeric_transformation_recommender.predict_proba(np.array(embedding).
                                                                                    reshape(1, -1))[0]
                # print(f'{column} - {probability}')
                if max(probability) >= self.numerical_thresh:
                    predicted_transformation = self.numeric_encoder. \
                        inverse_transform(np.array(self.numeric_transformation_recommender. \
                                                   predict(np.array(embedding).reshape(1, -1))[0]).reshape(1, -1))[0]
                else:
                    predicted_transformation = 'Negative'
                transformation_info[column] = predicted_transformation
                if predicted_transformation == 'Nominal encoding' or predicted_transformation == 'Ordinal encoding':
                    self.auto_insight_report[column] = predicted_transformation

        def classify_categorical_transformation(categorical_column_embeddings: dict):  # , word_embeddings: dict):
            for column, embedding in categorical_column_embeddings.items():
                # embedding.extend(word_embeddings.get(column))
                probability = self.categorical_transformation_recommender.predict_proba(np.array(embedding).
                                                                                        reshape(1, -1))[0]
                # print(f'{column} - {probability}')
                if max(probability) >= self.categorical_thresh:
                    predicted_transformation = self.categorical_encoder. \
                        inverse_transform(np.array(self.categorical_transformation_recommender. \
                                                   predict(np.array(embedding).reshape(1, -1))[0]).reshape(1, -1))[0]
                else:
                    predicted_transformation = 'Negative'
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
                print(f'{insight_n}. {column} is a numeric column that looks like a categorical feature')
                insight_n = insight_n + 1

        numeric_embeddings, string_embeddings = self.__compute_content_embeddings(entity_df=entity_df)
        # word_column_embeddings = self.__compute_word_embeddings(entity_df=entity_df)
        classify_numeric_transformation(
            numeric_column_embeddings=numeric_embeddings)  # , word_embeddings=word_column_embeddings)
        classify_categorical_transformation(
            categorical_column_embeddings=string_embeddings)  # , word_embeddings=word_column_embeddings)
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

    def get_cleaning_recommendation(self, entity_df: pd.DataFrame):
        numeric_embeddings, string_embeddings = self.__compute_content_embeddings(entity_df=entity_df)
        similar_columns_uris = self.embeddings.get_similar_columns(numeric_column_embeddings=numeric_embeddings,
                                                                   string_column_embeddings=string_embeddings)

        # get corresponding table uris
        def get_table_uri(column_uri):
            return column_uri.replace(column_uri.split('/')[-1], '')[:-1]

        similar_table_uris = {key: get_table_uri(value) for key, value in similar_columns_uris.items()}
        similar_table_uris = Counter(
            similar_table_uris.values())  # count similar tables and sort by most occurring tables
        return tuple(dict(sorted(similar_table_uris.items(), key=lambda item: item[1], reverse=True)).keys())

    def get_feature_selection_score(self, task: str, entity_df: pd.DataFrame, dependent_variable: str):

        def get_bin_repr(val):
            return [int(j) for j in bitstring.BitArray(float=float(val), length=32).bin]

        if task == 'regression':
            print(f'loading operations/recommendation/utils/models/feature_selector_{task}.pkl')
            self.feature_selector = joblib.load(f'operations/recommendation/utils/models/feature_selector_{task}.pkl')
        elif task == 'multi-class':
            print(f'loading operations/recommendation/utils/models/feature_selector_{task.replace("-", "")}_classification.pkl')
            entity_df[dependent_variable] = LabelEncoder().fit_transform(entity_df[dependent_variable])
            self.feature_selector = joblib.load(f'operations/recommendation/utils/models/feature_selector_{task.replace("-", "")}_classification.pkl')
        elif task == 'binary':
            print(f'loading operations/recommendation/utils/models/feature_selector_{task}_classification.pkl')
            entity_df[dependent_variable] = LabelEncoder().fit_transform(entity_df[dependent_variable])
            self.feature_selector = joblib.load(f'operations/recommendation/utils/models/feature_selector_{task}_classification.pkl')

        numerical_features = [feature for feature in entity_df.columns if
                              feature != dependent_variable and pd.api.types.is_numeric_dtype(entity_df[feature])]
        categorical_features = set(list(entity_df.columns)) - set(numerical_features)
        categorical_features.remove(dependent_variable)
        n_suggestion = 1
        for feature in categorical_features:
            print(
                f"{n_suggestion}. feature '{feature}' is non-numeric and should be transformed or dropped before selection")
            n_suggestion = n_suggestion + 1

        if pd.api.types.is_numeric_dtype(entity_df[dependent_variable]):
            bin_repr = entity_df[dependent_variable].apply(get_bin_repr, convert_dtype=False).to_list()
            bin_tensor = torch.FloatTensor(bin_repr).to('cpu')
            with torch.no_grad():
                target_embedding = self.numeric_embedding_model(bin_tensor).mean(axis=0).tolist()

        else:
            raise ValueError(f"target '{dependent_variable}' is not numeric and needs to be transformed")

        # get embeddings for numerical features
        numeric_feature_embeddings, _ = self.__compute_content_embeddings(entity_df[numerical_features])
        numeric_feature_embeddings = {feature: embedding + target_embedding for feature, embedding in
                                      numeric_feature_embeddings.items()}

        selection_info = {feature: (self.feature_selector.predict_proba(np.array(embedding).reshape(1, -1)).tolist()[0][1])
                for feature, embedding in numeric_feature_embeddings.items()}
        selection_info = dict(sorted(selection_info.items(), key=operator.itemgetter(1), reverse=True))

        selection_info = pd.DataFrame({'Feature': selection_info.keys(), 'Selection_score': selection_info.values()})

        def cal_selection_score(score, max_score_value):
            return score / max_score_value

        max_score = selection_info['Selection_score'].max()
        selection_info['Selection_score'] = selection_info['Selection_score'].apply(lambda x: cal_selection_score(score=x, max_score_value=max_score))

        return selection_info

    """
    def get_feature_selection_score_distributed(self, entity_df: pyspark.sql.dataframe.DataFrame):
        def compute_deep_embeddings(col):
            bin_repr = [[int(j) for j in bitstring.BitArray(float=float(i), length=32).bin] for i in col]
            bin_tensor = torch.FloatTensor(bin_repr).to('cpu')
            with torch.no_grad():
                embedding_tensor = self.numeric_embedding_model(bin_tensor).mean(axis=0)
            return embedding_tensor.tolist()

        deep_embeddingsUDF = udf(lambda z: compute_deep_embeddings(z), ArrayType(FloatType()))
        cols = entity_df.columns
        cols2 = ['`' + c + '`' for c in cols]
        df2 = entity_df.select([collect_list(c) for c in cols2]).toDF(*cols2)
        df2 = df2.toDF(*cols)
        for col in cols:
            df2 = df2.withColumn(col, deep_embeddingsUDF('`' + col + '`'))
        return df2
        """
import os
import torch
import joblib
import operator
import bitstring
import chars2vec
import numpy as np
import pandas as pd
from datasketch import MinHash
from dask.dataframe import from_pandas
from inference_manager.column_embeddings import load_numeric_embedding_model, load_embedding_model_for_cleaning


class InferenceManager:
    def __init__(self):
        # resolve model path
        kgfarm_storage_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/storage/'

        # load CoLR Embedding models
        self.numeric_embedding_model = load_numeric_embedding_model(
            model_path=kgfarm_storage_path + '20211123161253_numerical_embedding_model_epoch_4_3M_samples_gpu_cluster.pt')
        self.categorical_embedding_model = MinHash(num_perm=512)
        self.numeric_embedding_model_cleaning = load_embedding_model_for_cleaning(model_path=kgfarm_storage_path + '20221030142854_numerical_model_embedding_epoch_35.pt', model_type='numerical')
        self.categorical_embedding_model_cleaning = load_embedding_model_for_cleaning(model_path=kgfarm_storage_path + '20221020165957_string_model_embedding_epoch_100.pt', model_type='categorical')

        # load KGFarm Data Transformation models
        self.numeric_transformation_recommender = joblib.load(kgfarm_storage_path+'transformation_recommender_numerical.pkl')
        self.categorical_transformation_recommender = joblib.load(kgfarm_storage_path+'transformation_recommender_categorical.pkl')
        self.numeric_encoder = joblib.load(kgfarm_storage_path+'encoder_numerical.pkl')
        self.categorical_encoder = joblib.load(kgfarm_storage_path+'encoder_categorical.pkl')
        self.scaler_model = joblib.load(kgfarm_storage_path+'scaling_transformation_recommender.pkl')
        self.scaler_encoder = joblib.load(kgfarm_storage_path+'scaling_encoder.pkl')
        self.unary_model = joblib.load(kgfarm_storage_path+'unary_transformation_recommender.pkl')
        self.unary_encoder = joblib.load(kgfarm_storage_path+'unary_encoder.pkl')

        # load KGFarm Data Cleaning models
        self.cleaning_recommender = joblib.load(kgfarm_storage_path + 'cleaning.pkl')
        self.outlier_cleaning_recommender = joblib.load(kgfarm_storage_path + 'outlier_cleaning.pkl')

        # load KGFarm Feature Selection models
        self.binary_feature_selection_model = joblib.load(kgfarm_storage_path+'feature_selector_binary_classification.pkl')
        self.multiclass_feature_selection_model = joblib.load(kgfarm_storage_path + 'feature_selector_multiclass_classification.pkl')
        self.regression_feature_selection_model = joblib.load(kgfarm_storage_path + 'feature_selector_regression.pkl')

        # thresholds
        self.unary_transformation_threshold = 0.60
        self.categorical_thresh = 0.60
        self.numerical_thresh = 0.50

        # mapping info
        self.auto_insight_report = dict()
        self.transformation_technique = {'log': 'Log',
                                         'sqrt': 'Sqrt',
                                         'Ordinal encoding': 'OrdinalEncoder',
                                         'Nominal encoding': 'OneHotEncoder'}
        self.transformation_type = {'Log': 'unary',
                                    'Sqrt': 'unary',
                                    'OrdinalEncoder': 'categorical',
                                    'OneHotEncoder': 'categorical'}

    def compute_content_embeddings(self, entity_df: pd.DataFrame):  # DDE for numeric columns, Minhash for string columns
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

    def compute_content_embedding_parallel(self, df: pd.DataFrame):
        def apply_dask(entity_df: pd.DataFrame, n_workers: int = os.cpu_count() - 1):
            def get_features_type(feature_info: pd.Series):
                return feature_info.apply(lambda x: 'categorical' if x == 'object' else 'numerical').to_dict()

            def get_numerical_and_categorical_features(types: dict):
                return [k for k, v in types.items() if v == 'numerical'], [k for k, v in types.items() if
                                                                           v == 'categorical']

            def convert_to_bin_representation(row):
                return row.map(lambda x: [int(j) for j in bitstring.BitArray(float=float(x), length=32).bin])

            def get_meta(columns: list):
                return dict((column, 'object') for column in columns)

            def get_numerical_embeddings(partition):
                # partition = partition.values if isinstance(partition, pd.Series) else partition
                with torch.no_grad():
                    return self.numeric_embedding_model(torch.FloatTensor(list(partition)).to('cpu')).mean(
                        axis=0).tolist()

            def get_categorical_embeddings(partition):
                categorical_embedding_model = MinHash(num_perm=512)
                partition.map(lambda x: categorical_embedding_model.update(x.lower().encode('utf8')))
                return categorical_embedding_model.hashvalues.tolist()

            # determine numerical and categorical features
            features_type = get_features_type(feature_info=entity_df.dtypes)
            numerical_features, categorical_features = get_numerical_and_categorical_features(features_type)

            # transpose pandas dataframe and convert to dask dataframe
            numerical_features_df = from_pandas(entity_df[numerical_features].transpose(), npartitions=n_workers)
            categorical_features_df = from_pandas(entity_df[categorical_features].transpose(), npartitions=n_workers)

            # calculate embeddings using dask operations
            numerical_embeddings = numerical_features_df. \
                apply(convert_to_bin_representation, axis=1, meta=get_meta(columns=numerical_features_df.columns)). \
                apply(get_numerical_embeddings, axis=1, meta=('x', 'object'))
            categorical_embeddings = categorical_features_df.apply(get_categorical_embeddings, axis=1,
                                                                           meta=('x', 'object'))

            return numerical_embeddings, categorical_embeddings
        numerical_feature_embeddings, categorical_feature_embeddings = apply_dask(
            entity_df=df)
        return numerical_feature_embeddings.compute().to_dict(), categorical_feature_embeddings.compute().to_dict()

    def compute_cleaning_content_embeddings(self, df: pd.DataFrame):
        numeric_column_embeddings = {}
        categorical_column_embeddings = {}

        def get_bin_repr(val):
            return [int(j) for j in bitstring.BitArray(float=float(val), length=32).bin]

        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                bin_repr = df[column].apply(get_bin_repr, convert_dtype=False).to_list()
                bin_tensor = torch.FloatTensor(bin_repr).to('cpu')
                embedding_tensor = self.numeric_embedding_model_cleaning(bin_tensor).mean(axis=0)
                numeric_column_embeddings[column] = embedding_tensor.tolist()
            else:
                char_level_embed_model = chars2vec.load_model('eng_50')
                input_vector = char_level_embed_model.vectorize_words(df[column].dropna().tolist())
                input_tensor = torch.FloatTensor(input_vector).to('cpu')
                embedding_tensor = self.categorical_embedding_model_cleaning(input_tensor).mean(axis=0)
                categorical_column_embeddings[column] = embedding_tensor.tolist()
        return numeric_column_embeddings, categorical_column_embeddings

    def recommend_transformations(self, X: pd.DataFrame):
        recommendations = []

        def average_embeddings(embeddings: list):
            number_of_embeddings = len(embeddings)
            if number_of_embeddings == 1:
                return embeddings[0]
            else:
                embedding_length = len(embeddings[0])
                avg_embeddings = [0] * embedding_length
                for embedding in embeddings:
                    for i, e in enumerate(embedding):
                        avg_embeddings[i] = avg_embeddings[i] + e
                return [avg_embeddings[i] / number_of_embeddings for i in range(len(avg_embeddings))]

        numerical_feature_embeddings, categorical_feature_embeddings = self.compute_content_embedding_parallel(
            df=X)

        try:
            # scaling transformation
            recommended_scaler_transformation = list(self.scaler_encoder.inverse_transform(self.scaler_model. \
                predict(np.array(average_embeddings(embeddings=list(numerical_feature_embeddings.values()))).reshape(1, -1))))[0]
            recommendations.append(pd.DataFrame({'Feature': [list(X.columns)], 'Recommended_transformation': recommended_scaler_transformation}))

            # unary transformation (numeric features)
            numeric_embedding_df = pd.DataFrame(
                {'Feature': numerical_feature_embeddings.keys(), 'Embedding': numerical_feature_embeddings.values()})
            numeric_embedding_df['Probability'] = self.unary_model.predict_proba(list(numeric_embedding_df['Embedding'])).tolist()
            numeric_embedding_df['Transform'] = numeric_embedding_df['Probability'].apply(
                lambda x: True if self.unary_transformation_threshold <= max(x) else False)
            numeric_embedding_df = numeric_embedding_df.loc[numeric_embedding_df['Transform'] == True]
            if numeric_embedding_df.empty:
                recommendations.append(pd.DataFrame(columns=['Feature', 'Recommended_transformation']))
            else:
                numeric_embedding_df['Recommended_transformation'] = self.unary_encoder.inverse_transform(
                    self.unary_model.predict(list(numeric_embedding_df['Embedding']))).tolist()
                unary_numeric_transformation_df = numeric_embedding_df[['Feature', 'Recommended_transformation']].groupby('Recommended_transformation')['Feature'].apply(list).to_frame()
                unary_numeric_transformation_df['Recommended_transformation'] = list(unary_numeric_transformation_df.index)
                unary_numeric_transformation_df = unary_numeric_transformation_df.reset_index(drop=True)
                recommendations.append(unary_numeric_transformation_df)
        except IndexError:
            recommendations.append(
                pd.DataFrame({'Feature': [list(X.columns)], 'Recommended_transformation': 'StandardScaler'}))

        # unary transformation (categorical features)
        if categorical_feature_embeddings:
            categorical_embedding_df = pd.DataFrame(
                {'Feature': categorical_feature_embeddings.keys(), 'Embedding': categorical_feature_embeddings.values()})
            categorical_embedding_df['Probability'] = self.categorical_transformation_recommender.predict_proba(
                list(categorical_embedding_df['Embedding'])).tolist()
            categorical_embedding_df['Transform'] = categorical_embedding_df['Probability'].apply(
                lambda x: True if self.unary_transformation_threshold <= max(x) else False)
            categorical_embedding_df = categorical_embedding_df.loc[categorical_embedding_df['Transform'] == True]
            if categorical_embedding_df.empty:
                recommendations.append(pd.DataFrame(columns=['Feature', 'Recommended_transformation']))
            else:
                categorical_embedding_df['Recommended_transformation'] = self.categorical_encoder.inverse_transform(
                    self.categorical_transformation_recommender.predict(list(categorical_embedding_df['Embedding']))).tolist()
                unary_categorical_transformation_df = categorical_embedding_df[['Feature', 'Recommended_transformation']].groupby('Recommended_transformation')['Feature'].apply(list).to_frame()
                unary_categorical_transformation_df['Recommended_transformation'] = list(unary_categorical_transformation_df.index)
                unary_categorical_transformation_df = unary_categorical_transformation_df.reset_index(drop=True)
                recommendations.append(unary_categorical_transformation_df)

        recommendations = pd.concat(recommendations).reset_index(drop=True)
        recommendations['Recommended_transformation'] = recommendations['Recommended_transformation'].apply(lambda x: self.transformation_technique.get(x) if x in self.transformation_technique else x)
        recommendations['Transformation_type'] = recommendations['Recommended_transformation'].apply(lambda x: self.transformation_type.get(x) if x in self.transformation_type else 'scaling')

        # sort transformation recommendations by categorical, scaling and finally unary
        sequence_in_which_transformations_appear = ['categorical', 'scaling', 'unary']
        recommendations = recommendations.sort_values('Transformation_type', key=lambda x: x.map({v: i for i, v in enumerate(sequence_in_which_transformations_appear)}))
        return recommendations.reset_index(drop=True)

    def get_cleaning_recommendation(self, df: pd.DataFrame):
        # get embeddings for columns
        numeric_embeddings, string_embeddings = self.compute_cleaning_content_embeddings(df=df)
        # Get the average of the values for the string embeddings as well as the numerical embeddings
        array_sizes = {}
        for key, value in string_embeddings.items():
            array_sizes[key] = len(value)
        if string_embeddings:
            string_embeddings_avg = np.mean(list(string_embeddings.values()), axis=0)
        else:
            string_embeddings_avg = np.zeros(300)
        if numeric_embeddings:
            numeric_embeddings_avg = np.mean(list(numeric_embeddings.values()), axis=0)
        else:
            numeric_embeddings_avg = np.zeros(300)
        embedding = np.concatenate((string_embeddings_avg, numeric_embeddings_avg))
        # make prediction
        probability = self.cleaning_recommender.predict(np.array(embedding).reshape(1, -1))[0]
        probability = pd.DataFrame(probability)
        probability.index = np.array(['Impute', 'Fill', 'Interpolate'])
        probability.columns = ['Confidence']
        probability = probability.sort_values('Confidence', ascending=False)
        probability['Recommended_operation'] = probability.index
        probability['Confidence'] = probability['Confidence'].apply(lambda x: float(f'{x/100:.2f}'))
        return probability.reset_index(drop=True)[['Recommended_operation', 'Confidence']]

    def check_for_outliers(self, df: pd.DataFrame):
        numeric_embeddings, _ = self.compute_cleaning_content_embeddings(df=df)
        numeric_embeddings_avg = np.mean(list(numeric_embeddings.values()), axis=0)
        if self.outlier_cleaning_recommender.predict(numeric_embeddings_avg.reshape(1, -1)) == 1:
            return True
        else:
            return False

    def get_feature_selection_score(self, task: str, entity_df: pd.DataFrame, dependent_variable: str):

        def get_bin_repr(val):
            return [int(j) for j in bitstring.BitArray(float=float(val), length=32).bin]

        if task not in {'regression', 'multiclass', 'binary'}:
            raise ValueError('task must be binary or multiclass or regression')

        if task == 'regression':
            feature_selector = self.regression_feature_selection_model
        elif task == 'multiclass':
            feature_selector = self.multiclass_feature_selection_model
        else:
            feature_selector = self.binary_feature_selection_model

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
        numeric_feature_embeddings, _ = self.compute_content_embeddings(entity_df[numerical_features])
        numeric_feature_embeddings = {feature: embedding + target_embedding for feature, embedding in
                                      numeric_feature_embeddings.items()}

        selection_info = {
            feature: (feature_selector.predict_proba(np.array(embedding).reshape(1, -1)).tolist()[0][1])
            for feature, embedding in numeric_feature_embeddings.items()}
        selection_info = dict(sorted(selection_info.items(), key=operator.itemgetter(1), reverse=True))

        selection_info = pd.DataFrame({'Feature': selection_info.keys(), 'Selection_score': selection_info.values()})

        def cal_selection_score(score, max_score_value):
            return score / max_score_value

        max_score = selection_info['Selection_score'].max()
        selection_info['Selection_score'] = selection_info['Selection_score'].apply(
            lambda x: cal_selection_score(score=x, max_score_value=max_score))

        return selection_info

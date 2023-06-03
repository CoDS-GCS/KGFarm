import os
import copy
import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from datetime import timedelta
from tqdm.notebook import tqdm
from operations.template import *
from sklearn.preprocessing import *
from matplotlib import pyplot as plt
from helpers.helper import connect_to_stardog
from sklearn.impute import  SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from operations.recommendation.recommender import Recommender
from feature_discovery.src.graph_builder.governor import Governor

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
import sys


class KGFarm:
    def __init__(self, mode: str = 'Human in the loop', port: object = 5820, database: str = 'kgfarm_test',
                 show_connection_status: bool = True):
        sys.path.insert(0, str(Path(os.getcwd()).parent.absolute()))
        self.mode = mode
        if mode not in ['Human in the loop', 'Automatic']:
            raise ValueError("mode can be either 'Human in the Loop' or 'Automatic'")
        print('(KGFarm is running in {} mode)'.format(mode))
        self.config = connect_to_stardog(port, database, show_connection_status)
        if mode == 'Human in the loop':
            self.recommender = Recommender()
            self.recommender_config = connect_to_stardog(port, db='kgfarm_recommender', show_status=False)
        self.governor = Governor(self.config)
        self.__table_transformations = {}  # cols in enriched_df: tuple -> (entity_df_id, feature_view_id)
        """
        conf = SparkConf().setAppName('KGFarm')
        conf = (conf.setMaster('local[*]')
                .set('spark.executor.memory', '10g')
                .set('spark.driver.memory', '5g')
                .set('spark.driver.maxResultSize', '5g'))
        sc = SparkContext(conf=conf)
        self.spark = SparkSession(sc)
        """

    # re-arranging columns
    @staticmethod
    def __re_arrange_columns(last_column: str, df: pd.DataFrame):
        features = list(df.columns)
        features.remove(last_column)
        features.append(last_column)
        df = df[features]
        return df

    def __check_if_profiled(self, df: pd.DataFrame):
        table_id = search_entity_table(self.config, list(df.columns))
        if len(table_id) == 0:
            # search for enriched tables
            table_ids = self.__table_transformations.get(tuple(df.columns))
            if table_ids is None:
                return False  # unseen table
            else:
                return table_ids  # enriched table (return table urls which make the enriched table)

        else:
            return table_id['Table_id'][0]  # seen / profiled table

    # wrapper around pd.read_csv()
    def load_table(self, table_info: pd.Series, print_table_name: bool = True):
        table = table_info['Table']
        dataset = table_info['Dataset']
        if print_table_name:
            print(table)
        return pd.read_csv(get_table_path(self.config, table, dataset))

    def get_entities(self, show_query: bool = False):
        entity_df = get_entities(self.config, show_query)
        entity_df['Entity_data_type'] = entity_df['Entity_data_type'].map(entity_data_types_mapping)
        return entity_df

    def get_feature_views(self, feature_view_type: str = 'all', message_status: bool = True, show_query: bool = False):
        feature_view_df = get_feature_views_with_one_or_no_entity(self.config, show_query)
        feature_view_df = feature_view_df.where(pd.notnull(feature_view_df), None)
        feature_view_df.sort_values(by='Feature_view', inplace=True)

        if feature_view_type == 'single':
            if message_status:
                print('Showing feature view(s) with single entity')
            feature_view_df = feature_view_df.dropna()  # remove feature with no entity
            feature_view_df = feature_view_df.reset_index(drop=True)
            return feature_view_df

        feature_view_M = get_feature_views_with_multiple_entities(self.config, show_query)
        # group entities together for feature view with multiple entities
        """
        We need to do this because there is no direct/simple way to bind different entities to the associated feature
        view, same goes for the physical column, to ease this, the python script below handles these cases. 
        """
        update_info = []
        feature_view_dict = {}
        feature_view_to_be_processed = None
        for index, feature_view_info in feature_view_M.to_dict('index').items():
            if feature_view_to_be_processed == feature_view_info['Feature_view']:  # merge
                entity_list = feature_view_dict.get('Entity')
                entity_list.append(feature_view_info['Entity'])
                feature_view_dict['Entity'] = entity_list
                column_list = feature_view_dict.get('Physical_column')
                column_list.append(feature_view_info['Physical_column'])
                feature_view_dict['Physical_column'] = column_list
                if index == len(feature_view_M) - 1:  # last record
                    update_info.append(feature_view_dict)
            else:
                if feature_view_to_be_processed is None:  # pass for first record
                    feature_view_to_be_processed = feature_view_info['Feature_view']
                    feature_view_dict['Feature_view'] = feature_view_info['Feature_view']
                    feature_view_dict['Entity'] = [feature_view_info['Entity']]
                    feature_view_dict['Physical_column'] = [feature_view_info['Physical_column']]
                    feature_view_dict['Physical_table'] = feature_view_info['Physical_table']
                    feature_view_dict['File_source'] = feature_view_info['File_source']
                    continue
                update_info.append(feature_view_dict)
                feature_view_dict = {}
                feature_view_to_be_processed = feature_view_info['Feature_view']
                feature_view_dict['Feature_view'] = feature_view_to_be_processed
                feature_view_dict['Entity'] = [feature_view_info['Entity']]
                feature_view_dict['Physical_column'] = [feature_view_info['Physical_column']]
                feature_view_dict['Physical_table'] = feature_view_info['Physical_table']
                feature_view_dict['File_source'] = feature_view_info['File_source']

        if feature_view_type == 'multiple':
            if message_status:
                print('Showing feature view(s) with multiple entities')
            return pd.DataFrame(update_info)

        if feature_view_type == 'single and multiple':
            if message_status:
                print('Showing feature view(s) with single and multiple entities')
            feature_view_df = feature_view_df.dropna()  # remove feature with no entity

        if feature_view_type == 'all':
            if message_status:
                print('Showing all feature views')
        elif feature_view_type not in ['all', 'single', 'multiple', 'single and multiple']:
            raise ValueError("feature_view_type must be 'single', 'multiple', 'single and multiple', or 'all'")
        feature_view_df = pd.concat([feature_view_df, pd.DataFrame(update_info)], ignore_index=True)
        feature_view_df = feature_view_df.reset_index(drop=True)

        # add here
        feature_view_df['Features'] = feature_view_df['Feature_view'].apply(
            lambda x: get_features_in_feature_views(self.config, x, show_query))

        return feature_view_df

    def drop_feature_view(self, drop: list):
        self.governor.drop_feature_view(drop)
        return self.get_feature_views(message_status=False)

    def get_optional_physical_representations(self, show_query: bool = False):
        optional_physical_representations_df = get_optional_entities(self.config, show_query)
        optional_physical_representations_df['Data_type'] = optional_physical_representations_df['Data_type']. \
            map(entity_data_types_mapping)
        return optional_physical_representations_df

    def update_entity(self, entity_to_update_info: list):
        self.governor.update_entity(entity_to_update_info)
        return self.get_feature_views(message_status=False)

    def identify_features(self, entity: str, target: str, show_query: bool = False):
        feature_identification_info = identify_features(self.config, entity, target, show_query)
        feature_identification_info['Features'] = feature_identification_info.apply(lambda x:
                                                                                    get_columns(self.config,
                                                                                                table=x.Physical_table,
                                                                                                dataset=x.Dataset),
                                                                                    axis=1)

        for index, value in feature_identification_info.to_dict('index').items():
            features = []
            for feature_name in value['Features']:
                if entity not in feature_name and target not in feature_name and feature_name != 'event_timestamp':
                    features.append(feature_name)
            feature_identification_info.at[index, 'Features'] = features

        return feature_identification_info[['Entity', 'Physical_representation', 'Features', 'Feature_view',
                                            'Physical_table', 'Number_of_rows', 'File_source']]

    def search_enrichment_options(self, entity_df: pd.DataFrame = None, show_query: bool = False):
        # TODO: support for multiple entities.
        enrichable_tables = search_enrichment_options(self.config, show_query)
        # delete pairs where features are same i.e. nothing to join
        for index, pairs in tqdm(enrichable_tables.to_dict('index').items()):
            entity_dataset = pairs['Dataset']
            entity_table = pairs['Table']
            feature_view_dataset = pairs['Dataset_feature_view']
            feature_view_table = pairs['Physical_joinable_table']
            features_in_entity_df = get_columns(self.config, entity_table, entity_dataset)
            features_in_feature_view = get_columns(self.config, feature_view_table, feature_view_dataset)

            if set(features_in_feature_view).issubset(
                    set(features_in_entity_df)):  # nothing to enrich as those features already exist
                enrichable_tables = enrichable_tables.drop(index)

        enrichable_tables = enrichable_tables.sort_values(by=['Table', 'Joinability_strength', 'Enrich_with'],
                                                          ascending=False).reset_index(drop=True)

        enrichable_tables['Joinability_strength'] = enrichable_tables['Joinability_strength']. \
            apply(lambda x: str(int(x * 100)) + '%')

        if entity_df is not None:
            # filter enrichable_tables dataframe based on columns in entity_df
            if not len(search_entity_table(self.config, list(entity_df.columns))):
                print('nothing to enrich')
                return
            entity_table = search_entity_table(self.config, list(entity_df.columns))['Table'][0]
            enrichable_tables = enrichable_tables.loc[enrichable_tables['Table'] == entity_table]
            enrichable_tables.drop(['Table', 'Table_path', 'Dataset'], axis=1, inplace=True)
            # enrichable_tables.rename({'Dataset_feature_view': 'Dataset'}, axis=1, inplace=True)
            enrichable_tables = enrichable_tables[['Enrich_with', 'Physical_joinable_table', 'Join_key',
                                                   'Joinability_strength', 'File_source', 'Dataset_feature_view']]. \
                reset_index(drop=True)

        return enrichable_tables

    def get_features(self, enrichment_info: pd.Series, entity_df: pd.DataFrame = None, entity_df_columns: tuple = (),
                     show_status: bool = True):
        # TODO: add support for fetching features that originate from multiple feature views at once.
        feature_view = enrichment_info['Enrich_with']

        if len(entity_df_columns) > 0:  # process entity_df passed by the user
            entity_df_features = entity_df_columns
        else:  # process the choice passed by the user from search_enrichment_options
            entity_df_features = list(entity_df.columns)
        # features in feature view table
        feature_view_features = get_columns(self.config, enrichment_info['Physical_joinable_table'],
                                            enrichment_info['Dataset_feature_view'])
        # take difference
        features = ['{}:'.format(feature_view) + feature for feature in feature_view_features if
                    feature not in entity_df_features]
        if show_status:
            print(len(features), 'feature(s) were found!')
        return features

    def recommend_data_transformations(self, entity_df: pd.DataFrame = None, show_query: bool = False,
                                       show_insights: bool = True, n: int = None):

        def get_transformation_technique(t, f_values):
            if t == 'Ordinal encoding' and len(f_values) > 1:
                return 'OrdinalEncoder'
            elif t == 'Ordinal encoding' and len(f_values) == 1:
                return 'LabelEncoder'
            elif t == 'Scaling concerning outliers':
                return 'RobustScaler'
            elif t == 'Normalization':
                return 'MinMaxScaler'
            elif t == 'Scaling':
                return 'StandardScaler'
            elif t == 'Nominal encoding':
                return 'OneHotEncoder'
            elif t == 'Gaussian distribution':
                return 'PowerTransformer'
            elif t == 'LabelEncoder' and len(f_values) == 1:
                return 'LabelEncoder'
            elif t == 'OrdinalEncoder' and len(f_values) > 1:
                return 'OrdinalEncoder'

        # adds the transformation type mapping to the resultant recommendation dataframe
        def add_transformation_type(df):
            if df.empty:
                print('\nno recommendations found, did you clean your data?\n'
                      'try using kgfarm.recommend_cleaning_operations()')
                return

            df['Transformation_type'] = df['Transformation']
            df['Transformation'] = df.apply(lambda x: get_transformation_technique(x.Transformation, x.Feature), axis=1)

            # TODO: fix this temporary hack
            if None in list(df['Transformation']):
                df['Transformation'] = df['Transformation_type']
                for n_row, v in df.to_dict('index').items():
                    if v['Transformation'] == 'OrdinalEncoder' or v['Transformation'] == 'LabelEncoder':
                        df.loc[n_row, 'Transformation_type'] = 'Ordinal encoding'
                    elif v['Transformation'] == 'RobustEncoder':
                        df.loc[n_row, 'Transformation_type'] = 'Scaling concerning outliers'
                    elif v['Transformation'] == 'MinMaxScaler':
                        df.loc[n_row, 'Transformation_type'] = 'Normalization'
                    elif v['Transformation'] == 'StandardScaler':
                        df.loc[n_row, 'Transformation_type'] = 'Scaling'
                    elif v['Transformation'] == 'OneHotEncoding':
                        df.loc[n_row, 'Transformation_type'] = 'Nominal encoding'
                    elif v['Transformation'] == 'PowerTransformer':
                        df.loc[n_row, 'Transformation_type'] = 'Gaussian distribution'

            # post-processing to reduce false positives
            for n_row, v in df.to_dict('index').items():
                features_to_be_encoded = []
                if v['Transformation'] == 'OneHotEncoder':
                    for f in v['Feature']:
                        if len(entity_df[f].value_counts()) <= 5:
                            features_to_be_encoded.append(f)
                    if len(features_to_be_encoded) == 0:
                        df.drop(index=n_row, inplace=True)
                    else:
                        df.loc[n_row, ['Feature']] = [features_to_be_encoded]

            return df

        def handle_unseen_data(n_samples):
            if n_samples is None or n_samples > len(entity_df):
                n_samples = len(entity_df)
            return add_transformation_type(
                self.recommender.get_transformation_recommendations(entity_df.sample(n_samples),
                                                                    show_insight=show_insights))

        transformation_info = recommend_feature_transformations(self.config, show_query)

        # group features together per transformation
        transformation_info_grouped = []
        feature = []
        pipeline = None
        transformation = None
        # TODO: test the grouping script (especially for the transformation on single column)
        for row_number, value in transformation_info.to_dict('index').items():
            if transformation == value['Transformation'] and pipeline == value['Pipeline']:
                feature.append(value['Feature'])
                if row_number == len(transformation_info) - 1:  # last row
                    row = transformation_info.to_dict('index').get(row_number - 1)
                    transformation_info_grouped.append({'Transformation': transformation,
                                                        'Package': row['Package'],
                                                        'Function': row['Function'],
                                                        'Library': row['Library'],
                                                        'Feature': feature,
                                                        'Feature_view': row['Feature_view'],
                                                        'Table': row['Table'],
                                                        'Dataset': row['Dataset'],
                                                        'Author': row['Author'],
                                                        'Written_on': row['Written_on'],
                                                        'Pipeline': pipeline,
                                                        'Pipeline_url': row['Pipeline_url']})

            else:
                if row_number == 0:
                    transformation = value['Transformation']
                    pipeline = value['Pipeline']
                    feature = [value['Feature']]
                    continue
                row = transformation_info.to_dict('index').get(row_number - 1)
                transformation_info_grouped.append({'Transformation': transformation,
                                                    'Package': row['Package'],
                                                    'Function': row['Function'],
                                                    'Library': row['Library'],
                                                    'Feature': feature,
                                                    'Feature_view': row['Feature_view'],
                                                    'Table': row['Table'],
                                                    'Dataset': row['Dataset'],
                                                    'Author': row['Author'],
                                                    'Written_on': row['Written_on'],
                                                    'Pipeline': pipeline,
                                                    'Pipeline_url': row['Pipeline_url']})
                transformation = value['Transformation']
                pipeline = value['Pipeline']
                feature = [value['Feature']]

        transformation_info = pd.DataFrame(transformation_info_grouped)

        if {'Package', 'Function', 'Library', 'Author', 'Written_on', 'Pipeline_url'}.issubset(
                set(transformation_info.columns)):
            transformation_info.drop(['Package', 'Function', 'Library', 'Author', 'Written_on', 'Pipeline_url'],
                                     axis=1, inplace=True)

        if entity_df is not None:
            table_ids = self.__table_transformations.get(tuple(entity_df.columns))
            if not table_ids:
                print('Processing unseen data')
                return handle_unseen_data(n_samples=n)

            tables = list(map(lambda x: get_table_name(self.config, table_id=x), table_ids))

            # filtering transformations w.r.t entity_df
            for index, value in tqdm(transformation_info.to_dict('index').items()):
                if value['Table'] not in tables:
                    transformation_info.drop(index=index, axis=0, inplace=True)

            if len(transformation_info) < 1:
                return transformation_info

            transformation_info = transformation_info.reset_index(drop=True)
            transformation_info.drop(['Dataset', 'Dataset', 'Table'],
                                     axis=1, inplace=True)

        recommended_transformation = add_transformation_type(transformation_info)

        for index, value in recommended_transformation.to_dict('index').items():
            if value['Transformation_type'] == 'Ordinal encoding' and len(value['Feature']) > 1:
                recommended_transformation.at[index, 'Transformation'] = 'OrdinalEncoder'

        return recommended_transformation

    def apply_data_transformation(self, transformation_info: pd.Series, entity_df: pd.DataFrame = None,
                                  output_message: str = None):
        # TODO: add support for PowerTransformer
        if entity_df is not None:  # apply transformations directly on entity_df passed by user
            df = entity_df
        else:  # load the table from the choice/row passed by the user from recommend_feature_transformations()
            df = self.load_table(transformation_info, print_table_name=False)
        transformation = transformation_info['Transformation']
        features = transformation_info['Feature']
        if transformation == 'LabelEncoder':
            print('Applying LabelEncoder transformation')
            transformation_model = LabelEncoder()
            # label encoding is applied on single feature
            df[features[0]] = transformation_model.fit_transform(df[features[0]])
        elif transformation == 'StandardScaler':
            # print(
            #     'CAUTION: Make sure you apply {} transformation only on the train set (This ensures there is no over-fitting due to feature leakage)\n'.format(
            #         transformation) + \
            #     'Use the transformation_model returned from this api to transform test set independently.\n')
            print('Applying StandardScaler transformation')
            transformation_model = StandardScaler(copy=False)
            df[features] = transformation_model.fit_transform(df[features])
        elif transformation == 'OrdinalEncoder':
            print('Applying OrdinalEncoder transformation')
            transformation_model = OrdinalEncoder()
            df[features] = transformation_model.fit_transform(df[features])
        elif transformation == 'MinMaxScaler':
            print('Applying MinMaxScaler transformation')
            transformation_model = MinMaxScaler()
            df[features] = transformation_model.fit_transform(df[features])
        elif transformation == 'OneHotEncoder':
            print('Applying OneHotEncoder transformation')
            transformation_model = OneHotEncoder(handle_unknown='ignore')
            one_hot_encoded_features = pd.DataFrame(transformation_model.fit_transform(df[features]).toarray())
            df = df.join(one_hot_encoded_features)
            df = df.drop(features, axis=1)
        elif transformation == 'RobustScaler':
            print('Applying RobustScalar transformation')
            transformation_model = RobustScaler()
            df[features] = transformation_model.fit_transform(df[features])
        else:
            print(transformation, 'not supported yet!')
            return
        if output_message is None:
            print('{} feature(s) {} transformed successfully!'.format(len(features), features))
        else:
            print('{} feature(s) were transformed successfully!'.format(len(features)))
        return df, transformation_model

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

        # unary_categorical_transformations = recommendations[recommendations.loc['Transformation_type'] in {'ordinal encoding', 'nominal encoding'}]
        # unary_numerical_transformations = recommendations[recommendations.loc['Transformation_type'] == 'unary transformation']
        # scaling_transformation = recommendations[recommendations.loc['Transformation_type'] == 'scaling']

    def enrich(self, enrichment_info: pd.Series, entity_df: pd.DataFrame = None, freshness: int = 10):
        if entity_df is not None:  # entity_df passed by the user
            # get features to be enriched with
            features = self.get_features(enrichment_info=enrichment_info, entity_df_columns=tuple(entity_df.columns),
                                         show_status=False)
            features = [feature.split(':')[-1] for feature in features]
            print('Enriching {} with {} feature(s) {}'.format('entity_df', len(features), features))
        else:  # option selected from search_enrichment_options()
            entity_df = pd.read_csv(enrichment_info['Table_path'])
            features = self.get_features(enrichment_info=enrichment_info, entity_df=entity_df, show_status=False)
            features = [feature.split(':')[-1] for feature in features]
            print('Enriching {} with {} feature(s) {}'.format(enrichment_info['Table'], len(features), features))

        source_table_id = search_entity_table(self.config, entity_df.columns)['Table_id'][
            0]  # needed to track tables after enrichment
        # parse row passed as the input
        feature_view = pd.read_csv(enrichment_info['File_source'])
        join_jey = enrichment_info['Join_key']

        last_column = list(entity_df.columns)[-1]  # for re-arranging column

        # add timestamp and join-key to features in feature view to perform join
        features.extend([join_jey, 'event_timestamp'])
        feature_view = feature_view[features]
        enriched_df = pd.merge(entity_df, feature_view, on=join_jey)
        for row, row_info in tqdm(enriched_df.to_dict('index').items()):
            timestamp_entity = datetime.datetime.strptime(row_info['event_timestamp_x'], '%Y-%m-%d %H:%M:%S.%f')
            timestamp_feature_view = datetime.datetime.strptime(row_info['event_timestamp_y'], '%Y-%m-%d %H:%M:%S.%f')
            """
            delete record if the following either of the following 2 conditions were violated:
            1. Timestamp of entity < Timestamp of feature view or
            2. Timestamp of entity - freshness > timestamp of feature view 
            """
            if timestamp_entity < timestamp_feature_view or timestamp_entity - timedelta(
                    days=freshness) > timestamp_feature_view:
                enriched_df.drop(index=row, axis=0, inplace=True)

        enriched_df.drop('event_timestamp_y', axis=1, inplace=True)
        enriched_df.rename(columns={'event_timestamp_x': 'event_timestamp'}, inplace=True)

        # re-arrange columns
        columns = list(enriched_df.columns)
        columns.remove('event_timestamp')
        columns.insert(0, 'event_timestamp')
        enriched_df = enriched_df[columns]
        enriched_df = enriched_df.sort_values(by=join_jey).reset_index(drop=True)
        enriched_df = self.__re_arrange_columns(last_column, enriched_df)

        # maintain enrichment details (columns in enriched dataset : (table_ids of the joined tables))
        self.__table_transformations[tuple(enriched_df.columns)] = (source_table_id,
                                                                    get_physical_table(self.config,
                                                                                       feature_view=enrichment_info[
                                                                                           'Enrich_with']))
        return enriched_df

    def __get_features(self, entity_df: pd.DataFrame, filtered_columns: list, show_query: bool = False):
        table_id = search_entity_table(self.config, list(entity_df.columns))
        if len(table_id) < 1:
            print('Searching features for enriched dataframe\n')
            table_ids = self.__table_transformations.get(tuple(entity_df.columns))
            return [feature for feature in list(entity_df.columns) if feature not in
                    get_features_to_drop(self.config, table_ids[0], show_query)[
                        'Feature_to_drop'].tolist() and feature not in
                    get_features_to_drop(self.config, table_ids[1], show_query)[
                        'Feature_to_drop'].tolist() and feature in
                    filtered_columns]

        else:
            table_id = table_id['Table_id'][0]
            return [feature for feature in list(entity_df.columns)
                    if
                    feature not in get_features_to_drop(self.config, table_id, show_query)['Feature_to_drop'].tolist()
                    and feature in filtered_columns]

    def select_features(self, entity_df: pd.DataFrame, dependent_variable: str, select_by: str = None,
                        plot_correlation: bool = True,
                        plot_anova_test: bool = True, show_f_value: bool = False):

        def handle_unseen_data():
            raise NotImplementedError('under construction')

        def handle_data_by_statistics(dependent_var, f_score):
            def get_input():
                return int(input(f'Enter k (where k is the top-k ranked features out of {len(df.columns)} feature(s) '))

            if select_by not in {'anova', 'correlation'}:
                raise ValueError("select_by can either be 'anova' or 'correlation'")
            if select_by == 'anova':
                f_score = f_score.head(get_input())
                independent_var = df[f_score['Feature']]  # features (X)
                print('Top {} feature(s) {} were selected based on highest F-value'.
                      format(len(independent_var.columns), list(independent_var.columns)))
                return independent_var, dependent_var
            elif select_by == 'correlation':
                correlation = df.corr()
                correlation.drop(index=dependent_variable, inplace=True)
                columns = list(correlation.columns)
                columns = [f for f in columns if f != dependent_variable]
                correlation.drop(columns, axis=1, inplace=True)
                correlation.sort_values(by=dependent_variable, ascending=False, inplace=True)
                top_k_features = correlation.head(get_input())
                top_k_features = list(top_k_features.to_dict().get(dependent_variable))
                independent_var = df[top_k_features]  # features (X)
                if self.mode != 'Automatic':
                    print('Top {} feature(s) {} were selected based on highest Correlation'.
                          format(len(independent_var.columns), list(independent_var.columns)))
                return independent_var, dependent_var

        df = copy.copy(entity_df)
        for feature in tqdm(list(entity_df.columns)):  # drop entity column and timestamp
            if is_entity_column(self.config, feature=feature, dependent_variable=dependent_variable) \
                    or feature == 'event_timestamp':
                df.drop(feature, axis=1, inplace=True)

        print('Analyzing features')
        if plot_correlation:  # plot pearson correlation
            plt.rcParams['figure.dpi'] = 300
            corr = df.corr(method='pearson')
            plt.figure(figsize=(15, 10))
            sns.heatmap(corr, annot=True, cmap='Greens')
            plt.show()

        # calculate F-value for features
        y = entity_df[dependent_variable]  # dependent variable
        X = df.drop(dependent_variable, axis=1)  # independent variables
        best_features = SelectKBest(score_func=f_classif, k=5).fit(X, y)
        scores = pd.DataFrame(best_features.scores_)
        features = pd.DataFrame(X.columns)
        feature_scores = pd.concat([scores, features], axis=1)
        feature_scores.columns = ['F_value', 'Feature']
        feature_scores['F_value'] = feature_scores['F_value'].apply(lambda x: round(x, 2))
        feature_scores = feature_scores.sort_values(by='F_value', ascending=False).reset_index(drop=True)

        if plot_anova_test:  # plot ANOVA test graph
            plt.rcParams['figure.dpi'] = 300
            plt.figure(figsize=(15, 10))
            sns.set_color_codes('pastel')
            sns.barplot(x='F_value', y='Feature', data=feature_scores,
                        label='Total', palette="Greens_r", edgecolor='none')
            sns.set_color_codes('muted')
            sns.despine(left=True, bottom=True)
            plt.xlabel('F value', fontsize=15)
            plt.ylabel('Feature', fontsize=15)
            plt.grid(color='lightgray', axis='y')
            plt.tick_params(axis='both', which='major', labelsize=15)
            plt.show()

        if show_f_value:
            print(feature_scores, '\n')

        table_id = search_entity_table(self.config, list(entity_df.columns))
        if len(table_id) < 1:  # i.e. table not profiled
            table_ids = self.__table_transformations.get(tuple(entity_df.columns))
            if table_ids is None:
                # return handle_data_by_statistics(dependent_var=y, f_score=feature_scores)
                if select_by in {'anova', 'correlation'}:
                    return handle_data_by_statistics(dependent_var=y, f_score=feature_scores)
                else:
                    print('processing unseen data')
                    handle_unseen_data()

        if select_by is None:  # select by pipelines
            X = entity_df[self.__get_features(entity_df=entity_df, filtered_columns=list(df.columns))]
            print('{} feature(s) {} were selected based on previously abstracted pipelines'.format(len(X.columns),
                                                                                                   list(X.columns)))
            return X, y

    @staticmethod
    def get_columns_to_be_cleaned(df: pd.DataFrame):
        for na_type in {'none', 'n/a', 'na', 'nan', 'missing', '?', '', ' '}:
            if na_type in {'?', '', ' '}:
                df.replace(na_type, np.nan, inplace=True)
            else:
                df.replace(r'(^)' + na_type + r'($)', np.nan, inplace=True, regex=True)

        columns = pd.DataFrame(df.isnull().sum())
        columns.columns = ['Missing values']
        columns['Feature'] = columns.index
        columns = columns[columns['Missing values'] > 0]
        columns.sort_values(by='Missing values', ascending=False, inplace=True)
        columns.reset_index(drop=True, inplace=True)
        return columns

    def recommend_cleaning_operations(self, entity_df: pd.DataFrame, visualize_missing_data: bool = False,
                                      top_k: int = 10,
                                      show_query: bool = False):
        """
        1. visualize missing data
        2. check if data is profiled or unseen
        3. if unseen align and query else query directly
        """

        def plot_heat_map(df: pd.DataFrame):
            plt.rcParams['figure.dpi'] = 300
            plt.figure(figsize=(15, 7))
            sns.heatmap(df.isnull(), yticklabels=False, cmap='Greens_r')
            plt.show()

        def plot_bar_graph(columns: pd.DataFrame):
            if len(columns) == 0:
                return
            sns.set_color_codes('pastel')
            plt.rcParams['figure.dpi'] = 300
            plt.figure(figsize=(6, 3))

            ax = sns.barplot(x="Feature", y="Missing values", data=columns,
                             palette='Greens_r', edgecolor='gainsboro')
            ax.bar_label(ax.containers[0], fontsize=6)

            def change_width(axis, new_value):
                for patch in axis.patches:
                    current_width = patch.get_width()
                    diff = current_width - new_value
                    patch.set_width(new_value)
                    patch.set_x(patch.get_x() + diff * .5)

            change_width(ax, .20)
            plt.grid(color='lightgray', axis='y')
            plt.ylabel('Missing value', fontsize=5.5)
            plt.xlabel('')
            ax.tick_params(axis='both', which='major', labelsize=5.5)
            ax.tick_params(axis='x', labelrotation=90, labelsize=5.5)
            plt.show()

        print('scanning missing values')
        columns_to_be_cleaned = self.get_columns_to_be_cleaned(entity_df)
        if visualize_missing_data:
            plot_heat_map(df=entity_df)
            plot_bar_graph(columns=columns_to_be_cleaned)

        if len(columns_to_be_cleaned) == 0:
            print('nothing to clean')
            return entity_df
        
        similar_tables = self.recommender.get_cleaning_recommendation(entity_df[columns_to_be_cleaned['Feature']])
        print('Suggestions are:', similar_tables)
        print(similar_tables.columns)
        return similar_tables

    def clean(self, entity_df: pd.DataFrame, technique: pd.DataFrame = None):
        """
        cleans entity_df from info coming from kgfarm.recommend_cleaning_operations
        """

        def check_for_uncleaned_features(df: pd.DataFrame):  # clean by recommendations
            uncleaned_features = list(self.get_columns_to_be_cleaned(df=df)['Feature'])
            if len(uncleaned_features) == 0:
                print('\nall features look clean')
            else:
                print(f'\n{uncleaned_features} are still uncleaned')
                return 1

        def apply_operation(technique, entity_df):
            if technique is not None:  # clean by human-in-the-loop
                columns = list(self.get_columns_to_be_cleaned(df=entity_df)['Feature'])
                #Subset_df is columns in the original df that need to be cleaned
                subset_df = entity_df[columns]
                print('technique', technique, type(technique))
                if technique == 0:
                    print('The data is clean')
                    return entity_df
                elif technique == 'drop':
                    entity_df.dropna(how='any', inplace=True)
                    entity_df.reset_index(drop=True, inplace=True)
                    print(f'missing values from {columns} were dropped')
                    check_for_uncleaned_features(df=entity_df)
                    df_imputed = entity_df.copy()

                elif technique.__contains__('fill'):
                    def fillna_median(column):
                        median = column.median()
                        return column.fillna(median, inplace=True)
                    def fillna_mean(column):
                        mean = column.mean()
                        return column.fillna(mean, inplace=True)
                    def fillna_mode(column):
                        mode = column.mode()
                        return column.fillna(mode, inplace=True)

                    if technique == 'fill-median':
                        subset_df.apply(fillna_median)
                    elif technique == 'fill-mean':
                        subset_df.apply(fillna_mean)
                    elif technique == 'fill-mode':
                        subset_df.apply(fillna_mode)
                    elif technique == 'fill-outlier':
                        object_cols = subset_df.select_dtypes(include=['object']).columns
                        subset_df[object_cols] = subset_df[object_cols].fillna(value='x')
                        subset_df = subset_df.fillna(value=0)
                    print('resetting')
                    subset_df.reset_index(drop=True, inplace=True)
                    print('check for unclean')
                    check_for_uncleaned_features(df=subset_df)
                    df_imputed = subset_df.copy()
                elif technique == 'interpolate':
                    try:
                        subset_df.interpolate(axis=1, inplace=True)
                    except TypeError:
                        print('only numerical features can be interpolated')
                    check = check_for_uncleaned_features(df=subset_df)
                    count = 0

                    while check == 1 and count<10:
                        print(subset_df.isnull())
                        print('reinterpolating...')
                        count = count + 1
                        if count%2 ==0:
                            print('count',count)
                            subset_df.interpolate(inplace=True)
                            check = check_for_uncleaned_features(df=subset_df)
                        else:
                            print('count',count)
                            subset_df.interpolate(limit_direction='backward', inplace=True)
                            check = check_for_uncleaned_features(df=subset_df)
                    df_imputed = subset_df.copy()

                elif technique.__contains__('SimpleImputer'):
                    try:
                        if technique == 'SimpleImputer-median':
                            imputer = SimpleImputer(strategy='median')
                            imputer.fit(subset_df)
                            df_imputed = imputer.transform(subset_df)
                        elif technique == 'SimpleImputer-most_frequent':
                            imputer = SimpleImputer(strategy='most_frequent')
                            imputer.fit(subset_df)
                            df_imputed = imputer.transform(subset_df)
                        elif technique == 'SimpleImputer-constant':
                            imputer = SimpleImputer(strategy='constant')
                            imputer.fit(subset_df)
                            df_imputed = imputer.transform(subset_df)
                        else:
                            imputer = SimpleImputer(strategy='mean')
                            imputer.fit(subset_df)
                            df_imputed = imputer.transform(subset_df)
                    except TypeError:
                        print('only numerical features can be interpolated')
                    check_for_uncleaned_features(df=subset_df)
                elif technique == 'IterativeImputer':
                    imputer = IterativeImputer(BayesianRidge())
                    imputer.fit(subset_df)
                    df_imputed = imputer.transform(subset_df)
                    check_for_uncleaned_features(df=X_imputed)
                    #return X_imputed
                elif technique == 'KNNImputer':
                    try:
                        #Divide the columns and values
                        col = subset_df.columns
                        imputer = KNNImputer()
                        subset_df = imputer.fit_transform(subset_df)
                        df_imputed = pd.DataFrame(subset_df, columns=col)
                        check_for_uncleaned_features(df=df_imputed)
                    except TypeError:
                        print('only numerical features can be interpolated')
                else:
                    if technique not in {'drop', 'fill', 'interpolate', 'imputer'}:
                        raise ValueError("technique must be one out of 'drop', 'fill', 'interpolate, 'imputer''")
                # create a new dataframe with the imputed values
                df_imputed = pd.DataFrame(df_imputed, columns=columns)

                # merge the imputed values back into the original dataframe
                entity_df[columns] = df_imputed
                return entity_df

        entity_df = apply_operation(technique, entity_df)#.index[0]
        return entity_df



    """
    def recommend_features_to_be_selected(self, entity_df: pd.DataFrame, dependent_variable: str, k: int):
        recommended_features = list(self.recommender.get_feature_selection_score(entity_df=entity_df,
                                                                                 dependent_variable=dependent_variable).head(
            k)['Feature'])
        # print(f'Recommending top-{k} feature(s) {recommended_features}')
        return entity_df[recommended_features], entity_df[dependent_variable]  # return X, y
    """

    def recommend_features_to_be_selected(self, task: str, entity_df: pd.DataFrame, dependent_variable: str,
                                          n: int = None):
        if n is None or len(entity_df) < n:
            n = len(entity_df)

        return self.recommender.get_feature_selection_score(task=task, entity_df=entity_df.sample(n=n, random_state=1),
                                                            dependent_variable=dependent_variable)

    """
    def select_features_distributed(self, features: pd.DataFrame, target: pd.Series, n: int = None):
        if n is not None:  # subsample n data points
            features['target'] = target
            features = features.sample(n=n, random_state=1)
            # target = features['target']
            # features.drop('target', axis=1, inplace=True)
            # return features
        entity_df = self.spark.createDataFrame(features)
        return self.recommender.get_feature_selection_score_distributed(entity_df=entity_df)
    """

    def engineer_features(self):
        pass


entity_data_types_mapping = {'N_int': 'integer', 'N_float': 'float', 'N_bool': 'boolean',
                             'T': 'string', 'T_date': 'timestamp', 'T_loc': 'string (location)',
                             'T_person': 'string (person)',
                             'T_org': 'string', 'T_code': 'string (code)', 'T_email': 'string (email)'}

data_cleaning_operation_mapping = {'pandas.DataFrame.fillna': 'Fill missing values',
                                   'pandas.DataFrame.interpolate': 'Interpolate',
                                   'pandas.DataFrame.dropna': 'Drop missing values'}

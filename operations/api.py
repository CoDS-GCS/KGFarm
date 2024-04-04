import os
import warnings
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import *
from operations.template import *
from .Modeling.prepare_for_encoding import profile_to_csv, create_encoding_file
from .Modeling.encoding import encode
from .Modeling.embeddings_from_profile import *
from .Modeling.create_cleaning_model import graphSaint as graphSaint_modeling_cleaning
from .Modeling.create_scaling_model import graphSaint as graphSaint_modeling_scaler
from .Modeling.create_unary_model import graphSaint as graphSaint_modeling_unary
from .kglids import create_triplets
from .script_transform_biokg_to_ogb_datasets import triplet_encoding
from .inference_cleaning import graphSaint as graphSaint_cleaning
from .inference_scaling import graphSaint as graphSaint_scaling
from .inference_unary import graphSaint as graphSaint_unary
from .apply_recommendation import apply_cleaning
from helpers import helper

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


class KGFarm:
    def __init__(self, endpoint: str = 'http://localhost:7200', db: str = 'earthquake'):
        self.config = helper.connect_to_graphdb(endpoint, db)

    def build_cleaning_model(self, graph_name: str):
        create_encoding_file(graph_name, 'cleaning')
        encode(graph_name, "http://kglids.org/ontology/pipeline/HasCleaningOperation", "2Table")
        get_table_embeddings_cleaning(graph_name)
        get_column_embeddings(graph_name)
        graphSaint_modeling_cleaning(graph_name)

    def build_scaler_model(self, graph_name: str):
        # profile_to_csv(graph_name)
        # create_encoding_file(graph_name, 'scaler_transformation')
        # encode(graph_name, "http://kglids.org/ontology/pipeline/HasScalingTransformation", "2Table")
        get_table_embeddings_scaling(graph_name)
        get_column_embeddings(graph_name)
        graphSaint_modeling_scaler(graph_name)

    def build_unary_model(self, graph_name: str):
        # profile_to_csv(graph_name)
        # create_encoding_file(graph_name, 'unary_transformation')
        # encode(graph_name, "http://kglids.org/ontology/pipeline/HasUnaryTransformation", "1Column")
        get_column_embeddings_unary(graph_name)
        graphSaint_modeling_unary(graph_name)

    def recommend_cleaning_operations(self, table: pd.DataFrame, name: str = 'Cleaning_dataset'):
        cleaning_dict = {0: 'Fill', 1: 'Interpolate', 2: 'IterativeImputer', 3: 'KNNImputer', 4: 'SimpleImputer'}
        create_triplets(table, name)
        triplet_encoding(name, 'Table')
        cleaning_op = graphSaint_cleaning(table, name)
        recommended_op = pd.DataFrame(index=cleaning_op, columns=['Operation'])
        recommended_op['Operation'] = recommended_op.index.map(cleaning_dict.get)
        data = {'Cleaning Operation': cleaning_op[:3],
                'Feature': [[table.columns[table.isna().any()].tolist()],[table.columns[table.isna().any()].tolist()],[table.columns[table.isna().any()].tolist()]]}
        recommended_op = pd.DataFrame(data)
        recommended_op['Cleaning Operation'] = recommended_op[
            'Cleaning Operation'].replace(cleaning_dict)
        return recommended_op

    def apply_cleaning_operations(self, operation: pd.Series, df: pd.DataFrame):
        clean_df = apply_cleaning(df, operation['Cleaning Operation'])
        return clean_df

    def recommend_transformation_operations(self, table: pd.DataFrame, name: str = 'Transformation_dataset'):
        scaling_dict = {0: 'MinMaxScaler', 1: 'RobustScaler', 2: 'StandardScaler'}
        unary_dict = {1: 'Log', 2: 'Sqrt', 0: 'NoUnary'}
        create_triplets(table, name)
        triplet_encoding(name, 'Table')
        scaling_op = graphSaint_scaling(name, table)
        triplet_encoding(name, 'Column')
        unary_op = graphSaint_unary(name, table)
        data = {'Recommended_transformation': scaling_op,
                'Recommendation': ['rec1','rec2','rec3'],
                'Feature': ['All','All','All']}
        recommended_scaling_transformations = pd.DataFrame(data)
        recommended_scaling_transformations['Recommended_transformation'] = recommended_scaling_transformations[
            'Recommended_transformation'].replace(scaling_dict)

        df_unary_col = pd.read_csv(
            'operations/storage/' + name + '_gnn_Column/mapping/Column_entidx2name.csv')

        unary_op_labels = np.array([[unary_dict[op] for op in row] for row in unary_op])
        new_df = pd.concat([pd.DataFrame(unary_op_labels, columns=['rec1', 'rec2', 'rec3']), df_unary_col], axis=1)
        grouped_entities_col1 = new_df.groupby(['rec1']).apply(lambda x: x['ent name'].tolist())
        grouped_entities_col2 = new_df.groupby(['rec2']).apply(lambda x: x['ent name'].tolist())
        grouped_entities_col3 = new_df.groupby(['rec3']).apply(lambda x: x['ent name'].tolist())
        recommended_unary_transformations = pd.DataFrame({'Recommended_transformation': grouped_entities_col1.index,
                                 'Recommendation': 'rec1',
                                 'Feature': grouped_entities_col1.values})

        recommended_unary_transformations = pd.concat([recommended_unary_transformations, pd.DataFrame({'Recommended_transformation': grouped_entities_col2.index,
                                                      'Recommendation': 'rec2',
                                                      'Feature': grouped_entities_col2.values})], ignore_index=True)

        recommended_unary_transformations = pd.concat([recommended_unary_transformations, pd.DataFrame({'Recommended_transformation': grouped_entities_col3.index,
                                                      'Recommendation': 'rec3',
                                                      'Feature': grouped_entities_col3.values})], ignore_index=True)
        recommended_transformations = pd.concat(
            [recommended_scaling_transformations, recommended_unary_transformations], ignore_index=True)
        sorting_order = {"rec1": 1, "rec2": 2, "rec3": 3}
        recommended_transformations["Sort_Order"] = recommended_transformations["Recommendation"].map(sorting_order)
        recommended_transformations = recommended_transformations.sort_values(by="Sort_Order")
        recommended_transformations = recommended_transformations.drop(columns=["Sort_Order"])
        return recommended_transformations.reset_index(drop=True)

    def apply_transformation_operations(self, X: pd.DataFrame, recommended_transformations: pd.DataFrame,
                                        label_name: str = 'None'):
        for n, recommendation in recommended_transformations.to_dict('index').items():
            if recommendation['Recommended_transformation'] != 'NoUnary':
                transformation = recommendation['Recommended_transformation']
                feature = recommendation['Feature']

                if transformation in {'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'QuantileTransformer',
                                      'PowerTransformer'}:
                    # if label_name != 'None': # If we want exclude the label from being transformed
                    #     X = X.drop(columns=[label_name])

                    print(f'Applying {transformation}')  # on {list(X.columns)}')
                    if transformation == 'StandardScaler':
                        scaler = StandardScaler()
                    elif transformation == 'MinMaxScaler':
                        scaler = MinMaxScaler()
                    elif transformation == 'RobustScaler':
                        scaler = RobustScaler()
                    else:
                        scaler = RobustScaler()
                    numerical_columns = X.select_dtypes(include=['int', 'float']).columns
                    numerical_columns_wo_label = [col for col in numerical_columns if col != label_name]
                    X[numerical_columns_wo_label] = scaler.fit_transform(X[numerical_columns_wo_label])

                elif transformation in {'Log', 'Sqrt', 'square'}:
                    print(f'Applying {transformation}')  # on {list(feature)}')
                    if transformation == 'Log':
                        def log_plus_const(x, const=0):
                            return np.log(x + np.abs(const) + 0.0001)

                        for f in tqdm(feature):
                            if str(f) != label_name and pd.api.types.is_numeric_dtype(X[str(f)].dtype):
                                min_neg_val = X[str(f)].min()
                                unary_transformation_model = FunctionTransformer(func=log_plus_const,
                                                                                 kw_args={'const': min_neg_val},
                                                                                 validate=True)

                                X[str(f)] = unary_transformation_model.fit_transform(
                                    X=np.array(X[str(f)]).reshape(-1, 1))

                    elif transformation == 'Sqrt':
                        def sqrt_plus_const(x, const=0):
                            return np.sqrt(x + np.abs(const) + 0.0001)

                        for f in tqdm(feature):
                            if str(f) != label_name and pd.api.types.is_numeric_dtype(X[str(f)].dtype):
                                min_neg_val = X[str(f)].min()
                                unary_transformation_model = FunctionTransformer(func=sqrt_plus_const,
                                                                                 kw_args={'const': min_neg_val},
                                                                                 validate=True)
                                X[str(f)] = unary_transformation_model.fit_transform(
                                    X=np.array(X[str(f)]).reshape(-1, 1))
                    else:
                        unary_transformation_model = FunctionTransformer(func=np.square, validate=True)
                        X[feature] = unary_transformation_model.fit_transform(X=X[feature])
                else:
                    raise ValueError(f'{transformation} not supported')

        return X
    # def __init__(self, mode: str = 'Human in the loop', port: object = 5820, database: str = 'kgfarm_test',
    #              show_connection_status: bool = True):
    #     sys.path.insert(0, str(Path(os.getcwd()).parent.absolute()))
    #     self.mode = mode
    #     if mode not in ['Human in the loop', 'Automatic']:
    #         raise ValueError("mode can be either 'Human in the Loop' or 'Automatic'")
    #     print('(KGFarm is running in {} mode)'.format(mode))
    #     self.config = connect_to_stardog(port, database, show_connection_status)
    #     if mode == 'Human in the loop':
    #         self.recommender = Recommender()
    #         self.recommender_config = connect_to_stardog(port, db='kgfarm_recommender', show_status=False)
    #     self.governor = Governor(self.config)
    #     self.__table_transformations = {}  # cols in enriched_df: tuple -> (entity_df_id, feature_view_id)
    #     """
    #     conf = SparkConf().setAppName('KGFarm')
    #     conf = (conf.setMaster('local[*]')
    #             .set('spark.executor.memory', '10g')
    #             .set('spark.driver.memory', '5g')
    #             .set('spark.driver.maxResultSize', '5g'))
    #     sc = SparkContext(conf=conf)
    #     self.spark = SparkSession(sc)
    #     """

    # re-arranging columns
    @staticmethod
    def __re_arrange_columns(last_column: str, df: pd.DataFrame):
        features = list(df.columns)
        features.remove(last_column)
        features.append(last_column)
        df = df[features]
        return df

    # def __check_if_profiled(self, df: pd.DataFrame):
    #     table_id = search_entity_table(self.config, list(df.columns))
    #     if len(table_id) == 0:
    #         # search for enriched tables
    #         table_ids = self.__table_transformations.get(tuple(df.columns))
    #         if table_ids is None:
    #             return False  # unseen table
    #         else:
    #             return table_ids  # enriched table (return table urls which make the enriched table)
    #
    #     else:
    #         return table_id['Table_id'][0]  # seen / profiled table
    #
    # # wrapper around pd.read_csv()
    # def load_table(self, table_info: pd.Series, print_table_name: bool = True):
    #     table = table_info['Table']
    #     dataset = table_info['Dataset']
    #     if print_table_name:
    #         print(table)
    #     return pd.read_csv(get_table_path(self.config, table, dataset))
    #
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

    # def drop_feature_view(self, drop: list):
    #     self.governor.drop_feature_view(drop)
    #     return self.get_feature_views(message_status=False)
    #
    # def get_optional_physical_representations(self, show_query: bool = False):
    #     optional_physical_representations_df = get_optional_entities(self.config, show_query)
    #     optional_physical_representations_df['Data_type'] = optional_physical_representations_df['Data_type']. \
    #         map(entity_data_types_mapping)
    #     return optional_physical_representations_df
    #
    # def update_entity(self, entity_to_update_info: list):
    #     self.governor.update_entity(entity_to_update_info)
    #     return self.get_feature_views(message_status=False)

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
        try:
            file_path = enrichment_info['File_source']
            feature_view = pd.read_csv(file_path)
        except:
            file_path = enrichment_info['File_source'].replace('\\', '/')
            feature_view = pd.read_csv(file_path)

        join_jey = enrichment_info['Join_key']

        last_column = list(entity_df.columns)[-1]  # for re-arranging column

        features.extend([join_jey])
        feature_view = feature_view[features]
        enriched_df = pd.merge(entity_df, feature_view, on=join_jey)

        # re-arrange columns
        columns = list(enriched_df.columns)
        enriched_df = enriched_df[columns]
        enriched_df = enriched_df.sort_values(by=join_jey).reset_index(drop=True)
        enriched_df = self.__re_arrange_columns(last_column, enriched_df)

        return enriched_df

entity_data_types_mapping = {'N_int': 'integer', 'N_float': 'float', 'N_bool': 'boolean',
                             'T': 'string', 'T_date': 'timestamp', 'T_loc': 'string (location)',
                             'T_person': 'string (person)',
                             'T_org': 'string', 'T_code': 'string (code)', 'T_email': 'string (email)'}

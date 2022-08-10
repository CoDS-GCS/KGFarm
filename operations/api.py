import sklearn
import datetime
import pandas as pd
from tqdm import tqdm
from datetime import timedelta
from operations.template import *
from sklearn.preprocessing import *
from feature_discovery.src.graph_builder.governor import Governor
from helpers.helper import connect_to_stardog
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


class KGFarm:
    def __init__(self, port: object = 5820, database: str = 'kgfarm_test', show_connection_status: bool = True):
        self.config = connect_to_stardog(port, database, show_connection_status)
        self.governor = Governor(self.config)
        # TODO: add info from self.__table_transformations to graph via Governor
        self.__table_transformations = {}  # for enrichment -> enriched with : source table

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

    def get_feature_views(self, show_query: bool = False):
        feature_view_df = get_feature_views_with_one_or_no_entity(self.config, show_query)
        feature_view_M = get_feature_views_with_multiple_entities(self.config, show_query)

        # group entities together for feature view with multiple entities
        """
        We need to do this because there is no direct/simple way to bind different entities to the associated feature
        view, same goes for the physical column, to ease this, the python script below handles these cases. 
        """
        duplicates = feature_view_M.groupby(feature_view_M['Feature_view'].tolist(), as_index=False).size()
        duplicates['size'] = duplicates['size'].apply(lambda x: True if x > 1 else False)
        duplicates = set(duplicates['size'])
        if True in duplicates:
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

        else:  # no feature view with multiple entity
            update_info = feature_view_M
        feature_view_df = pd.concat([feature_view_df, pd.DataFrame(update_info)], ignore_index=True)
        feature_view_df = feature_view_df.where(pd.notnull(feature_view_df), None)
        feature_view_df.sort_values(by='Feature_view', inplace=True)
        feature_view_df = feature_view_df.reset_index(drop=True)
        return feature_view_df

    def drop_feature_view(self, drop: list):
        self.governor.drop_feature_view(drop)
        return self.get_feature_views()

    def get_optional_physical_representations(self, show_query: bool = False):
        optional_physical_representations_df = get_optional_entities(self.config, show_query)
        optional_physical_representations_df['Data_type'] = optional_physical_representations_df['Data_type'].\
            map(entity_data_types_mapping)
        return optional_physical_representations_df

    def update_entity(self, entity_to_update_info: list):
        self.governor.update_entity(entity_to_update_info)
        return self.get_feature_views()

    def search_enrichment_options(self, entity_df: pd.DataFrame = None, show_query: bool = False):
        # TODO: investigate why some recommendations here have no feature/column to join
        # TODO: support for multiple entities.
        enrichable_tables = get_enrichable_tables(self.config, show_query)
        # delete pairs where features are same i.e. nothing to join
        for index, pairs in tqdm(enrichable_tables.to_dict('index').items()):
            entity_dataset = pairs['Dataset']
            entity_table = pairs['Table']
            feature_view_dataset = pairs['Dataset_feature_view']
            feature_view_table = pairs['Physical_joinable_table']
            features_in_entity_df = get_columns(self.config, entity_table, entity_dataset)
            features_in_feature_view = get_columns(self.config, feature_view_table, feature_view_dataset)
            # sort features
            features_in_entity_df.sort()
            features_in_feature_view.sort()
            if features_in_entity_df == features_in_feature_view:
                enrichable_tables = enrichable_tables.drop(index)

        enrichable_tables = enrichable_tables.sort_values(by=['Table', 'Joinability_strength', 'Enrich_with'],
                                                          ascending=False).reset_index(drop=True)

        enrichable_tables['Joinability_strength'] = enrichable_tables['Joinability_strength']. \
            apply(lambda x: str(int(x * 100)) + '%')

        if entity_df is not None:
            # filter enrichable_tables dataframe based on columns in entity_df
            entity_table = search_entity_table(self.config, list(entity_df.columns))
            enrichable_tables = enrichable_tables.loc[enrichable_tables['Table'] == entity_table]
            enrichable_tables.drop(['Table', 'Table_path', 'Dataset'], axis=1, inplace=True)
            # enrichable_tables.rename({'Dataset_feature_view': 'Dataset'}, axis=1, inplace=True)
            enrichable_tables = enrichable_tables[['Enrich_with', 'Physical_joinable_table', 'Join_key',
                                                   'Joinability_strength', 'File_source',	'Dataset_feature_view']].\
                reset_index(drop=True)

        return enrichable_tables

    def get_features(self, enrichment_info: pd.Series, entity_df: pd.DataFrame = None, entity_df_columns: tuple = (), show_status: bool = True):
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

    def recommend_feature_transformations(self, entity_df: pd.DataFrame = None, show_metadata: bool = True,
                                          show_query: bool = False):
        transformation_info = recommend_feature_transformations(self.config, show_query)

        # group features together per transformation
        transformation_info_grouped = []
        feature = []
        pipeline = None
        transformation = None
        for row_number, value in transformation_info.to_dict('index').items():
            if transformation == value['Transformation'] and pipeline == value['Pipeline']:
                feature.append(value['Feature'])
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
        if not show_metadata:
            transformation_info.drop(['Package', 'Function', 'Library', 'Author', 'Written_on', 'Pipeline_url'],
                                     axis=1, inplace=True)

        if entity_df is not None:
            table = self.__table_transformations.get(tuple(entity_df.columns))
            transformation_info = transformation_info.loc[transformation_info['Table'] == table]
            transformation_info = transformation_info.reset_index(drop=True)

            transformation_info.drop(['Dataset', 'Written_on', 'Pipeline_url', 'Dataset', 'Author', 'Table'],
                                 axis=1, inplace=True)
        return transformation_info

    def apply_transformation(self, transformation_info: pd.Series, entity_df: pd.DataFrame = None):
        # TODO: add support for other transformations as well (ex. one-hot encoding, min-max scaling, etc.)
        if entity_df is not None:  # apply transformations directly on entity_df passed by user
            df = entity_df
        else:  # load the table from the choice/row passed by the user from recommend_feature_transformations()
            df = self.load_table(transformation_info, print_table_name=False)
        transformation = transformation_info['Transformation']
        features = transformation_info['Feature']
        if transformation == 'LabelEncoder':
            print('Applying LabelEncoder transformation')
            for feature in tqdm(features):
                try:
                    encoder = LabelEncoder()
                    df[feature] = encoder.fit_transform(df[feature])
                except sklearn.exceptions:
                    print("{} couldn't be transformed".format(transformation_info['Table']))
        elif transformation == 'StandardScaler':
            print('Applying StandardScaler transformation')
            try:
                scaler = StandardScaler(copy=False)
                df[features] = scaler.fit_transform(df[features])
            except sklearn.exceptions:
                print("{} couldn't be transformed".format(transformation_info['Table']))
        else:
            print(transformation, ' not supported yet!')
            return
        print('{} feature(s) {} transformed successfully!'.format(len(features), features))
        return df

    def enrich(self, enrichment_info: pd.Series, entity_df: pd.DataFrame = None, ttl: int = 10):
        if entity_df is not None:  # entity_df passed by the user
            # get features to be enriched with
            features = self.get_features(enrichment_info=enrichment_info, entity_df_columns=tuple(entity_df.columns), show_status=False)
            features = [feature.split(':')[-1] for feature in features]
            print('Enriching {} with {} feature(s) {}'.format('entity_df', len(features), features))
        else:  # option selected from search_enrichment_options()
            entity_df = pd.read_csv(enrichment_info['Table_path'])
            features = self.get_features(enrichment_info=enrichment_info, entity_df=entity_df, show_status=False)
            features = [feature.split(':')[-1] for feature in features]
            print('Enriching {} with {} feature(s) {}'.format(enrichment_info['Table'], len(features), features))

        # parse row passed as the input
        feature_view = pd.read_csv(enrichment_info['File_source'])
        join_jey = enrichment_info['Join_key']

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
            2. Timestamp of entity - ttl > timestamp of feature view 
            """
            if timestamp_entity < timestamp_feature_view or timestamp_entity - timedelta(
                    days=ttl) > timestamp_feature_view:
                enriched_df.drop(index=row, axis=0, inplace=True)

        enriched_df.drop('event_timestamp_y', axis=1, inplace=True)
        enriched_df.rename(columns={'event_timestamp_x': 'event_timestamp'}, inplace=True)

        # re-arrange columns
        columns = list(enriched_df.columns)
        columns.remove('event_timestamp')
        columns.insert(0, 'event_timestamp')
        enriched_df = enriched_df[columns]
        enriched_df = enriched_df.sort_values(by=join_jey).reset_index(drop=True)

        # maintain enrichment details
        self.__table_transformations[tuple(enriched_df.columns)] = enrichment_info['Physical_joinable_table']

        return enriched_df


entity_data_types_mapping = {'N_int': 'numeric (integer)', 'N_float': 'numeric (float)', 'N_bool': 'numeric (boolean)',
        'T': 'string (generic)', 'T_date': 'timestamp', 'T_loc': 'string (location)', 'T_person': 'string (person)',
        'T_org': 'string (organization)', 'T_code': 'string (code)', 'T_email': 'string (email)'}

if __name__ == "__main__":
    kgfarm = KGFarm(port=5820, database='kgfarm_test', show_connection_status=False)

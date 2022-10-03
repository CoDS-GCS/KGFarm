import copy
import os
import sklearn
import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from datetime import timedelta
from sklearn.preprocessing import *
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from feature_discovery.src.graph_builder.governor import Governor
from helpers.helper import connect_to_stardog
from operations.template import *
from operations.recommendation.recommender import Recommender
from operations.recommendation.utils.transformation_mappings import transformation_mapping
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
        self.governor = Governor(self.config)
        self.__table_transformations = {}  # cols in enriched_df: tuple -> (entity_df_id, feature_view_id)
        self.recommender = Recommender()

    # re-arranging columns
    @staticmethod
    def __re_arrange_columns(last_column: str, df: pd.DataFrame):
        features = list(df.columns)
        features.remove(last_column)
        features.append(last_column)
        df = df[features]
        return df

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

    def search_entity(self, entity_name: str, show_query: bool = False):
        return search_entity(self.config, entity_name, show_query)

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

    # TODO: concrete set of ontology is required to know which columns are being dropped (user may drop features for transformations / other experiments)
    def recommend_feature_transformations(self, entity_df: pd.DataFrame = None, show_metadata: bool = True,
                                          show_query: bool = False):

        # adds the transformation type mapping to the resultant recommendation dataframe
        def add_transformation_type(df):
            df['Transformation_type'] = df['Transformation'].apply(lambda x: transformation_mapping.get(x))
            return df

        def handle_unseen_data():
            return add_transformation_type(self.recommender.get_transformation_recommendations(entity_df))

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

        if not show_metadata:
            transformation_info.drop(['Package', 'Function', 'Library', 'Author', 'Written_on', 'Pipeline_url'],
                                     axis=1, inplace=True)

        if entity_df is not None:
            table_ids = self.__table_transformations.get(tuple(entity_df.columns))
            if not table_ids:
                print('Processing unseen entity_df')
                return handle_unseen_data()

            tables = list(map(lambda x: get_table_name(self.config, table_id=x), table_ids))

            # filtering transformations w.r.t entity_df
            for index, value in tqdm(transformation_info.to_dict('index').items()):
                if value['Table'] not in tables:
                    transformation_info.drop(index=index, axis=0, inplace=True)

            if len(transformation_info) < 1:
                return transformation_info

            transformation_info = transformation_info.reset_index(drop=True)
            transformation_info.drop(['Dataset', 'Written_on', 'Pipeline_url', 'Dataset', 'Author', 'Table'],
                                     axis=1, inplace=True)
        return add_transformation_type(transformation_info)

    def apply_transformation(self, transformation_info: pd.Series, entity_df: pd.DataFrame = None):
        # TODO: add support for other transformations as well (ex. one-hot encoding, min-max scaling, etc.)
        transformation_model = None
        if entity_df is not None:  # apply transformations directly on entity_df passed by user
            df = entity_df
        else:  # load the table from the choice/row passed by the user from recommend_feature_transformations()
            df = self.load_table(transformation_info, print_table_name=False)
        transformation = transformation_info['Transformation']
        features = transformation_info['Feature']
        last_column = list(df.columns)[-1]  # for re-arranging columns
        if transformation == 'LabelEncoder':
            if self.mode != 'Automatic':
                print('Applying LabelEncoder transformation')
            for feature in tqdm(features):
                transformation_model = LabelEncoder()
                df[feature] = transformation_model.fit_transform(df[feature])
        elif transformation == 'StandardScaler':
            if self.mode != 'Automatic':
                print(
                    'CAUTION: Make sure you apply {} transformation only on the train set (This ensures there is no over-fitting due to feature leakage)\n'.format(
                        transformation) + \
                    'Use the transformation_model returned from this api to transform test set independently.\n')
                print('Applying StandardScaler transformation')
            try:
                transformation_model = StandardScaler(copy=False)
                df[features] = transformation_model.fit_transform(df[features])
            except sklearn.exceptions:
                print("{} couldn't be transformed".format(transformation_info['Table']))
        else:
            print(transformation, 'not supported yet!')
            return
        if self.mode != 'Automatic':
            print('{} feature(s) {} transformed successfully!'.format(len(features), features))

        return self.__re_arrange_columns(last_column, df), transformation_model

    def enrich(self, enrichment_info: pd.Series, entity_df: pd.DataFrame = None, ttl: int = 10):
        if entity_df is not None:  # entity_df passed by the user
            # get features to be enriched with
            features = self.get_features(enrichment_info=enrichment_info, entity_df_columns=tuple(entity_df.columns),
                                         show_status=False)
            features = [feature.split(':')[-1] for feature in features]
            if self.mode != 'Automatic':
                print('Enriching {} with {} feature(s) {}'.format('entity_df', len(features), features))
        else:  # option selected from search_enrichment_options()
            entity_df = pd.read_csv(enrichment_info['Table_path'])
            features = self.get_features(enrichment_info=enrichment_info, entity_df=entity_df, show_status=False)
            features = [feature.split(':')[-1] for feature in features]
            if self.mode != 'Automatic':
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
            if self.mode != 'Automatic':
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

        def handle_data_by_statistics(dependent_var, f_score):
            # TODO: add more feature selection techniques
            if select_by not in {'anova'}:
                raise ValueError("select_by can either be 'anova' or 'correlation'")
            if select_by == 'anova':
                # filter features based on f_value threshold
                f_value_threshold = float(input(' Enter F-value threshold '))
                f_score = f_score[f_score['F_value'] > f_value_threshold]
                independent_var = df[f_score['Feature']]  # features (X)
                if self.mode != 'Automatic':
                    print('Top {} feature(s) {} were selected based on highest F-value'.
                          format(len(independent_var.columns), list(independent_var.columns)))
                return independent_var, dependent_var

        df = copy.copy(entity_df)
        for feature in tqdm(list(entity_df.columns)):  # drop entity column and timestamp
            if is_entity_column(self.config, feature=feature, dependent_variable=dependent_variable) \
                    or feature == 'event_timestamp':
                df.drop(feature, axis=1, inplace=True)

        if self.mode != 'Automatic':
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
                return handle_data_by_statistics(dependent_var=y, f_score=feature_scores)

        if select_by == 'anova':
            return handle_data_by_statistics(dependent_var=y, f_score=feature_scores)

        elif select_by is None:  # select by pipelines
            X = entity_df[self.__get_features(entity_df=entity_df, filtered_columns=list(df.columns))]
            if self.mode != 'Automatic':
                print('{} feature(s) {} were selected based on previously abstracted pipelines'.format(len(X.columns),
                                                                                                       list(X.columns)))
            return X, y

    def clean_data(self, entity_df: pd.DataFrame, technique: str = None, visualize_missing_data: bool = True, show_query: bool = False):

        def get_columns_to_be_cleaned(df: pd.DataFrame):
            columns = pd.DataFrame(df.isnull().sum())
            columns.columns = ['Missing values']
            columns['Feature'] = columns.index
            columns = columns[columns['Missing values'] > 0]
            columns.sort_values(by='Missing values', ascending=False, inplace=True)
            columns.reset_index(drop=True, inplace=True)
            return columns

        def handle_unseen_data(df: pd.DataFrame, columns_to_clean: pd.DataFrame):
            columns = list(columns_to_clean['Feature'])
            if technique == 'drop':
                df.dropna(how='any', inplace=True)
                df.reset_index(drop=True, inplace=True)
                print(f'missing values from {columns} were dropped')
                return df
            elif technique == 'fill':
                fill_value = input('Enter the value to fill the missing data ')
                if fill_value not in {'mean', 'median', 'mode'}:  # fill constant value
                    df.fillna(fill_value, inplace=True)
                else:
                    if fill_value == 'median':
                        df.fillna(df.median(), inplace=True)
                    elif fill_value == 'mean':
                        df.fillna(df.mean(), inplace=True)
                    else:
                        def get_mode(feature: pd.Series):  # fill categorical data with mode
                            return feature.mode()[0]

                        for column in tqdm(columns):
                            mode = get_mode(entity_df[column])
                            entity_df[column].fillna(mode, inplace=True)

                df.reset_index(drop=True, inplace=True)
                print(f'missing values from {columns} were filled with {fill_value}')
                return df
            elif technique == 'interpolate':
                try:
                    df.interpolate(inplace=True)
                    df.reset_index(drop=True, inplace=True)
                    print(f'missing values from {columns} were interpolated')
                except TypeError:
                    print('only numerical features can be interpolated')
                return df
            else:
                if technique is None:
                    raise ValueError(
                        "pass cleaning technique, technique must be one out of 'drop', 'fill' or 'interpolate'")
                if technique not in {'drop', 'fill', 'interpolate'}:
                    raise ValueError("technique must be one out of 'drop', 'fill' or 'interpolate'")

        for na_type in tqdm({'none', 'n/a', 'na', 'nan', 'missing', '?', '', ' '}):
            if na_type in {'?', '', ' '}:
                entity_df.replace(na_type, np.nan, inplace=True)
            else:
                entity_df.replace('(?i)' + na_type, np.nan, inplace=True, regex=True)

        columns_to_be_cleaned = get_columns_to_be_cleaned(entity_df)

        if len(columns_to_be_cleaned) == 0:  # no missing values
            if self.mode != 'Automatic':
                print('nothing to clean')
            return entity_df

        if visualize_missing_data:
            # plot heatmap of missing values
            plt.rcParams['figure.dpi'] = 300
            plt.figure(figsize=(15, 7))
            sns.heatmap(entity_df.isnull(), yticklabels=False, cmap='Greens_r')
            plt.show()

            # plot bar-graph of missing values
            sns.set_color_codes('pastel')
            plt.rcParams['figure.dpi'] = 300
            plt.figure(figsize=(6, 3))

            ax = sns.barplot(x="Feature", y="Missing values", data=columns_to_be_cleaned,
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
            plt.show()

        table_id = search_entity_table(self.config, list(entity_df.columns))

        if len(table_id) < 1:  # i.e. table not profiled
            table_ids = self.__table_transformations.get(tuple(entity_df.columns))
            if table_ids is None:
                return handle_unseen_data(entity_df, columns_to_be_cleaned)

            data_cleaning_info = pd.concat([get_data_cleaning_info(self.config,
                                                                   table_id=table_ids[0], show_query=show_query),
                                            get_data_cleaning_info(
                                                self.config, table_id=table_ids[1], show_query=False)])
        else:
            data_cleaning_info = get_data_cleaning_info(self.config, table_id=table_id['Table_id'][0],
                                                        show_query=False)

        if len(data_cleaning_info) < 1:
            print('No cleaning technique found.')
            return entity_df

        # TODO: provide support for dropna()
        # TODO: handle all parameters and values
        for cleaning_info in data_cleaning_info.to_dict('index').values():
            function = cleaning_info['Function']
            parameter = cleaning_info['Parameter']
            value = cleaning_info['Value']

            if 'pandas.DataFrame.fillna' == function:
                entity_df.fillna(value, inplace=True)
                if self.mode != 'Automatic':
                    print("filled {} features using '{}' by {} = '{}'".format(len(columns_to_be_cleaned), function,
                                                                               parameter, value))

            elif 'pandas.DataFrame.interpolate' == function:
                entity_df.interpolate(value, inplace=True)
                if self.mode != 'Automatic':
                    print("interpolated {} features using '{}' by {} = '{}'".format(len(columns_to_be_cleaned), function,
                                                                               parameter, value))

        return entity_df


# TODO: refactor (make a generic function to return enrich table_ids from self.__table_transformations)
entity_data_types_mapping = {'N_int': 'integer', 'N_float': 'float', 'N_bool': 'boolean',
                             'T': 'string', 'T_date': 'timestamp', 'T_loc': 'string (location)',
                             'T_person': 'string (person)',
                             'T_org': 'string', 'T_code': 'string (code)', 'T_email': 'string (email)'}

if __name__ == "__main__":
    kgfarm = KGFarm(port=5820, database='kgfarm_test', show_connection_status=False)

import pandas as pd

from feature_discovery.src.api.template import *
from helpers.helper import *
from helpers.feast_templates import entity_skeleton, feature_view_skeleton, definitions
from tqdm import tqdm


class KGFarm:
    def __init__(self, port: object = 5820, database: str = 'kgfarm_test',
                 path_to_feature_repo: str = 'feature_repo/', show_connection_status: bool = True):
        # remove old feast meta
        if os.path.exists(path_to_feature_repo + 'data/registry.db'):
            os.remove(path_to_feature_repo + 'data/registry.db')
        if os.path.exists(path_to_feature_repo + 'data/online_store.db'):
            os.remove(path_to_feature_repo + 'data/online_store.db')
        if os.path.exists(path_to_feature_repo + 'data/driver_stats.parquet'):
            os.remove(path_to_feature_repo + 'data/driver_stats.parquet')

        self.path_to_feature_repo = path_to_feature_repo
        self.config = connect_to_stardog(port, database, show_connection_status)
        self.entities = {}  # for now needed for populating entity metadata while generating register .py file
        self.feature_views = {}
        # Need to maintain these because we do not update the KG
        self.__dropped_feature_views = set()
        self.__get_entities_and_feature_views()

    def __get_entities_and_feature_views(self, show_query: bool = False):
        def format_entities(entities):
            for entity_info in entities.to_dict('index').values():
                self.entities[entity_info['Entity_name']] = {'Entity_data_type': entity_info['Entity_data_type'],
                                                             'Physical_column': entity_info['Physical_column'],
                                                             'Physical_table': entity_info['Physical_table'],
                                                             'Uniqueness_ratio': entity_info['Uniqueness_ratio']}

        format_entities(get_default_entities(self.config, show_query))
        format_entities(get_multiple_entities(self.config, show_query))

        # load feature views with one or more entities
        for feature_view_info in get_feature_views(self.config, show_query). \
                to_dict('index').values():
            if feature_view_info['Feature_view'] in self.feature_views:  # multiple entity
                entity_to_update = self.feature_views[feature_view_info['Feature_view']]['Entity']
                entity_to_update.extend([feature_view_info['Entity']])
                self.feature_views[feature_view_info['Feature_view']] = {'Entity': entity_to_update,
                                                                         'Physical_table': feature_view_info[
                                                                             'Physical_table'],
                                                                         'File_source': feature_view_info[
                                                                             'File_source']}
            else:  # single entity
                self.feature_views[feature_view_info['Feature_view']] = {'Entity': [feature_view_info['Entity']],
                                                                         'Physical_table': feature_view_info[
                                                                             'Physical_table'],
                                                                         'File_source': feature_view_info[
                                                                             'File_source']}

        for feature_view_info in get_feature_views_without_entities(self.config, show_query).to_dict('index').values():
            self.feature_views[feature_view_info['Feature_view']] = {'Entity': None,
                                                                     'Physical_table': feature_view_info[
                                                                         'Physical_table'],
                                                                     'File_source': feature_view_info['File_source']}

    def get_entities(self):
        return convert_dict_to_dataframe('Entity', self.entities). \
            sort_values(by=['Uniqueness_ratio'], ascending=False).reset_index(drop=True)

    def get_feature_views(self):
        return convert_dict_to_dataframe('Feature_view', self.feature_views). \
            sort_values(by=['Feature_view']).reset_index(drop=True)

    def drop_feature_view(self, drop: list):
        if len(drop) == 0:
            print('Nothing to drop')
            return
        else:
            print('Dropped ', end='')
            for feature_view_to_be_dropped in drop:
                feature_view = feature_view_to_be_dropped['Feature_view']
                drop_status = self.feature_views.pop(feature_view, 'None')
                self.__dropped_feature_views.add(feature_view)
                if drop_status == 'None':
                    print('unsuccessful!\n')
                    raise ValueError(feature_view, ' not found!')
                else:
                    print(feature_view, end=' ')
            return self.get_feature_views()

    def get_optional_entities(self, show_query: bool = False):
        return get_optional_entities(self.config, show_query)

    def update_entity(self, entity_to_update_info: list):
        for update_info in tqdm(entity_to_update_info):
            feature_view = update_info['Feature_view']
            entity = update_info['Optional_entity'].replace('[', '').replace(']', '')
            feature_view_to_be_updated = self.feature_views.get(feature_view)
            feature_view_to_be_updated['Entity'] = [entity]
            # add optional entity info to finalized set of entities
            self.entities[entity] = {'Entity_data_type': update_info['Entity_data_type'],
                                     'Physical_column': update_info['Physical_column'],
                                     'Physical_table': update_info['Physical_table'],
                                     'Uniqueness_ratio': update_info['Uniqueness_ratio']}
            print("{} Updated! {} now uses '{}' entity".format(feature_view, feature_view, entity))
        return self.get_feature_views()

    # writes to file the predicted feature views and entities
    def finalize_feature_views(self, ttl: int = 30, save_as: str = 'predicted_register.py'):
        # delete the default feast file
        if os.path.exists(self.path_to_feature_repo + 'example.py'):
            os.remove(self.path_to_feature_repo + 'example.py')

        # add time to leave parameter to feature views
        for value in self.feature_views.values():
            value['Time_to_leave'] = ttl

        # generates the finalized feature view(s) and entities python file
        print('Finalizing feature views')
        with open(self.path_to_feature_repo + save_as, 'w') as py_file:
            # write basic library imports + documentation
            py_file.write(definitions())
            # write all entities
            for entity, entity_info in self.entities.items():
                py_file.write(entity_skeleton().
                              format(entity, entity, entity_info['Entity_data_type'], entity_info['Physical_column']))
            # write all feature views
            for feature_view, feature_view_info in tqdm(self.feature_views.items()):
                if feature_view_info['Entity']:  # feature view with one or multiple entities
                    feature_view = feature_view_skeleton(). \
                        format(feature_view,
                               feature_view,
                               feature_view_info['Entity'],
                               ttl,
                               feature_view_info['File_source'])
                else:  # feature views with no entity
                    feature_view = feature_view_skeleton(). \
                        format(feature_view,
                               feature_view,
                               [],
                               ttl,
                               feature_view_info['File_source'])

                py_file.write(feature_view)
            py_file.close()
        print('Predicted feature view(s) file saved at: ',
              os.path.abspath(self.path_to_feature_repo) + '/' + save_as)

    def get_enrichable_tables(self, show_query: bool = False):
        # TODO: gather information on how get_historical_features() work with feature view with multiple entities.
        enrichable_tables = get_enrichable_tables(self.config, show_query)
        enrichable_tables = enrichable_tables[~enrichable_tables['Enrich_with'].isin(self.__dropped_feature_views)]
        enrichable_tables = enrichable_tables.sort_values(by=['Table', 'Joinability_strength', 'Enrich_with'],
                                                          ascending=False). \
            reset_index(drop=True)
        enrichable_tables['Joinability_strength'] = enrichable_tables['Joinability_strength']. \
            apply(lambda x: str(int(x * 100)) + '%')
        return enrichable_tables

    def get_features(self, entity_df: pd.Series):
        # TODO: add support for fetching features that originate from multiple feature views at once.
        feature_view = entity_df['Enrich_with']
        # features in entity dataframe
        entity_df_features = get_columns(self.config, entity_df['Table'], entity_df['Dataset'])
        # features in feature view table
        feature_view_features = get_columns(self.config, entity_df['Physical_joinable_table'],
                                            entity_df['Dataset_feature_view'])
        # take difference
        return ['{}:'.format(feature_view) + feature for feature in feature_view_features if
                feature not in entity_df_features]


if __name__ == "__main__":
    kgfarm = KGFarm(path_to_feature_repo='../../../feature_repo/', show_connection_status=False)
    kgfarm.get_entities()

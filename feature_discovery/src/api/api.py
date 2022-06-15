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
        self.entities = {}
        self.feature_views = {}
        # self.entities_to_feature_views = {}
        self.__get_entities_and_feature_views()
        # self.entities, self.feature_views = self.__predict_entities_and_feature_views()

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
            if feature_view_info['Feature_view'] in self.feature_views:
                pass
                print('Multiple keys detected')
                # entities = self.feature_views[feature_view_info['Feature_view']]['Entity']
                # entities.extend([feature_view_info['Entity']])
                # self.feature_views[feature_view_info['Feature_view']] =
            else:
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

    def __predict_entities_and_feature_views(self, ttl: int = 10, show_query: bool = False):
        entities = {}
        feature_views = {}
        # get entities (here entity = primary key extracted from KG)
        entities_df = predict_entities(self.config, show_query)

        # add feature_views corresponding to each entity
        entities_df.insert(loc=0, column='Feature_view',
                           value=list('feature_view_' + str(i) for i in range(1, len(entities_df) + 1)))

        # convert to dictionary
        for k, v in entities_df.to_dict('index').items():
            entities[v['Entity']] = {'Entity_data_type': v['Entity_data_type']}

            feature_views[v['Feature_view']] = {'Entity': v['Entity'], 'Entity_data_type': v['Entity_data_type'],
                                                'Time_to_leave': ttl,
                                                'File_source': v['File_source'],
                                                'File_source_path': v['File_source_path'],
                                                'Dataset': v['Dataset']}
            self.entities_to_feature_views[v['Entity']] = v['Feature_view']

        # add feature views with entities
        # 1. get all tables
        # 2. get tables for which entities exist ()
        # 3. take difference
        # 4. add feature view id for the result from step 3
        # 5. add these feature views to the global feature view dictionary
        all_tables = get_all_tables(self.config, show_query=False)
        file_source_with_entities = entities_df['File_source'].tolist()
        feature_views_without_entities = all_tables[
            ~all_tables.File_source.isin(file_source_with_entities)].reset_index(
            drop=True)
        number_of_feature_views = len(feature_views)
        feature_views_without_entities.insert(loc=0, column='Feature_view',
                                              value=list('feature_view_' + str(i + number_of_feature_views) for i in
                                                         range(1, len(feature_views_without_entities) + 1)))

        for k, v in feature_views_without_entities.to_dict('index').items():
            feature_views[v['Feature_view']] = {'Entity': None,
                                                'Entity_data_type': None,
                                                'Time_to_leave': ttl,
                                                'File_source': v['File_source'],
                                                'File_source_path': v['File_source_path'],
                                                'Dataset': v['Dataset']}

        return entities, feature_views
        # return feature_views_without_entities

    def show_entities(self):
        return convert_dict_to_dataframe('Entity', self.entities)

    def show_feature_views(self):
        return convert_dict_to_dataframe('Feature_view', self.feature_views).sort_values(by=['Feature_view'])

    def drop_feature_view(self, drop: list):
        if len(drop) == 0:
            print('Nothing to drop')
            return
        else:
            print('Dropped ', end='')
            for feature_view_to_be_dropped in drop:
                feature_view = feature_view_to_be_dropped['Feature_view']
                drop_status = self.feature_views.pop(feature_view, 'None')
                if drop_status == 'None':
                    print('unsuccessful!\n')
                    raise ValueError(feature_view, ' not found!')
                else:
                    print(feature_view, end=' ')
            return self.show_feature_views()

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
                    feature_view = feature_view_skeleton().\
                        format(feature_view,
                               feature_view,
                               feature_view_info['Entity'],
                               ttl,
                               feature_view_info['File_source'])
                else:  # feature views with no entity
                    feature_view = feature_view_skeleton().\
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
        df = get_enrichable_tables(self.config, show_query)
        df['Enrich_with'] = df['Entity'].apply(lambda x: self.entities_to_feature_views.get(x))
        df['File_source_path'] = df['Enrich_with'].apply(lambda x: self.feature_views.get(x)['File_source_path'])
        df['Dataset_feature_view'] = df['Enrich_with'].apply(lambda x: self.feature_views.get(x)['Dataset'])
        df['File_source'] = df['Enrich_with'].apply(lambda x: self.feature_views.get(x)['File_source'])
        df = df.drop('Entity', axis=1)
        # rearrange columns
        # df = df[df.columns.tolist()[-1:] + df.columns.tolist()[:-1]]
        df = df[['Table', 'Enrich_with', 'Path_to_table', 'File_source_path', 'File_source', 'Dataset',
                 'Dataset_feature_view']]
        return df.sort_values('Enrich_with').reset_index(drop=True)

    def get_features(self, entity_df: pd.Series, show_query: bool = False):
        table = entity_df['Table']
        dataset = entity_df['Dataset']
        feature_view = entity_df['Enrich_with']
        entity_df_features = get_columns(self.config, table, dataset, show_query)
        features = get_columns(self.config, self.feature_views.get(feature_view)['File_source'],
                               self.feature_views.get(feature_view)['Dataset'], show_query)

        # subtract both feature lists
        features = ['{}:'.format(feature_view) + feature for feature in features if feature not in entity_df_features]

        return features


if __name__ == "__main__":
    kgfarm = KGFarm(path_to_feature_repo='../../../feature_repo/', show_connection_status=False)
    kgfarm.show_entities()

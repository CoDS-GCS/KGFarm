import os

import pandas as pd
import numpy as np
from feature_discovery.src.api.template import *
from helpers.helper import *
from helpers.feast_templates import entity_skeleton, feature_view_skeleton, definitions


class KGFarm:
    def __init__(self, blazegraph_port=9999, blazegraph_namespace: str = 'glac',
                 path_to_feature_repo: str = 'feature_repo/', show_connection_status: bool = True):
        # remove old feast meta
        if os.path.exists(path_to_feature_repo + 'data/registry.db'):
            os.remove(path_to_feature_repo + 'data/registry.db')
        if os.path.exists(path_to_feature_repo + 'data/online_store.db'):
            os.remove(path_to_feature_repo + 'data/online_store.db')
        if os.path.exists(path_to_feature_repo + 'data/driver_stats.parquet'):
            os.remove(path_to_feature_repo + 'data/driver_stats.parquet')

        self.path_to_feature_repo = path_to_feature_repo
        self.config = connect_to_blazegraph(blazegraph_port, blazegraph_namespace, show_connection_status)
        self.entities_to_feature_views = {}
        self.entities, self.feature_views = self.__predict_entities_and_feature_views()

    # predicts the default feature views and entities at the time of initialization
    def __predict_entities_and_feature_views(self, ttl: int = 10, show_query: bool = False):
        entities = {}
        feature_views = {}
        entities_df = predict_entities(self.config, show_query)
        # add feature_views corresponding to each entity (here entity = primary key extracted from KG)
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
        return entities, feature_views

    def show_entities(self):
        return convert_dict_to_dataframe('Entity', self.entities)

    def show_feature_views(self):
        return convert_dict_to_dataframe('Feature_view', self.feature_views)

    # writes to file the predicted feature views and entities
    def predict_feature_views(self, ttl: int = 10,
                              show_feature_views: bool = False):
        if os.path.exists(self.path_to_feature_repo + 'predicted_register.py'):
            os.remove(self.path_to_feature_repo + 'predicted_register.py')
        if os.path.exists(self.path_to_feature_repo + 'example.py'):
            os.remove(self.path_to_feature_repo + 'example.py')

        # update feature views
        for value in self.feature_views.values():
            value['Time_to_leave'] = ttl

        # write to file
        with open(self.path_to_feature_repo + 'predicted_register.py', 'a') as py_file:
            # write basic library imports + documentation
            py_file.write(definitions())
            # write all entities
            for entity, v in self.entities.items():
                entity = entity_skeleton().format(entity, entity, v['Entity_data_type'], entity)
                py_file.write(entity)
            # write all feature views
            for feature_view, v in self.feature_views.items():
                feature_view = feature_view_skeleton().format(feature_view, feature_view, v['Entity'], ttl,
                                                              v['File_source_path'])
                py_file.write(feature_view)
            py_file.close()
        print('Predicted entities and feature view(s) File saved at: ',
              os.path.abspath(self.path_to_feature_repo) + '/predicted_register.py\n')
        if show_feature_views:
            return self.show_feature_views()

    def get_enrichment_tables(self, show_query: bool = False):
        df = get_enrichment_tables(self.config, show_query)
        df['Enrich_with'] = df['Entity'].apply(lambda x: self.entities_to_feature_views.get(x))
        df['File_source_path'] = df['Enrich_with'].apply(lambda x: self.feature_views.get(x)['File_source_path'])
        df['Dataset_feature_view'] = df['Enrich_with'].apply(lambda x: self.feature_views.get(x)['Dataset'])
        df['File_source'] = df['Enrich_with'].apply(lambda x: self.feature_views.get(x)['File_source'])
        df = df.drop('Entity', axis=1)
        # rearrange columns
        # df = df[df.columns.tolist()[-1:] + df.columns.tolist()[:-1]]
        df = df[['Table', 'Enrich_with', 'Path_to_table', 'File_source_path', 'File_source', 'Dataset', 'Dataset_feature_view']]
        return df.sort_values('Enrich_with')

    # def predict_features(self, table_info: pd.Series, show_query: bool = False):
    #     table = table_info['table_name']
    #     dataset = table_info['dataset_name']
    #     # print('table: ', table)
    #     joinable_table, cols = predict_features(self.config, table, dataset, show_query)
    #     # print('joinable table: ', joinable_table)
    #     fv = ''
    #     for k, v in self.feature_views.items():
    #         if joinable_table == self.feature_views.get(k):
    #             fv = k
    #     predicted_features = []
    #     for c in cols:
    #         predicted_features.append(fv + ':' + c)
    #     return predicted_features


if __name__ == "__main__":
    kgfarm = KGFarm(path_to_feature_repo='../../../feature_repo/', show_connection_status=False)
    print(kgfarm.predict_feature_views(ttl=10, show_feature_views=True))
    print(kgfarm.get_enrichment_tables())
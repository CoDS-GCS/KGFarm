import pandas as pd
import numpy as np
from feature_discovery.src.api.template import *
from helpers.helper import *
from helpers.feast_templates import entity_skeleton, feature_view, definitions

# TODO: populate entities and feature_views while initializing

class KGFarm:
    def __init__(self, blazegraph_port=9999, blazegraph_namespace: str = 'glac', show_connection_status: bool = True):
        self.config = connect_to_blazegraph(blazegraph_port, blazegraph_namespace, show_connection_status)
        self.entities = {}
        self.feature_views = {}
    # def predict_entities_and_feature_views(self, ttl: int = 1000, thresh=0.70, show_query=False):
    #
    #     def create_file(df_feast: pd.DataFrame, path: str = 'feature_repo/'):
    #         entities = df_feast[['Source_table', 'Source_column', 'Source_column_type', 'Source_table_path']] \
    #             .drop_duplicates()
    #         feast_driver_file = definitions()
    #         count = 1
    #         for index, row in entities.iterrows():
    #             e = entity().format(count, row['Source_column'], row['Source_column_type'], row['Source_column'])
    #             f = feature_view().format(count, count, row['Source_column'], ttl, row['Source_table_path'])
    #             count = count + 1
    #             feast_driver_file = feast_driver_file + e + f
    #
    #         with open(path + 'predicted_register.py', 'w') as py_file:
    #             py_file.write(feast_driver_file)
    #             print('Predicted entities and feature view(s) File saved at: ', path + 'predicted_register.py')
    #
    #     df = pd.DataFrame(predict_entities_and_feature_views(self.config, thresh, show_query)).\
    #         replace(['N', 'T'], ['INT64', 'STRING'])
    #     create_file(df)
    #     return df

    def predict_entities_and_feature_views(self, ttl: int = 1000, path_to_feature_repo: str = 'feature_repo/', show_feature_views: bool = False, show_query: bool = False):
        entities, self.feature_views, entities_df = predict_entities(self.config, show_query)
        if os.path.exists(path_to_feature_repo + 'predicted_register.py'):
            os.remove(path_to_feature_repo + 'predicted_register.py')

        with open(path_to_feature_repo + 'predicted_register.py', 'a') as py_file:
            py_file.write(definitions())
            count = 0
            for k, v in entities.items():
                e = entity_skeleton().format(v['name'], v['name'], v['datatype'], v['name'])
                py_file.write(e)
                for file_source in v['FileSource_path']:
                    count = count + 1
                    f = feature_view().format(count, count, v['name'], ttl, file_source)
                    py_file.write(f)
            py_file.close()
            print('Predicted entities and feature view(s) File saved at: ', os.path.abspath(path_to_feature_repo) + '/predicted_register.py\n')

        if show_feature_views:
            print('Showing predicted Feature views:')
            cols = entities_df.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            return entities_df[cols]

    def predict_features(self, table_info: pd.Series, show_query: bool = False):
        table = table_info['table_name']
        dataset = table_info['dataset_name']
        # print('table: ', table)
        joinable_table, cols = predict_features(self.config, table, dataset, show_query)
        # print('joinable table: ', joinable_table)
        fv = ''
        for k, v in self.feature_views.items():
            if joinable_table == self.feature_views.get(k):
                fv = k
        predicted_features = []
        for c in cols:
            predicted_features.append(fv+':'+c)
        return predicted_features


if __name__ == "__main__":
    kgfarm = KGFarm(show_connection_status=False)
    df = kgfarm.predict_entities_and_feature_views(path_to_feature_repo='../../../feature_repo/', show_query=False)
    print(df)

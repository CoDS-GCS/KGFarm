import pandas as pd

from feature_discovery.src.api.template import *
from helpers.helper import *
from helpers.feast_templates import entity, entity_skeleton, feature_view, definitions


class KGFarm:
    def __init__(self, blazegraph_port=9999, blazegraph_namespace: str = 'glac', show_connection_status: bool = True):
        self.config = connect_to_blazegraph(blazegraph_port, blazegraph_namespace, show_connection_status)

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

    def predict_entities_and_feature_views(self, ttl: int = 1000, path_to_feature_repo: str = 'feature_repo/', show_query: bool = False):
        entities, entities_df = predict_entities(self.config, show_query)
        if os.path.exists(path_to_feature_repo + 'predicted_register.py'):
            os.remove(path_to_feature_repo + 'predicted_register.py')

        with open(path_to_feature_repo + 'predicted_register.py', 'a') as py_file:
            py_file.write(definitions())
            count = 0
            for k, v in entities.items():
                count = count + 1
                e = entity_skeleton().format(v['name'], v['name'], v['datatype'], v['name'])
                f = feature_view().format(count, count, v['name'], ttl, v['Table_path'])
                py_file.write(e)
                py_file.write(f)
            py_file.close()
            print('Predicted entities and feature view(s) File saved at: ', os.path.abspath(path_to_feature_repo) + '/predicted_register.py')

        return entities_df


if __name__ == "__main__":
    kgfarm = KGFarm(show_connection_status=False)
    kgfarm.predict_entities_and_feature_views(path_to_feature_repo='../../../feature_repo/', show_query=False)

from feature_discovery.src.api.template import *
from helpers.helper import *
from helpers.feast_templates import *


class KGFarm:
    def __init__(self, blazegraph_port=9999, blazegraph_namespace: str = 'glac'):
        self.config = connect_to_blazegraph(blazegraph_port, blazegraph_namespace)

    def predict_entities_and_feature_views(self, ttl: int = 1000, thresh=0.70, show_query=False):

        def create_file(df_feast: pd.DataFrame, path: str = 'feature_repo/'):
            entities = df_feast[['source_table', 'source_column', 'source_column_type', 'source_table_path']] \
                .drop_duplicates()
            feast_driver_file = definitions()
            count = 1
            for index, row in entities.iterrows():
                e = entity().format(count, row['source_column'], row['source_column_type'], row['source_column'])
                f = feature_view().format(count, count, row['source_column'], ttl, row['source_table_path'])
                count = count + 1
                feast_driver_file = feast_driver_file + e + f

            with open(path + 'predicted_register.py', 'w') as py_file:
                py_file.write(feast_driver_file)
                print('Predicted entities and feature view(s) File saved at: ', path + 'predicted_register.py')

        data = predict_entities_and_feature_views(self.config, thresh, show_query=show_query)
        df = pd.DataFrame(list(data), columns=['source_table', 'source_column', 'source_column_type',
                                               'source_table_path',
                                               'target_table', 'target_column', 'confidence_score']). \
            replace(['N', 'T'], ['INT64', 'STRING'])
        create_file(df)

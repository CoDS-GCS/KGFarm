from tqdm import tqdm
from operations.template import *
from helpers.helper import connect_to_stardog
from feature_discovery.src.graph_builder.farm_graph_builder import build_farm_graph


class Governor:
    def __init__(self, config=None, graph_name: str = 'farm.ttl', lids_db: str = 'kgfarm_test', port: int = 5820,
                 show_connection_status: bool = False):
        if config is None:
            self.config = connect_to_stardog(port=port, db=lids_db, show_status=show_connection_status)
        else:
            self.config = config
        self.graph_name = graph_name

    def drop_feature_view(self, drop: list):
        for feature_view_info in tqdm(drop):
            feature_view = feature_view_info['Feature_view']
            if exists(self.config, feature_view):
                drop_feature_view(self.config, feature_view)
            else:
                error = feature_view + " doesn't exists!"
                raise ValueError(error)
        print('Dropped ', [i['Feature_view'] for i in drop], 'successfully!')

    def update_entity(self, entity_updation_info: list):
        for updation_info in tqdm(entity_updation_info):
            feature_view = updation_info['Feature_view']
            entity = updation_info['Entity']
            column_physical_representation_to_be_inserted = updation_info['Optional_physical_representation']
            remove_current_physical_representation_of_an_entity(self.config, feature_view=feature_view)
            insert_current_physical_representation_of_an_entity(self.config, feature_view=feature_view,
                                                                column=column_physical_representation_to_be_inserted,
                                                                entity=entity)
            print("Entity '{}' uses '{}' as its physical representation".
                  format(entity, column_physical_representation_to_be_inserted))

    def build_farm_graph(self):
        build_farm_graph(config=self.config, graph_name=self.graph_name)


if __name__ == "__main__":
    governor = Governor()
    governor.build_farm_graph()

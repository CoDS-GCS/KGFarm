from tqdm import tqdm
from datetime import datetime
from operations.template import *
from helpers.helper import connect_to_stardog


class Governor:
    def __init__(self, config=None, graph_path: str = 'farm.ttl', lids_db: str = 'kgfarm_test', port: int = 5820,
                 show_connection_status: bool = False):
        if config is None:  # for graph generation
            self.config = connect_to_stardog(port=port, db=lids_db, show_status=show_connection_status)
        else:  # for KGFarm APIs (use the db config coming from user)
            self.config = config
        self.graph_path = graph_path
        self.graph = open(self.graph_path, 'w')
        self.graph.write('# Farm Graph generated on ' + str(datetime.now()) + '\n')
        self.triples = set()
        self.farm_ontology = {'prefix':   'http://kgfarm.com/ontology/',
                              'rdf-type': '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>',
                              'name':     '<http://schema.org/name>'}

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

    def __dump_triples(self):
        self.graph.write('\n'.join(self.triples))
        self.triples = set()

    def build_farm_graph(self):
        """
        1. query LiDS graph to fetch column pairs with content similarity
        2. use PKFK classifier to get the column pairs (from 1) which have a primary key-foreign key relationship
        3. for all tables in LiDS graph:
            i)   generate a feature view node
            ii)  link the feature view node to physical table in LiDS
            iii) name the feature view
        4. tbc...
        """
        # step 3
        digits = len(str(len(get_table_ids(self.config))))  # zero padding for clean feature view numbering
        table_ids = set(get_table_ids(config=self.config)['Table_id'])
        feature_view_count = 0
        for table_url in tqdm(table_ids):
            feature_view_count = feature_view_count + 1
            zeroes_to_be_padded = digits - len(str(feature_view_count))
            feature_view = '0' * zeroes_to_be_padded + str(feature_view_count)
            self.triples.add(f'<{self.farm_ontology.get("prefix")}feature-view-{feature_view}> {self.farm_ontology.get("rdf-type")} <{self.farm_ontology.get("prefix")}FeatureView>.')
            self.triples.add(f'<{table_url}> <{self.farm_ontology.get("prefix")}hasFeatureView> <{self.farm_ontology.get("prefix")}feature-view-{feature_view}>.')
            self.triples.add(f'<{self.farm_ontology.get("prefix")}feature-view-{feature_view}> {self.farm_ontology.get("name")} "Feature view {feature_view}".')
            self.__dump_triples()

        self.graph.close()


if __name__ == "__main__":
    governor = Governor()
    governor.build_farm_graph()

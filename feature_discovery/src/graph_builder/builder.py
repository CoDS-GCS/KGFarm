from time import time
from datetime import datetime
from helpers.helper import *
from feature_discovery.src.api.template import *

# TODO: unify template headers


class Builder:
    def __init__(self, output_path: str = 'Farm.nq', port: int = 5820, database: str = 'kgfarm_dev',
                 show_connection_status: bool = False):
        self.config = connect_to_stardog(port, database, show_connection_status)
        self.output_path = output_path
        self.graph = open(output_path, 'w')
        self.graph.write('# Farm Graph generated on ' + str(datetime.now()) + '\n')
        self.ontology = {'kgfarm': 'http://kgfarm.com/ontology/',
                         'featureView': 'http://kgfarm.com/ontology/featureView/',
                         'entity': 'http://kgfarm.com/ontology/entity/'}

        self.triple_format = '<{}> <{}> "{}".'
        self.triples = set()
        self.table_to_feature_view = {}
        self.column_to_entity = {}
        self.unmapped_tables = set()

    def __dump_triples(self):
        self.graph.write('\n'.join(self.triples))
        self.triples = set()

    def __annotate_entity_and_feature_view_mapping(self, column_id, entity_name, table_id, uniqueness_ratio):
        triple_format = '<{}> <{}> <{}>'
        # triple for entity name -> physical column id : entity name
        self.triples.add(self.triple_format.format(column_id, self.ontology.get('entity') + 'name', entity_name))
        # triple for feature view - entity mapping -> physical table id : column id
        self.triples.add('<<' + triple_format.format(table_id, self.ontology.get('kgfarm') + 'uses',
                                                     column_id) + '>> <' + self.ontology.get(
            'kgfarm') + 'confidence>' + ' "{}"^^xsd:double.'.format(str(uniqueness_ratio)))

    # does one-to-one mapping of table -> feature view
    def annotate_feature_views(self):
        self.graph.write('# 1. Feature Views, one-to-one mapping with tables \n')
        table_ids = get_table_ids(self.config)['Table_id'].tolist()
        feature_view_count = 0
        for table_id in table_ids:
            feature_view_count = feature_view_count + 1
            self.triples.add(self.triple_format.format(
                table_id,
                self.ontology.get('featureView') + 'name',
                'Feature view {}'.format(feature_view_count)))
            self.table_to_feature_view[table_id] = 'Feature view {}'.format(feature_view_count)
        self.__dump_triples()

    def annotate_entity_mapping(self):
        self.graph.write('\n# 2. Entities and feature view - entity mappings \n')
        entities = get_entities(self.config)
        mapped_tables = set()
        for entity_info in entities.to_dict('index').values():
            entity_name = (entity_info['Candidate_entity_name'] + '_' + entity_info['File_source']). \
                replace('id', '').replace('.parquet', '')

            table_id = entity_info['Table_id']
            column_id = entity_info['Candidate_entity_id']
            uniqueness_ratio = entity_info['Score']

            self.__annotate_entity_and_feature_view_mapping(column_id, entity_name,
                                                            table_id, uniqueness_ratio)
            self.column_to_entity[column_id] = entity_name
            mapped_tables.add(table_id)

        all_tables = set(list(self.table_to_feature_view.keys()))
        self.unmapped_tables = all_tables.difference(mapped_tables)
        self.__dump_triples()

    def annotate_unmapped_feature_views(self):
        pkfk_relations = get_pkfk_relations(self.config)
        # filter relationships to the ones that were left unmapped
        pkfk_relations = pkfk_relations[pkfk_relations.Primary_table_id.isin(self.unmapped_tables)]

        for unmapped_feature_view in pkfk_relations.to_dict('index').values():
            entity_name = (unmapped_feature_view['Primary_column'] + '_' + unmapped_feature_view['Primary_table']). \
                replace('id', '').replace('.parquet', '')
            table_id = unmapped_feature_view['Primary_table_id']
            column_id = unmapped_feature_view['Primary_column_id']
            uniqueness_ratio = unmapped_feature_view['Primary_key_uniqueness_ratio']

            self.__annotate_entity_and_feature_view_mapping(column_id, entity_name,
                                                            table_id, uniqueness_ratio)
            self.column_to_entity[column_id] = entity_name
            self.unmapped_tables.remove(table_id)
        self.__dump_triples()

    def summarize_graph(self):
        feature_views_generated = len(self.table_to_feature_view.values())
        entities_generated = len(self.column_to_entity.values())
        print('\n• {} summary\n\t- Total feature view(s) generated: {}'
              '\n\t- Total entities generated: {}\n\t- Feature view(s) with no entity: {} / {}'
              .format(self.output_path, feature_views_generated, entities_generated, len(self.unmapped_tables), feature_views_generated))


def generate_farm_graph():
    start = time()
    builder = Builder(port=5820, database='kgfarm_dev', show_connection_status=True)
    builder.annotate_feature_views()
    builder.annotate_entity_mapping()
    builder.annotate_unmapped_feature_views()
    print('\n• Farm graph generated successfully\n\t- Time taken: {}\n\t- Saved at: {}'.
          format(time_taken(start, time()), os.path.abspath(builder.output_path)))
    builder.summarize_graph()


def upload_farm_graph(db: str = 'kgfarm_test', graph: str = 'Farm.nq'):
    print('\n• Uploading {} to {} database'.format(graph, db))
    os.system('stardog data remove --all kgfarm_test')
    os.system('stardog data add --format turtle kgfarm_test ../../../helpers/sample_data/graph/LiDS.nq')
    os.system('stardog data add --format turtle {} {}'.format(db, graph))


if __name__ == "__main__":
    generate_farm_graph()
    upload_farm_graph(db='kgfarm_test', graph='Farm.nq')

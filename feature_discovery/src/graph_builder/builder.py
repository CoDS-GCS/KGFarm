import sys
from time import time
from tqdm import tqdm
from datetime import datetime
sys.path.append('../../../')
from helpers.helper import *
from feature_discovery.src.api.template import *


class Builder:
    def __init__(self, output_path: str = 'Farm.nq', port: int = 5820, database: str = 'kgfarm_dev',
                 show_connection_status: bool = False):
        self.config = connect_to_stardog(port, database, show_connection_status)
        self.output_path = output_path
        self.graph = open(output_path, 'w')
        self.graph.write('# Farm Graph generated on ' + str(datetime.now()) + '\n')

        # TODO: revise Farm ontology
        self.ontology = {'kgfarm': 'http://kgfarm.com/ontology/',
                         'featureView': 'http://kgfarm.com/ontology/featureView/',
                         'entity': 'http://kgfarm.com/ontology/entity/'}

        # TODO: refactor triple formatting for all 3 triple types used.
        """
        1. <> <> "".
        2. <> <> <>.
        3. <<<> <> <>>> certainty "".
        """
        self.triple_format = '<{}> <{}> "{}".'

        self.triples = set()
        self.table_to_feature_view = {}
        self.column_to_entity = {}
        self.unmapped_tables = set()
        self.tables_with_multiple_entities = set()
        self.direct_entity_table_mapping = {}

    def __dump_triples(self):
        self.graph.write('\n'.join(self.triples))
        self.triples = set()

    def __annotate_entity_and_feature_view_mapping(self, column_id, table_id, uniqueness_ratio, relation):
        triple_format = '<{}> <{}> <{}>'
        self.triples.add('<<' + triple_format.format(table_id, self.ontology.get('kgfarm') + relation,
                                                     column_id) + '>> <' + self.ontology.get(
            'kgfarm') + 'confidence>' + ' "{}"^^xsd:double.'.format(str(uniqueness_ratio)))
       
    def __annotate_entity_name(self, column_id, entity_name):
        # triple for entity name -> physical column id : entity name
        self.triples.add(self.triple_format.format(column_id, self.ontology.get('entity') + 'name', entity_name))

    def __elect_default_entity(self, table_id, default_entities, column_id=None, entity_name=None):
        if len(default_entities) == 1:  # table with single entity detected
            column_id = list(default_entities.keys())[0]
            uniqueness_ratio = default_entities.get(column_id)['uniqueness']
            entity_name = default_entities.get(column_id)['name']

        else:  # table with multiple entities detected
            uniqueness_ratios = list(score['uniqueness'] for score in default_entities.values())
            uniqueness_ratio = max(uniqueness_ratios)
            if uniqueness_ratios.count(uniqueness_ratio) == 1:  # table with single maximum entity
                # if count == 1 i.e. there exists only one maximum value of uniqueness
                for candidate_column_id, candidate_column_info in default_entities.items():
                    if uniqueness_ratio == candidate_column_info['uniqueness']:
                        column_id = candidate_column_id
                        entity_name = candidate_column_info['name']
                        break
            else:  # table with multiple entities having equal uniqueness ratio
                candidate_column_ids = set()
                max_number_of_relations = 0
                column_id = None
                for candidate_column_id, candidate_column_info in default_entities.items():
                    if candidate_column_info['uniqueness'] == uniqueness_ratio:
                        candidate_column_ids.add(candidate_column_id)
                        n_relations = int(get_number_of_relations(self.config,
                                                            candidate_column_id)[0]['Number_of_relations']['value'])

                        if max_number_of_relations <= n_relations:
                            column_id = candidate_column_id
                            entity_name = candidate_column_info['name']
                            max_number_of_relations = n_relations

        self.column_to_entity[column_id] = entity_name
        self.direct_entity_table_mapping[table_id] = column_id
        self.__annotate_entity_and_feature_view_mapping(column_id, table_id, uniqueness_ratio, 'hasDefaultEntity')
        self.__annotate_entity_name(column_id, entity_name)

    # does one-to-one mapping of table -> feature view
    def annotate_feature_views(self):
        print('\n• Annotating feature views')
        self.graph.write('# 1. Feature Views, one-to-one mapping with tables \n')
        table_ids = get_table_ids(self.config)['Table_id'].tolist()
        feature_view_count = 0
        for table_id in tqdm(table_ids):
            feature_view_count = feature_view_count + 1
            self.triples.add(self.triple_format.format(
                table_id,
                self.ontology.get('featureView') + 'name',
                'Feature_view_{}'.format(feature_view_count)))
            self.table_to_feature_view[table_id] = 'Feature_view_{}'.format(feature_view_count)
        self.__dump_triples()

    def annotate_entity_mapping(self):
        mapped_tables = set()
        candidates_for_default_entities = {}
        print('• Annotating entities to feature views')
        self.graph.write('\n# 2. Entities and feature view - entity mappings \n')
        # get candidate entities (columns with high uniqueness) sorted by table names
        entities = detect_entities(self.config)
        """
        The candidate entity columns are sorted/ / grouped based upon the Table from which they originate.
        i.e. first, we populate all detected entities PER TABLE and then call the function that elects the default entity.  
        """""
        table_to_process = list(entities.to_dict('index').values())[0]['Primary_table_id']
        for entity_info in tqdm(entities.to_dict('index').values()):
            entity_name = (entity_info['Primary_column'].replace(' ', '') + '_' + entity_info['Primary_table']). \
                replace('id', '').replace('.parquet', '')
            table_id = entity_info['Primary_table_id']
            column_id = entity_info['Primary_column_id']
            uniqueness_ratio = entity_info['Primary_key_uniqueness_ratio']

            if table_id != table_to_process:
                """
                moved to new set of entities for different table, given by table_id, 
                while table_to_process contains that very table for which we populated candidates for default entity.
                """
                self.__elect_default_entity(table_to_process, candidates_for_default_entities)
                table_to_process = table_id
                candidates_for_default_entities = {}

            candidates_for_default_entities[column_id] = {'name': entity_name, 'uniqueness': uniqueness_ratio}

            self.__annotate_entity_and_feature_view_mapping(column_id, table_id, uniqueness_ratio, 'hasEntity')
            mapped_tables.add(table_id)
        self.__elect_default_entity(table_to_process, candidates_for_default_entities)
        all_tables = set(list(self.table_to_feature_view.keys()))
        self.unmapped_tables = all_tables.difference(mapped_tables)
        self.__dump_triples()

    # get all pkfk relationships from the graph and remove the Tables for which Entities are already generated
    def annotate_unmapped_feature_views(self):
        print('• Annotating unmapped feature views')
        pkfk_relations = get_pkfk_relations(self.config)
        # filter relationships to the ones that were left unmapped
        pkfk_relations = pkfk_relations[pkfk_relations.Primary_table_id.isin(self.unmapped_tables)]
        for unmapped_feature_view in tqdm(pkfk_relations.to_dict('index').values()):
            entity_name = (unmapped_feature_view['Primary_column'] + '_' + unmapped_feature_view['Primary_table']). \
                replace('id', '').replace('.parquet', '')
            table_id = unmapped_feature_view['Primary_table_id']
            column_id = unmapped_feature_view['Primary_column_id']
            uniqueness_ratio = unmapped_feature_view['Primary_key_uniqueness_ratio']

            self.__annotate_entity_and_feature_view_mapping(column_id, table_id, uniqueness_ratio, 'hasMultipleEntities')
            self.__annotate_entity_name(column_id, entity_name)
            self.column_to_entity[column_id] = entity_name
            # if table_id is absent from self.unmapped_tables that means that table_id has multiple entities
            if table_id in self.unmapped_tables:
                self.unmapped_tables.remove(table_id)
            else:
                self.tables_with_multiple_entities.add(table_id)

        # TODO: look for better ways of annotating feature views without entity
        for table_id in self.unmapped_tables:
            self.triples.add(self.triple_format.format(table_id, self.ontology.get('kgfarm') + 'hasNoEntity', 0))

        self.__dump_triples()

    def summarize_graph(self):
        # feature view info
        total_feature_views_generated = len(self.table_to_feature_view.values())
        feature_views_with_multiple_entities_generated = len(self.tables_with_multiple_entities)
        feature_views_with_no_entity = len(self.unmapped_tables)
        feature_view_with_single_entity = total_feature_views_generated - feature_views_with_multiple_entities_generated - feature_views_with_no_entity

        # TODO: add info about how many entities are backup entities, default and multiple in graph summary

        # entity info
        total_entities_generated = len(self.column_to_entity.values())

        graph_size = str(round((os.path.getsize(self.output_path) * 0.001), 3))

        print('\n• {} summary\n\t- Total entities generated: {}\n\t- Total feature views generated: {}'
              '\n\t- Feature view breakdown:\n\t\t-> Feature view with single entity: {} / {}'
              '\n\t\t-> Feature view with multiple entities: {} / {}'
              '\n\t\t-> Feature view with no entity: {} / {}'
              '\n\t- Graph size: {} KB'.
              format(self.output_path,
                     total_entities_generated,
                     total_feature_views_generated,
                     feature_view_with_single_entity,
                     total_feature_views_generated,
                     feature_views_with_multiple_entities_generated,
                     total_feature_views_generated,
                     feature_views_with_no_entity,
                     total_feature_views_generated,
                     graph_size))


def generate_farm_graph(db, port):
    start = time()
    builder = Builder(port=port, database=db, show_connection_status=True)
    builder.annotate_feature_views()
    builder.annotate_entity_mapping()
    builder.annotate_unmapped_feature_views()
    print('\n• Farm graph generated successfully!\n\t- Time taken: {}\n\t- Saved at: {}'.
          format(time_taken(start, time()), os.path.abspath(builder.output_path)))
    builder.summarize_graph()


def upload_farm_graph(db: str = 'kgfarm_test', graph: str = 'Farm.nq'):
    print('\nUploading {} to {} database'.format(graph, db))
    os.system('stardog data remove --all {}'.format(db))
    os.system('stardog data add --format turtle {} ../../../helpers/sample_data/graph/LiDS.nq'.format(db))
    os.system('stardog data add --format turtle {} {}'.format(db, graph))


if __name__ == "__main__":
    generate_farm_graph(db='Sample_banking_data', port=5820)
    upload_farm_graph(db='kgfarm_test', graph='Farm.nq')



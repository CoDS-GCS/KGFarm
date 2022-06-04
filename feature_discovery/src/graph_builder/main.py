import os
from time import time
from helpers.helper import *
from feature_discovery.src.api.template import *


class Builder:
    def __init__(self, output_path: str = 'Farm.nq', port: int = 5820, database: str = 'kgfarm_dev',
                 show_connection_status: bool = False):
        self.config = connect_to_stardog(port, database, show_connection_status)
        self.output_path = output_path
        self.graph = open(output_path, 'w')
        self.ontology = {'kgfarm': 'http://kgfarm.com/ontology/'}
        self.triple_format = '<{}> <{}> "{}".'
        self.triples = set()
        self.table_to_feature_view = {}

    def dump_triples(self):
        self.graph.write('\n'.join(self.triples))
        self.triples = set()

    # does one-to-one mapping of table -> feature view
    def annotate_feature_views(self):
        self.graph.write('# Feature Views, one-to-one mapping with tables \n')
        table_ids = get_table_ids(self.config)['Table_id'].tolist()
        feature_view_count = 0
        for table_id in table_ids:
            feature_view_count = feature_view_count + 1
            self.triples.add(self.triple_format.format(
                table_id,
                self.ontology.get('kgfarm') + 'name',
                'Feature view {}'.format(feature_view_count)))
            self.table_to_feature_view[table_id] = 'Feature view {}'.format(feature_view_count)
        self.dump_triples()

    def annotate_entities(self):
        triple_format = '<{}> <{}> <{}>.'
        self.graph.write('\n# Entities\n')
        entities = get_entities(self.config)

        for entity_info in entities.to_dict('index').values():
            entity_name = (entity_info['Candidate_entity_name'] + '_' + entity_info['File_source']).\
                replace('id', '').replace('.parquet', '')

            table_id = entity_info['Table_id']
            column_id = entity_info['Candidate_entity_id']
            self.triples.add(self.triple_format.format(column_id, self.ontology.get('kgfarm') + 'name', entity_name))
            self.triples.add(triple_format.format(table_id, self.ontology.get('kgfarm')+'uses', column_id))
        self.dump_triples()


def main():
    start = time()
    builder = Builder(port=5820, database='kgfarm_dev')
    builder.annotate_feature_views()
    builder.annotate_entities()
    print('\n• graph generated in :', time_taken(start, time()))
    print('• uploading new graph to database')
    os.system('stardog data remove --all kgfarm_test')
    os.system('stardog data add --format turtle kgfarm_test Farm.nq')


if __name__ == "__main__":
    main()

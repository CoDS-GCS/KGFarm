from helpers.helper import *
from time import time
from feature_discovery.src.api.template import *


class Builder:
    def __init__(self, output_path: str = 'Farm.nq', port: int = 5820, database: str = 'kgfarm_dev',
                 show_connection_status: bool = False):
        self.config = connect_to_stardog(port, database, show_connection_status)
        self.output_path = output_path
        self.graph = open(output_path, 'w')
        self.ontology = {'kgfarm': 'http://kgfarm.com/type/'}
        self.triples = []

    def dump_triples(self):
        self.graph.write('\n'.join(self.triples))

    def annotate_feature_views(self):
        triple = '<{}> <{}> "feature view {}".'
        table_ids = get_table_ids(self.config)['Table_id'].tolist()
        count = 1
        for table_id in table_ids:
            self.triples.append(triple.format(table_id, self.ontology.get('kgfarm')+'FeatureView', count))
            count = count + 1

        self.dump_triples()




def main():
    start = time()
    builder = Builder(port=5820, database='kgfarm_dev')
    builder.annotate_feature_views()
    print('\n• graph generated in :', time_taken(start, time()))


if __name__ == "__main__":
    main()

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm import tqdm
from datetime import datetime
from operations.template import *
from helpers.helper import connect_to_stardog


class Governor:
    def __init__(self, config=None, graph_path: str = 'farm.ttl', lids_db: str = 'Niki-kaggle-fixed', port: int = 5820,
                 show_connection_status: bool = False):
        if config is None:  # for graph generation
            self.config = connect_to_stardog(port=port, db=lids_db, show_status=show_connection_status)
        else:  # for KGFarm APIs (use the db config coming from user)
            self.config = config
        self.graph_path = graph_path
        self.graph = open(self.graph_path, 'w')
        self.graph.write('# Farm Graph generated on ' + str(datetime.now()) + '\n')
        self.triples = set()
        self.alpha = 0.90  # uniqueness for entity generation
        self.farm_ontology = {'prefix':   'http://kgfarm.com/ontology/',
                              'rdf-type': '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>',
                              'name':     '<http://schema.org/name>',
                              'certainty': '<http://kgfarm.com/ontology/withCertainty>'}

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
        def dump_triples():
            self.graph.write('\n'.join(self.triples))
            self.triples = set()

        def create_entity_names(physical_column_name: str):
            entity = physical_column_name.lower().replace('.', '').replace('_', '').replace('-', '').\
                replace(' ', '').replace('id', '').replace('name', '').replace('number', '').replace('code', '')
            return entity.capitalize()

        def elect_default_physical_representation(entity_df_info: pd.DataFrame):
            def only_one_default_representation_exists(df):
                if df.value_counts().get('hasDefaultEntity') == 1:  # one default physical representation
                    return True
                else:
                    return False

            max_uniqueness = max(list(entity_df_info['Column_uniqueness']))
            candidate_entity_type = []
            for candidate_info in entity_df_info.to_dict('index').values():
                if candidate_info['Column_uniqueness'] < max_uniqueness:
                    candidate_entity_type.append('hasOptionalEntity')
                else:
                    max_uniqueness = candidate_info['Column_uniqueness']
                    candidate_entity_type.append('hasDefaultEntity')
            entity_df_info['Entity_type'] = candidate_entity_type

            if only_one_default_representation_exists(df=entity_df_info):
                return entity_df_info
            else:
                """
                if multiple candidates have equal uniqueness, use PKFK classifier to end ties
                """
                # TODO: use pkfk for tie-breaker
                if only_one_default_representation_exists(df=entity_df_info):
                    return entity_df_info
                else:
                    entity_df_info['Entity_type'] = 'hasOptionalEntity'
                    entity_df_info.loc[list(entity_df_info.index)[0], 'Entity_type'] = 'hasDefaultEntity'
                    return entity_df_info

        """
        1. query LiDS graph to fetch column pairs with content similarity ()
        2. use PKFK classifier to get the column pairs (from 1) which have a primary key-foreign key relationship
        3. for all tables in LiDS graph:
            3.1 feature view generation
                I   generate a feature view node
                II  link the feature view node to physical table in LiDS
                III name the feature view
            3.2  entity generation
                I fetch any column in this table has uniqueness > 𝛼 (alpha) and no missing values else ∈ PKFK paris from 2:
                    i)   create an entity node
                    ii)  link the entity node to physical column in LiDS
                    iii) name the entity
                    iv)  link entity node to feature view node by deciding the relationship
        """

        # step 3
        digits = len(str(len(get_table_ids(self.config))))  # zero padding for clean feature view numbering
        table_ids = set(get_table_ids(config=self.config)['Table_id'])
        feature_view_count = 0
        for table_url in tqdm(table_ids):
            # 3.1 feature view generation
            feature_view_count = feature_view_count + 1
            zeroes_to_be_padded = digits - len(str(feature_view_count))
            feature_view = '0' * zeroes_to_be_padded + str(feature_view_count)
            # I generate a feature view node
            self.triples.add(f'<{self.farm_ontology.get("prefix")}feature-view-{feature_view}> {self.farm_ontology.get("rdf-type")} <{self.farm_ontology.get("prefix")}FeatureView>.')
            # II  link the feature view node to physical table in LiDS
            self.triples.add(f'<{table_url}> <{self.farm_ontology.get("prefix")}hasFeatureView> <{self.farm_ontology.get("prefix")}feature-view-{feature_view}>.')
            # III name the feature view
            self.triples.add(f'<{self.farm_ontology.get("prefix")}feature-view-{feature_view}> {self.farm_ontology.get("name")} "Feature view {feature_view}".')
            # 3.2 entity generation
            # TODO: use PKFK classifier, if default entity gives no result and result our obtained using pkfk entity_type = hasMultipleEntities
            # I fetch any column in this table has uniqueness > 𝛼 (alpha) and no missing values else ∈ PKFK paris from 2.
            entity_candidates_per_table = get_column_with_high_uniqueness_and_no_missing_values(config=self.config, table_url=table_url, alpha=self.alpha)
            if len(entity_candidates_per_table) == 0:
                continue  # feature view with no entity
            entity_candidates_per_table['Column_uniqueness'] = pd.to_numeric(entity_candidates_per_table['Column_uniqueness'])
            entity_candidates_per_table['Entity_name'] = entity_candidates_per_table['Column_name'].apply(lambda x: create_entity_names(x))
            entity_candidates_per_table = elect_default_physical_representation(
                entity_df_info=entity_candidates_per_table)

            entity_candidates_per_table = entity_candidates_per_table.to_dict('index')

            table_name = table_url.split('/')[-1].replace('.csv', '').upper()
            for column_url, entity_info in entity_candidates_per_table.items():
                entity_name = entity_info.get('Entity_name')
                column_name = entity_info.get('Column_name').replace(' ', '_').replace('.', '_').lower()
                uniqueness = entity_info.get('Column_uniqueness')
                entity_type = entity_info.get('Entity_type')

                # i) create an entity node
                self.triples.add(f'<{self.farm_ontology.get("prefix")}entity-{column_name}-{table_name}> {self.farm_ontology.get("rdf-type")} <{self.farm_ontology.get("prefix")}Entity>.')
                # ii)  link the entity node to physical column in LiDS
                self.triples.add(f'<{self.farm_ontology.get("prefix")}entity-{column_name}-{table_name}>  <{self.farm_ontology.get("prefix")}representedBy> <{column_url}>.')
                # iii)  name the entity
                self.triples.add(f'<{self.farm_ontology.get("prefix")}entity-{column_name}-{table_name}>  {self.farm_ontology.get("name")} "{entity_name}".')
                # iv) link entity node to feature view node by deciding the relationship
                self.triples.add(f'<<<{self.farm_ontology.get("prefix")}feature-view-{feature_view}> <{self.farm_ontology.get("prefix")}{entity_type}> <{self.farm_ontology.get("prefix")}entity-{column_name}-{table_name}>>> {self.farm_ontology.get("certainty")} "{"%.2f" % uniqueness}"^^xsd:double.')

            dump_triples()
        self.graph.close()


if __name__ == "__main__":
    governor = Governor()
    governor.build_farm_graph()

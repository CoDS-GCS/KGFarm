from tqdm import tqdm
from operations.template import *


class Governor:
    def __init__(self, config):
        self.config = config

    def drop_feature_view(self, drop: list):
        for feature_view_info in tqdm(drop):
            feature_view = feature_view_info['Feature_view']
            if exists(self.config, feature_view):
                drop_feature_view(self.config, feature_view)
            else:
                error = feature_view + " doesn't exists!"
                raise ValueError(error)
        print('Dropped ', [i['Feature_view'] for i in drop], 'successfully!')

    # TODO: investigate why after updating default entity, kgfarm.get_optional_representations() doesn't returns the removed default entity
    def update_entity(self, entity_updation_info: list):
        for updation_info in tqdm(entity_updation_info):
            feature_view = updation_info['Feature_view']
            entity = updation_info['Entity']
            column_physical_representation_to_be_inserted = updation_info['Optional_physical_representation']
            remove_current_physical_representation_of_an_entity(self.config, feature_view=feature_view)
            insert_current_physical_representation_of_an_entity(self.config, feature_view=feature_view,
                                   column=column_physical_representation_to_be_inserted, entity=entity)
            print("Entity '{}' uses '{}' as its physical representation".
                  format(entity, column_physical_representation_to_be_inserted))


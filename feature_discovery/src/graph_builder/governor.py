import pandas as pd

from operations.template import *


class Governor:
    def __init__(self, config):
        self.config = config

    def drop_feature_view(self, drop: list):
        for feature_view_info in drop:
            feature_view = feature_view_info['Feature_view']
            if exists(self.config, feature_view):
                drop_feature_view(self.config, feature_view)
            else:
                error = feature_view + " doesn't exists!"
                raise ValueError(error)
        print('Dropped ', [i['Feature_view'] for i in drop], 'successfully!')

    def update_entity(self, entity_updation_info: list):
        print('Governor under construction!')

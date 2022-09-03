from helpers.helper import connect_to_stardog
from operations.template import get_transformations_on_columns


class Recommender:
    """A classifier that calculates embeddings and recommends type of transformation"""

    def __init__(self, feature_type: str, port: int = 5820, database: str = 'recommender',
                 show_connection_status: bool = False):
        self.feature_type = feature_type
        self.config = connect_to_stardog(port, database, show_connection_status)

    def generate_modelling_data(self):
        transformations_on_columns = get_transformations_on_columns(self.config)
        print(transformations_on_columns)

    def fit(self):
        pass

    def evaluate(self):
        pass


if __name__ == '__main__':
    recommender = Recommender(feature_type='all')
    recommender.generate_modelling_data()

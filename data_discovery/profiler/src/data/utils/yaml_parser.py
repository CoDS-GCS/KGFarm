import yaml


class YamlParser:

    def __init__(self, path: str):
        with open(path) as configFile:
            self.data = yaml.load(configFile, Loader=yaml.FullLoader)
            self.datasetsInfo = []

    def process_config_file(self):
        self.datasetsInfo = self.data['datasets']

    def get_datasets_info(self) -> list:
        return self.datasetsInfo

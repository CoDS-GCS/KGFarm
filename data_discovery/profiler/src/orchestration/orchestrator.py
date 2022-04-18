import queue
import sys
import threading
sys.path.append('../')
from data.utils.yaml_parser import YamlParser
from orchestration.utils import create_sources_from_datasets
from orchestration.worker import Worker
from storage.elasticsearchDB import ElasticSearchDB


class Orchestrator:

    def __init__(self):
        self.tables = queue.Queue()  # will hold the different tables to be processed
        self.document_db = ElasticSearchDB()

    def create_tables(self, path: str):
        parser = YamlParser(path)
        parser.process_config_file()
        datasets = parser.get_datasets_info()
        create_sources_from_datasets(datasets, self.tables)

    def process_tables(self, num_threads: int):
        screenLock = threading.Lock()
        for i in range(num_threads):
            worker = Worker('Thread ' + str(i), self.tables, screenLock, self.document_db)
            worker.setDaemon(True)
            worker.start()
        self.tables.join()
        self.document_db.close_db()

    def get_remaining_tables(self):
        return self.tables.qsize()

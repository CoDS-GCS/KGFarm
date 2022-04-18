from elasticsearch import Elasticsearch, ElasticsearchException
from elasticsearch.helpers import parallel_bulk
from logger import Logger
from storage.i_documentDB import IDocumentDB
from storage.utils import serialize_rawData, serialize_profiles

logger = Logger(__name__).create_console_logger()


class ElasticSearchDB(IDocumentDB):

    def __init__(self, host: str = 'localhost', port: int = 9200):
        self.host = host
        self.port = port
        self.client = Elasticsearch([{'host': self.host, 'port': self.port}])

    def close_db(self):
        try:
            self.client.transport.close()
        except ElasticsearchException:
            logger.error('Could not close the connection')

    def store_data(self, rawData: list):
        try:
            s = parallel_bulk(self.client, serialize_rawData(rawData), request_timeout=30)
            tuple(s)
        except ElasticsearchException as es_exception:
            logger.error('Could not store the raw data because:\n' + str(es_exception))

    def store_profiles(self, profiles: list):
        try:
            s = parallel_bulk(self.client, serialize_profiles(profiles), request_timeout=30)
            tuple(s)
        except ElasticsearchException as es_exception:
            logger.error('Could not store the profiles because:\n' + str(es_exception))

    def delete_index(self, index: str):
        try:
            self.client.indices.delete(index=index, ignore=[400, 404])
        except ElasticsearchException as es_exception:
            logger.error(index + ' was not deleted because:\n' + str(es_exception))

    def count_per_index(self, index: str) -> float:
        self.client.indices.refresh(index)
        res = self.client.cat.count(index, params={"format": "json"})
        return int(res[0]['count'])

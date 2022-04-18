from abc import ABC, abstractmethod


class IDocumentDB(ABC):

    @abstractmethod
    def close_db(self):
        pass

    @abstractmethod
    def store_data(self, rawData: list):
        pass

    @abstractmethod
    def store_profiles(self, profiles: list):
        pass

    @abstractmethod
    def delete_index(self, index: str):
        pass

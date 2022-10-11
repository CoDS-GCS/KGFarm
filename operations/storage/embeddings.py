import faiss
import pickle
import numpy as np


class Embeddings:
    def __init__(self, path_to_embeddings: str = 'embedding_store/embeddings_120K.pickle'):

        self.numeric_embedding_size = 300  # DDE
        self.string_embedding_size = 512   # Minhash

        def load_embeddings():
            with open(path_to_embeddings, 'rb') as handle:
                return pickle.load(handle)

        def index_embeddings(embeddings: dict):
            numeric_index = faiss.IndexFlatL2(self.numeric_embedding_size)
            string_index = faiss.IndexFlatL2(self.string_embedding_size)
            numeric_embeddings = np.array(list(embeddings.get('numeric').keys()), dtype=np.float32)
            string_embeddings = np.array(list(embeddings.get('string').keys()), dtype=np.float32)
            numeric_index.add(numeric_embeddings)
            string_index.add(string_embeddings)
            return numeric_embeddings, string_embeddings, numeric_index, string_index

        self.embeddings = load_embeddings()
        self.numeric_embeddings, self.string_embeddings, self.numeric_index, self.string_index = \
            index_embeddings(embeddings=self.embeddings)

    def get_embeddings(self):
        return self.numeric_embeddings, self.string_embeddings

    def get_similar_columns(self, numeric_column_embeddings: dict, string_column_embeddings: dict, k: int = 1):

        def search_most_similar_numeric_column(e):
            e = np.array(e, dtype=np.float32).reshape(1, -1)
            _, c = self.numeric_index.search(e, k)
            return self.embeddings.get('numeric').get(tuple(self.numeric_embeddings[c[0][0]]))

        def search_most_similar_string_column(e):
            e = np.array(e, dtype=np.float32).reshape(1, -1)
            _, c = self.string_index.search(e, k)
            return self.embeddings.get('string').get(tuple(self.string_embeddings[c[0][0]]))

        similar_numeric_columns = {k: search_most_similar_numeric_column(v) for k, v in numeric_column_embeddings.items()}
        similar_string_columns = {k: search_most_similar_string_column(v) for k, v in string_column_embeddings.items()}

        similar_numeric_columns.update(similar_string_columns)
        return similar_numeric_columns

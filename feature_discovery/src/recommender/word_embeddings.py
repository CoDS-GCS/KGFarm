import os
import pickle
import re
import numpy as np
from urllib.parse import unquote_plus

from camelsplit import camelsplit


class WordEmbedding:
    def __init__(self):
        def initialize_and_normalize_word_embeddings(embedding_path: str):
            if not os.path.exists(embedding_path):
                embedding_path = embedding_path.replace('pickle', 'txt')  # initializing raw word embeddings
                with open(embedding_path, 'r') as f:
                    lines = [i.strip() for i in f.readlines()]
                vectors = {}
                for line in lines:
                    split = line.split()
                    word = split[0]
                    vector = np.array([float(i) for i in split[1:]])
                    vector /= np.linalg.norm(vector)  # normalize embeddings to unit length
                    vectors[word] = vector
                with open(embedding_path, 'wb') as f:
                    pickle.dump(vectors, f)

            else:
                with open(embedding_path, 'rb') as f:  # dump embeddings as a pickle file (loads faster)
                    vectors = pickle.load(f)
                    print(embedding_path, 'loaded.')
                return vectors

        self.embeddings = initialize_and_normalize_word_embeddings('utils/glove_embeddings/glove.6B.100d.pickle')

    def get_embeddings(self, column_id: str):
        column_name = unquote_plus(column_id).rsplit('/', 1)[-1]
        column_name = re.sub('[^0-9a-zA-Z]+', ' ', column_name)
        column_name = " ".join(camelsplit(column_name.strip()))
        column_name = re.sub('\s+', ' ', column_name.strip())
        column_name = column_name.lower()
        tokens = column_name.split()

        embeddings = []
        for token in tokens:
            embedding = self.embeddings.get(token)
            if embedding is not None:
                embeddings.append(embedding)

        if len(embeddings) == 0:
            return None
        else:
            sum_embedding = np.zeros(100)
            for embedding in embeddings:
                sum_embedding = sum_embedding + embedding
            return sum_embedding

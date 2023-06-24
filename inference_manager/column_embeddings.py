import torch
from torch import nn

EMBEDDING_SIZE = 300


class NumericalEmbeddingModel(nn.Module):
    def __init__(self, embedding_size):
        super(NumericalEmbeddingModel, self).__init__()
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(32, embedding_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        hidden1 = self.fc1(x)
        tanh1 = self.tanh(hidden1)
        hidden2 = self.fc2(tanh1)
        tanh2 = self.tanh(hidden2)
        hidden3 = self.fc3(tanh2)
        output = self.tanh(hidden3)
        return output


class StringEmbeddingModel(nn.Module):
    def __init__(self, embedding_size):
        super(StringEmbeddingModel, self).__init__()
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(50, embedding_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        hidden1 = self.fc1(x)
        tanh1 = self.tanh(hidden1)
        hidden2 = self.fc2(tanh1)
        tanh2 = self.tanh(hidden2)
        hidden3 = self.fc3(tanh2)
        output = self.tanh(hidden3)

        return output


def load_numeric_embedding_model(model_path: str):
    model = NumericalEmbeddingModel(EMBEDDING_SIZE)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


def load_embedding_model_for_cleaning(model_path: str, model_type: str):
    if model_type not in ['numerical', 'categorical']:
        raise ValueError("model_type must be 'numerical' or 'categorical'")

    if model_type == 'numerical':
        model = NumericalEmbeddingModel(EMBEDDING_SIZE)
    else:
        model = StringEmbeddingModel(EMBEDDING_SIZE)

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

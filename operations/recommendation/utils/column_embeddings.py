import torch
from torch import nn

EMBEDDING_SIZE = 300


# Define model: network h in the paper
class NumericalEmbeddingModel(nn.Module):
    def __init__(self, embedding_size):
        super(NumericalEmbeddingModel, self).__init__()
        self.embedding_size = embedding_size
        # layer dimensions: 32 -> 300 -> 300 -> 300 (if embedding size is 300)
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


def load_numeric_embedding_model(path: str = 'operations/recommendation/utils/models/20211123161253_numerical_embedding_model_epoch_4_3M_samples_gpu_cluster.pt'):
    model = NumericalEmbeddingModel(EMBEDDING_SIZE)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


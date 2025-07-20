import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class URLGNN(nn.Module):
    """
    Graph Neural Network for URL classification.
    Takes PyG Data objects (graph representations of URLs) as input.
    """
    def __init__(self, vocab_size, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = GCNConv(embed_dim, 128)
        self.conv2 = GCNConv(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, 1)

    def forward(self, data):
        x = self.embedding(data.x.squeeze(1))
        x = torch.relu(self.conv1(x, data.edge_index))
        x = torch.relu(self.conv2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        x = self.dropout(x)
        return torch.sigmoid(self.fc(x)).squeeze()

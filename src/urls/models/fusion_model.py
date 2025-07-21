# models/fusion_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


# -------------------- CharCNN Module --------------------
class CharCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, seq_len=200):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, 64)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # [B, E, L]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # [B, 128]
        x = self.fc(x)
        return x  # [B, 64]


# -------------------- GNN Module --------------------
class URLGNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = GCNConv(embed_dim, 128)
        self.conv2 = GCNConv(128, 64)

    def forward(self, data):
        x = self.embedding(data.x.squeeze(1))
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return x  # [B, 64]


# -------------------- Fusion Model --------------------
class FusionModel(nn.Module):
    """
    A multi-modal phishing detection model fusing character-level CNN features
    and graph-based GNN features from URLs.
    """

    def __init__(self, cnn_vocab_size, gnn_vocab_size):
        super().__init__()
        self.cnn = CharCNN(cnn_vocab_size)
        self.gnn = URLGNN(gnn_vocab_size)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, char_input, graph_data):
        x1 = self.cnn(char_input)
        x2 = self.gnn(graph_data)
        x = torch.cat([x1, x2], dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        return torch.sigmoid(self.fc2(x)).squeeze()

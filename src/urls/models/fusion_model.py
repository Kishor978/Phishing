# models/fusion_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from urls.models.charcnn_model import CharCNN # Import the CharCNN module
from urls.models.gnn_model import URLGNN     # Import the URLGNN module

class FusionModel(nn.Module):
    """
    A multi-modal phishing detection model fusing character-level CNN features
    and graph-based GNN features from URLs.
    """
    def __init__(self, cnn_vocab_size, gnn_vocab_size, cnn_embed_dim=64, gnn_embed_dim=64):
        super().__init__()
        # Initialize the individual sub-models
        self.cnn = CharCNN(cnn_vocab_size, embed_dim=cnn_embed_dim)
        self.gnn = URLGNN(gnn_vocab_size, embed_dim=gnn_embed_dim)

        # Fusion layers: concatenate outputs and pass through a shared MLP
        # Output of CharCNN's fc layer is 64, output of GNN's final conv is 64.
        # So, concatenated features will be 64 + 64 = 128
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 64) # Hidden layer after concatenation
        self.fc2 = nn.Linear(64, 1)   # Final output layer for binary classification

    def forward(self, char_input, graph_data):
        # Pass inputs through respective sub-models
        x_cnn = self.cnn(char_input) # Output: (batch_size, 64)
        x_gnn = self.gnn(graph_data) # Output: (batch_size, 64)

        # Concatenate the features from both models
        x_fused = torch.cat([x_cnn, x_gnn], dim=1) # Output: (batch_size, 128)

        # Pass through fusion layers
        x_fused = self.dropout(F.relu(self.fc1(x_fused)))
        # Final sigmoid activation for binary classification probability
        return torch.sigmoid(self.fc2(x_fused)).squeeze() # Squeeze to get (batch_size,)

if __name__ == '__main__':
    # Simple test for FusionModel
    cnn_vocab_size = 70
    gnn_vocab_size = 100
    model = FusionModel(cnn_vocab_size, gnn_vocab_size)
    print(model)

    # Create dummy inputs for a batch of 2
    dummy_char_input = torch.randint(low=1, high=cnn_vocab_size, size=(2, 200), dtype=torch.long)

    from torch_geometric.data import Data, Batch
    # Dummy graph data for two graphs in a batch
    data1 = Data(x=torch.tensor([[1], [2], [3]], dtype=torch.long),
                 edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t(),
                 y=torch.tensor([0], dtype=torch.float))
    data2 = Data(x=torch.tensor([[4], [5]], dtype=torch.long),
                 edge_index=torch.tensor([[0, 1]], dtype=torch.long).t(),
                 y=torch.tensor([1], dtype=torch.float))
    dummy_graph_data = Batch.from_data_list([data1, data2])

    output = model(dummy_char_input, dummy_graph_data)
    print(f"Fusion Model output shape: {output.shape}") # Should be (2,)
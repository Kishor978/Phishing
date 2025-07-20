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
        self.embedding = nn.Embedding(vocab_size, embed_dim) # Embed node features (token IDs)
        self.conv1 = GCNConv(embed_dim, 128) # First Graph Convolutional Layer
        self.conv2 = GCNConv(128, 64) # Second Graph Convolutional Layer
        # You had a dropout and fc layer here in gnn.ipynb, but for the GNN *module*
        # within a fusion model, it's common to output the pooled features.
        # The final classification head will be in the FusionModel.
        # If this GNN is used standalone, you might add:
        # self.dropout = nn.Dropout(0.3)
        # self.fc = nn.Linear(64, 1)

    def forward(self, data):
        # data is a Batch object (a batch of graphs)
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # x comes in as (num_nodes_in_batch, 1) so squeeze the last dim
        x = self.embedding(x.squeeze(1)) # (num_nodes_in_batch, embed_dim)

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Global pooling (e.g., global_mean_pool) aggregates node features for each graph in the batch
        # This results in one feature vector per graph
        x = global_mean_pool(x, batch) # (batch_size, 64)

        # If this GNN is used as a standalone model, uncomment and use these:
        # x = self.dropout(x)
        # return torch.sigmoid(self.fc(x)).squeeze()
        return x # Return pooled graph features

if __name__ == '__main__':
    # Simple test for URLGNN
    from torch_geometric.data import Data, Batch
    vocab_size = 100 # Example GNN vocabulary size
    model = URLGNN(vocab_size=vocab_size)
    print(model)

    # Create dummy graph data (simulating a batch of 2 graphs)
    # Graph 1: nodes 1,2,3 with edge (1,2), (2,3)
    data1 = Data(x=torch.tensor([[1], [2], [3]], dtype=torch.long),
                 edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t(),
                 y=torch.tensor([0], dtype=torch.float))
    # Graph 2: nodes 4,5 with edge (4,5)
    data2 = Data(x=torch.tensor([[4], [5]], dtype=torch.long),
                 edge_index=torch.tensor([[0, 1]], dtype=torch.long).t(),
                 y=torch.tensor([1], dtype=torch.float))

    batch = Batch.from_data_list([data1, data2])
    output = model(batch)
    print(f"Output shape: {output.shape}") # Should be (2, 64) for a batch of 2 graphs
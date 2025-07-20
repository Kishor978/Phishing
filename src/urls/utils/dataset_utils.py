import torch
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset
from urls.utils.urls_preprocessing import build_char_vocab_from_df, build_vocab_gnn,encode_url_char
from urls.utils.fusion_model_utils import tokenize_char_url, url_to_graph

class URLCharDataset(Dataset):
    """
    Dataset for CharCNN model, encoding URLs as character sequences.
    """
    def __init__(self, df):
        self.urls = df['text'].astype(str).values
        self.labels = df['label'].values

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        url = self.urls[idx]
        # Use the encoding function from url_processing
        x = torch.tensor(encode_url_char(url), dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float)
        return x, y

class URLGraphDataset(InMemoryDataset):
    """
    Dataset for GNN model, converting URLs to graph structures.
    Uses InMemoryDataset for efficient batching of small graphs.
    """
    def __init__(self, df, vocab):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        # The InMemoryDataset expects data to be loaded/processed on initialization
        # The 'root' argument is usually where processed data is cached.
        # For small datasets, '.' is fine; for larger, specify a cache directory.
        super().__init__('.')
        # Filter out URLs that cannot be converted to graphs (e.g., less than 2 nodes)
        data_list = [
            url_to_graph(url, label, vocab)
            for url, label in zip(self.df['text'], self.df['label'])
            if url_to_graph(url, label, vocab) is not None
        ]
        if not data_list:
            raise ValueError("No valid graphs could be generated from the dataset.")
        # Collate method prepares data for PyG's DataLoader
        self.data, self.slices = self.collate(data_list)

class URLFusionDataset(Dataset):
    """
    Dataset for the Fusion Model, returning both character-encoded URL and graph data.
    """
    def __init__(self, df, char_vocab, graph_vocab, max_len=200):
        self.data = []
        for _, row in df.iterrows():
            char_tensor = tokenize_char_url(row['text'], char_vocab, max_len)
            graph_data = url_to_graph(row['text'], row['label'], graph_vocab)
            if graph_data is not None:
                self.data.append((char_tensor, graph_data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
if __name__ == '__main__':
    # Example usage (requires a dummy DataFrame)

    dummy_df = pd.DataFrame({
        'text': ["https://www.example.com/login", "http://phishing.site/verify?id=123", "https://google.com"],
        'label': [0, 1, 0]
    })

    print("--- Testing URLCharDataset ---")
    char_dataset = URLCharDataset(dummy_df)
    print(f"CharDataset length: {len(char_dataset)}")
    x_char, y_char = char_dataset[0]
    print(f"First Char entry: x.shape={x_char.shape}, y={y_char}")

    print("\n--- Testing URLGraphDataset ---")
    gnn_vocab = build_vocab_gnn(dummy_df)
    try:
        graph_dataset = URLGraphDataset(dummy_df, gnn_vocab)
        print(f"GraphDataset length: {len(graph_dataset)}")
        graph_data_sample = graph_dataset[0]
        print(f"First Graph entry: x.shape={graph_data_sample.x.shape}, edge_index.shape={graph_data_sample.edge_index.shape}, y={graph_data_sample.y}")
    except ValueError as e:
        print(f"Could not create GraphDataset: {e}")


    print("\n--- Testing URLFusionDataset ---")
    char_vocab = build_char_vocab_from_df(dummy_df)
    gnn_vocab_fusion = build_vocab_gnn(dummy_df) # Use a separate instance if needed
    try:
        fusion_dataset = URLFusionDataset(dummy_df, char_vocab, gnn_vocab_fusion)
        print(f"FusionDataset length: {len(fusion_dataset)}")
        char_input, graph_data_fusion = fusion_dataset[0]
        print(f"First Fusion entry: Char input shape={char_input.shape}, Graph x.shape={graph_data_fusion.x.shape}")
    except ValueError as e:
        print(f"Could not create FusionDataset: {e}")
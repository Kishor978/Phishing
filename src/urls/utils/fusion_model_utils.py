from collections import Counter
import torch
from urllib.parse import urlparse
from torch_geometric.data import Data

# -------------------- Tokenizer and Vocab Builder --------------------
def build_char_vocab(df, max_len=200):
    chars = Counter()
    for url in df['text']:
        chars.update(url[:max_len])
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for i, (ch, _) in enumerate(chars.most_common(), start=2):
        vocab[ch] = i
    return vocab

def tokenize_char_url(url, vocab, max_len=200):
    tokens = [vocab.get(c, vocab['<UNK>']) for c in url[:max_len]]
    if len(tokens) < max_len:
        tokens += [vocab['<PAD>']] * (max_len - len(tokens))
    return torch.tensor(tokens)

def tokenize_graph_url(url):
    parsed = urlparse(url)
    tokens = []
    if parsed.scheme:
        tokens.append(parsed.scheme)
    if parsed.hostname:
        tokens += parsed.hostname.split('.')
    if parsed.path:
        tokens += parsed.path.strip("/").split("/")
    return tokens[:30]

def build_graph_vocab(df):
    token_counter = Counter()
    for url in df['text']:
        token_counter.update(tokenize_graph_url(url))
    vocab = {'<UNK>': 0}
    for i, (tok, _) in enumerate(token_counter.items(), start=1):
        vocab[tok] = i
    return vocab

def url_to_graph(url, label, vocab):
    tokens = tokenize_graph_url(url)
    node_ids = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    if len(node_ids) < 2:
        return None
    edge_index = torch.tensor([[i, i+1] for i in range(len(node_ids)-1)], dtype=torch.long).t()
    x = torch.tensor(node_ids, dtype=torch.long).unsqueeze(1)
    return Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.float))

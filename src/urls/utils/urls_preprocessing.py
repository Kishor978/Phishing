import pandas as pd
import torch
import string
from urllib.parse import urlparse, parse_qs
from collections import Counter
from torch_geometric.data import Data
import tldextract
import ipaddress


# --- Character-level Constants for CharCNN ---
CHAR_VOCAB = list(string.ascii_letters + string.digits + string.punctuation)
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHAR_VOCAB)}  # +1 for padding_idx=0
VOCAB_SIZE_CHAR = len(CHAR2IDX) + 1
MAX_URL_LEN = 200 # Max URL length for CharCNN (pad/truncate)

def encode_url_char(url, max_len=MAX_URL_LEN, char2idx=CHAR2IDX):
    """
    Encodes a URL into a sequence of character indices for CharCNN.
    Pads or truncates URLs to max_len.
    """
    url = url[:max_len].ljust(max_len)  # Truncate and pad right
    # Unknown chars get 0 (padding_idx)
    return [char2idx.get(c, 0) for c in url]

def build_char_vocab_from_df(df, max_len=MAX_URL_LEN):
    """
    Builds a character vocabulary from a DataFrame's 'text' column.
    """
    chars = Counter()
    for url in df['text']:
        chars.update(url[:max_len])
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for i, (ch, _) in enumerate(chars.most_common(), start=2):
        vocab[ch] = i
    return vocab

# --- Tokenization and Vocab for GNN ---
def tokenize_url_gnn(url):
    """
    Tokenizes a URL into components for GNN graph construction.
    Includes scheme, hostname parts, path parts, and query parameter keys.
    """
    parsed = urlparse(url)
    tokens = []
    if parsed.scheme:
        tokens.append(parsed.scheme)
    if parsed.hostname:
        tokens += parsed.hostname.split('.')
    if parsed.path:
        tokens += parsed.path.strip("/").split("/")
    if parsed.query:
        tokens += list(parse_qs(parsed.query).keys())
    return tokens

def build_vocab_gnn(df, max_vocab_size=5000):
    """
    Builds a word/token vocabulary for GNN based on tokenized URL components.
    """
    token_counts = Counter()
    for url in df['text']:
        tokens = tokenize_url_gnn(url)
        token_counts.update(tokens)
    vocab = {'<PAD>': 0, '<UNK>': 1} # Reserve 0 for padding, 1 for unknown
    for i, (token, _) in enumerate(token_counts.most_common(max_vocab_size), start=2):
        vocab[token] = i
    return vocab

def url_to_graph(url, label, vocab, max_nodes=30):
    """
    Converts a URL and its label into a PyG Data object (graph).
    Nodes are token IDs, edges are sequential.
    """
    tokens = tokenize_url_gnn(url)[:max_nodes]
    node_ids = [vocab.get(tok, vocab['<UNK>']) for tok in tokens]
    num_nodes = len(node_ids)

    # A graph needs at least 2 nodes to form an edge in a simple sequential graph
    if num_nodes < 2:
        return None

    # Create sequential edges: (0,1), (1,2), ..., (n-2, n-1)
    edge_index = torch.tensor([[i, i + 1] for i in range(num_nodes - 1)], dtype=torch.long).t()
    x = torch.tensor(node_ids, dtype=torch.long).unsqueeze(1) # Node features (token IDs)
    y = torch.tensor([label], dtype=torch.float) # Graph-level label

    return Data(x=x, edge_index=edge_index, y=y)

# --- Feature extraction for Baseline (TF-IDF + Lexical) ---
def extract_safe_url_features(url):
    """
    Extracts a set of hand-crafted lexical and statistical features from a URL.
    This function corresponds to the commented-out feature extraction in baseline.ipynb.
    """
    ext = tldextract.extract(url)
    domain = ext.domain + '.' + ext.suffix
    subdomain = ext.subdomain
    try:
        ipaddress.ip_address(domain)
        is_domain_ip = 1
    except ValueError:
        is_domain_ip = 0

    letters = sum(c.isalpha() for c in url)
    digits = sum(c.isdigit() for c in url)
    specials = sum(not c.isalnum() for c in url) # Counts punctuation and other non-alphanumeric

    # Avoid division by zero for empty URLs
    url_len = len(url)
    letter_ratio = round(letters / url_len, 3) if url_len > 0 else 0.0
    digit_ratio = round(digits / url_len, 3) if url_len > 0 else 0.0
    special_char_ratio = round(specials / url_len, 3) if url_len > 0 else 0.0

    return {
        "URLLength": url_len,
        "DomainLength": len(domain),
        "TLDLength": len(ext.suffix),
        "NoOfSubDomain": subdomain.count('.') + (1 if subdomain else 0),
        "IsDomainIP": is_domain_ip,
        "NoOfLettersInURL": letters,
        "NoOfDegitsInURL": digits,
        "LetterRatioInURL": letter_ratio,
        "DegitRatioInURL": digit_ratio,
        "NoOfEqualsInURL": url.count('='),
        "NoOfQMarkInURL": url.count('?'),
        "NoOfAmpersandInURL": url.count('&'),
        "SpacialCharRatioInURL": special_char_ratio,
        "Bank": int("bank" in url.lower()),
        "Pay": int("pay" in url.lower()),
        "Crypto": int("crypto" in url.lower())
    }

if __name__ == '__main__':
    # Simple test for character tokenization
    test_url_char = "https://example.com/login"
    encoded_char = encode_url_char(test_url_char)
    print(f"Char encoded for '{test_url_char}': {encoded_char[:10]}...")

    # Simple test for GNN tokenization and graph conversion
    dummy_df = pd.DataFrame({'text': ["https://www.google.com/search?q=phishing", "http://malicious.net/login.php"], 'label': [0, 1]})
    gnn_vocab = build_vocab_gnn(dummy_df)
    print(f"GNN Vocab size: {len(gnn_vocab)}")
    graph_data = url_to_graph("https://www.google.com/search?q=phishing", 0, gnn_vocab)
    if graph_data:
        print(f"GNN Graph nodes (x): {graph_data.x.tolist()}")
        print(f"GNN Graph edges (edge_index): {graph_data.edge_index.tolist()}")

    # Simple test for safe URL features
    features = extract_safe_url_features("https://www.paypal.com/signin")
    print(f"Safe URL features: {features}")
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
from email_classification.preprocessing import clean_text, preprocess_text_spacy
from email_classification.email_utils import setup_logging

logger = setup_logging()


class EmailDataset(Dataset):
    """
    PyTorch Dataset for email text classification.
    Handles tokenization, vocabulary building, and padding.
    """

    def __init__(self, df, text_col="text", label_col="label", max_len=500, vocab=None):
        self.texts = df[text_col].tolist()
        self.labels = df[label_col].tolist()
        self.max_len = max_len
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}  # Reserve 0 for padding, 1 for unknown
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

        if vocab is None:
            self.vocab = self._build_vocab()
        else:
            self.vocab = vocab
            self._update_word_idx_maps()  # Ensure word2idx/idx2word are updated from provided vocab

        logger.info(
            f"EmailDataset created. Vocab size: {len(self.vocab)}. Max length: {self.max_len}"
        )

    def _build_vocab(self):
        """Builds vocabulary from texts."""
        logger.info("Building vocabulary...")
        all_tokens = []
        for text in self.texts:
            processed_text = preprocess_text_spacy(clean_text(text))
            all_tokens.extend(processed_text.split())

        token_counts = Counter(all_tokens)
        # Sort by frequency and assign indices
        sorted_vocab = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

        for word, _ in sorted_vocab:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        logger.info(f"Vocabulary built. Size: {len(self.word2idx)}")
        return self.word2idx  # Store word2idx as vocab

    def _update_word_idx_maps(self):
        """Updates word2idx and idx2word based on an external vocabulary."""
        self.word2idx = self.vocab
        self.idx2word = {idx: word for word, idx in self.vocab.items()}
        # Ensure PAD and UNK are correctly set even if not in external vocab
        if "<PAD>" not in self.word2idx:
            self.word2idx["<PAD>"] = 0
            self.idx2word[0] = "<PAD>"
        if "<UNK>" not in self.word2idx:
            self.word2idx["<UNK>"] = 1
            self.idx2word[1] = "<UNK>"

    def _tokenize_and_encode(self, text):
        """Tokenizes, preprocesses, and encodes text to numerical IDs."""
        processed_text = preprocess_text_spacy(clean_text(text))
        tokens = processed_text.split()
        encoded = [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]
        return encoded

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoded_text = self._tokenize_and_encode(text)

        # Pad or truncate
        if len(encoded_text) < self.max_len:
            padded_text = encoded_text + [self.word2idx["<PAD>"]] * (
                self.max_len - len(encoded_text)
            )
        else:
            padded_text = encoded_text[: self.max_len]

        return torch.tensor(padded_text, dtype=torch.long), torch.tensor(
            label, dtype=torch.float
        )


class DualEncoderEmailDataset(Dataset):
    """
    PyTorch Dataset for dual encoder email models.
    Handles tokenization, vocabulary building, and padding for subject and body.
    """

    def __init__(
        self,
        df,
        subject_col="subject",
        body_col="body",
        label_col="label",
        max_len_subject=50,
        max_len_body=500,
        vocab=None,
    ):
        self.subjects = df[subject_col].tolist()
        self.bodies = df[body_col].tolist()
        self.labels = df[label_col].tolist()
        self.max_len_subject = max_len_subject
        self.max_len_body = max_len_body
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

        if vocab is None:
            self.vocab = self._build_vocab()
        else:
            self.vocab = vocab
            self._update_word_idx_maps()

        logger.info(f"DualEncoderEmailDataset created. Vocab size: {len(self.vocab)}.")
        logger.info(
            f"Max length subject: {self.max_len_subject}, Max length body: {self.max_len_body}."
        )

    def _build_vocab(self):
        """Builds vocabulary from both subject and body texts."""
        logger.info("Building vocabulary for dual encoder...")
        all_tokens = []
        for text in self.subjects + self.bodies:  # Combine for vocab building
            processed_text = preprocess_text_spacy(clean_text(text))
            all_tokens.extend(processed_text.split())

        token_counts = Counter(all_tokens)
        sorted_vocab = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

        for word, _ in sorted_vocab:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        logger.info(f"Dual encoder vocabulary built. Size: {len(self.word2idx)}")
        return self.word2idx

    def _update_word_idx_maps(self):
        """Updates word2idx and idx2word based on an external vocabulary."""
        self.word2idx = self.vocab
        self.idx2word = {idx: word for word, idx in self.vocab.items()}
        # Ensure PAD and UNK are correctly set
        if "<PAD>" not in self.word2idx:
            self.word2idx["<PAD>"] = 0
            self.idx2word[0] = "<PAD>"
        if "<UNK>" not in self.word2idx:
            self.word2idx["<UNK>"] = 1
            self.idx2word[1] = "<UNK>"

    def _tokenize_and_encode(self, text, max_len):
        """Tokenizes, preprocesses, encodes, and pads/truncates text."""
        processed_text = preprocess_text_spacy(clean_text(text))
        tokens = processed_text.split()
        encoded = [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]

        if len(encoded) < max_len:
            padded_text = encoded + [self.word2idx["<PAD>"]] * (max_len - len(encoded))
        else:
            padded_text = encoded[:max_len]
        return padded_text

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_text = self.subjects[idx]
        body_text = self.bodies[idx]
        label = self.labels[idx]

        encoded_subject = self._tokenize_and_encode(subject_text, self.max_len_subject)
        encoded_body = self._tokenize_and_encode(body_text, self.max_len_body)

        return (
            torch.tensor(encoded_subject, dtype=torch.long),
            torch.tensor(encoded_body, dtype=torch.long),
            torch.tensor(label, dtype=torch.float),
        )

import torch
import torch.nn as nn
import torch.nn.functional as F

class CharCNN(nn.Module):
    """
    Character-level Convolutional Neural Network for URL classification.
    Takes character-encoded URLs as input.
    """
    def __init__(self, vocab_size, embed_dim=64):
        super().__init__()
        # padding_idx=0 means that the embedding for index 0 (our <PAD> token) will be zero
        # and won't contribute to gradients.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Using 1D convolutions for sequence data
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2)
        # self.relu = nn.ReLU() # F.relu is often used directly in forward
        self.pool = nn.AdaptiveMaxPool1d(1) # Pools across the sequence dimension, outputting 1 value per channel

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 1) # Output a single logit for binary classification
        # self.sigmoid = nn.Sigmoid() # Applied in the training loop with BCELoss

    def forward(self, x):
        # x shape: (batch_size, sequence_length) -> e.g., (64, 200)
        x = self.embedding(x) # (batch_size, sequence_length, embed_dim) -> e.g., (64, 200, 64)
        x = x.permute(0, 2, 1) # (batch_size, embed_dim, sequence_length) for Conv1d -> e.g., (64, 64, 200)

        x = F.relu(self.conv1(x)) # (batch_size, 128, sequence_length)
        x = self.pool(x).squeeze(-1) # (batch_size, 128, 1) -> (batch_size, 128)

        x = self.dropout(x)
        return self.fc(x) # Return logits, sigmoid will be applied by BCELossWithLogits or in train loop

if __name__ == '__main__':
    # Simple test for CharCNN
    vocab_size = 70 # Example vocabulary size
    model = CharCNN(vocab_size=vocab_size)
    print(model)

    # Create dummy input
    dummy_input = torch.randint(low=1, high=vocab_size, size=(4, 200), dtype=torch.long) # Batch size 4, seq len 200
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Should be (4, 1)
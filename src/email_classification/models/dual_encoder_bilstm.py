import torch
import torch.nn as nn
from email_classification.model_config import (
    DUAL_BILSTM_EMBED_DIM, DUAL_BILSTM_HIDDEN_DIM, DUAL_BILSTM_NUM_LAYERS,
    DUAL_BILSTM_DROPOUT, DUAL_BILSTM_BIDIRECTIONAL,
    DUAL_BILSTM_FUSION_INPUT_DIM, DUAL_BILSTM_FUSION_HIDDEN_DIM, DUAL_BILSTM_FUSION_DROPOUT
)

class BiLSTMEncoder(nn.Module):
    """
    A single BiLSTM encoder for either subject or body.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, bidirectional):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        
        # Get the final hidden state from the last layer, concatenating directions
        if self.num_directions == 2:
            return self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            return self.dropout(hidden[-1, :, :])

class DualEncoderFusion(nn.Module):
    """
    Dual Encoder model using BiLSTMs for subject and body,
    followed by a fusion layer.
    """
    def __init__(self, vocab_size,
                 embed_dim=DUAL_BILSTM_EMBED_DIM,
                 hidden_dim=DUAL_BILSTM_HIDDEN_DIM,
                 num_layers=DUAL_BILSTM_NUM_LAYERS,
                 dropout=DUAL_BILSTM_DROPOUT,
                 bidirectional=DUAL_BILSTM_BIDIRECTIONAL,
                 fusion_hidden_dim=DUAL_BILSTM_FUSION_HIDDEN_DIM,
                 fusion_dropout=DUAL_BILSTM_FUSION_DROPOUT):
        super().__init__()

        # Both encoders share the same vocabulary and embedding layer
        # They are separate instances to process subject and body independently
        self.subject_encoder = BiLSTMEncoder(vocab_size, embed_dim, hidden_dim,
                                             num_layers, dropout, bidirectional)
        self.body_encoder = BiLSTMEncoder(vocab_size, embed_dim, hidden_dim,
                                          num_layers, dropout, bidirectional)

        # Fusion layer
        # Input dimension: (hidden_dim * num_directions) for subject + (hidden_dim * num_directions) for body
        self.fusion_fc1 = nn.Linear(DUAL_BILSTM_FUSION_INPUT_DIM, fusion_hidden_dim)
        self.fusion_dropout = nn.Dropout(fusion_dropout)
        self.fusion_fc2 = nn.Linear(fusion_hidden_dim, 1) # Output a single logit

    def forward(self, subject_input, body_input):
        # Encode subject and body
        subject_features = self.subject_encoder(subject_input) # (batch_size, hidden_dim * num_directions)
        body_features = self.body_encoder(body_input)       # (batch_size, hidden_dim * num_directions)

        # Concatenate features from both encoders
        fused_features = torch.cat((subject_features, body_features), dim=1)

        # Pass through fusion layers
        fused_output = self.fusion_dropout(torch.relu(self.fusion_fc1(fused_features)))
        return self.fusion_fc2(fused_output) # Return logits
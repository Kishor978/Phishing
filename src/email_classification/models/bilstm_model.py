# email_classification/src/models/bilstm_model.py
import torch
import torch.nn as nn
from email_classification.model_config import BILSTM_EMBED_DIM, BILSTM_HIDDEN_DIM, BILSTM_NUM_LAYERS, BILSTM_DROPOUT, BILSTM_BIDIRECTIONAL

class BiLSTMClassifier(nn.Module):
    """
    A Bidirectional LSTM (BiLSTM) model for email text classification.
    """
    def __init__(self, vocab_size, embed_dim=BILSTM_EMBED_DIM, hidden_dim=BILSTM_HIDDEN_DIM,
                 num_layers=BILSTM_NUM_LAYERS, dropout=BILSTM_DROPOUT, bidirectional=BILSTM_BIDIRECTIONAL):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=bidirectional, dropout=dropout, batch_first=True)
        
        # Calculate output dimension of LSTM
        # If bidirectional, hidden_dim * 2. If single layer, just hidden_dim.
        # This model uses the *last* hidden state (or concatenated for bidirectional).
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, 1) # Output a single logit

    def forward(self, text):
        # text shape: (batch_size, sequence_length)
        embedded = self.embedding(text) # embedded shape: (batch_size, sequence_length, embed_dim)
        
        # Pass through LSTM
        # output: (batch_size, sequence_length, hidden_dim * num_directions)
        # hidden, cell: (num_layers * num_directions, batch_size, hidden_dim)
        output, (hidden, cell) = self.lstm(embedded)
        
        # Use the final hidden state for classification
        # For bidirectional LSTM, hidden state is (num_layers * 2, batch_size, hidden_dim)
        # We take the last layer's forward and backward hidden states and concatenate them.
        # hidden[-2, :, :] is the last forward hidden state
        # hidden[-1, :, :] is the last backward hidden state
        if self.lstm.bidirectional:
            hidden_final = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden_final = self.dropout(hidden[-1, :, :])
            
        return self.fc(hidden_final) # Return logits
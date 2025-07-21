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
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, 1) # Output a single logit

    def forward(self, text):
        embedded = self.embedding(text) # embedded shape: (batch_size, sequence_length, embed_dim)
        
        output, (hidden, cell) = self.lstm(embedded)
        
        if self.lstm.bidirectional:
            hidden_final = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden_final = self.dropout(hidden[-1, :, :])
            
        return self.fc(hidden_final) # Return logits
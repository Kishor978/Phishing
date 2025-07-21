# email_classification/src/models/dual_encoder_transformer.py
import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, encoder_outputs):
        weights = self.attn(encoder_outputs).squeeze(-1)
        weights = torch.softmax(weights, dim=1)
        context = torch.sum(encoder_outputs * weights.unsqueeze(-1), dim=1)
        return context

# --------------------- Dual Encoder with Attention Fusion ---------------------
class DualEncoderAttentionFusion(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm_subj = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.lstm_body = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attn_subj = Attention(hidden_dim)
        self.attn_body = Attention(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_dim * 4, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, subj, body):
        subj_embed = self.embedding(subj)
        body_embed = self.embedding(body)

        subj_out, _ = self.lstm_subj(subj_embed)
        body_out, _ = self.lstm_body(body_embed)

        subj_ctx = self.attn_subj(subj_out)
        body_ctx = self.attn_body(body_out)

        fusion = torch.cat((subj_ctx, body_ctx), dim=1)
        fusion = self.dropout(fusion)
        fusion = torch.relu(self.fc1(fusion))
        return torch.sigmoid(self.fc2(fusion)).squeeze()


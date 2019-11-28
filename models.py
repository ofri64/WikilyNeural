import torch
import torch.nn as nn
from datasets import TokenMapper
from configs import BiLSTMConfig


class BiLSTM(nn.Module):
    def __init__(self, config: BiLSTMConfig, mapper: TokenMapper):
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.tokens_dim = mapper.get_tokens_dim()
        self.labels_dim = mapper.get_labels_dim()
        self.hidden_dim = config.hidden_dim
        self.padding_index = 0 if mapper.with_padding else None

        # layers
        self.embedding = nn.Embedding(self.tokens_dim, self.embedding_dim, padding_idx=self.padding_index)
        self.bi_lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(in_features=2 * self.hidden_dim, out_features=self.labels_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        embeddings = self.embedding(x)
        hidden, _ = self.bi_lstm(embeddings)  # hidden is of shape (seq, batch, 2 * hidden_dim)
        hidden = hidden.permute(1, 0, 2)  # transpose to Batch, Sequence, Features
        y_hat = self.linear(hidden)

        return y_hat

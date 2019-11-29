import torch
import torch.nn as nn
from datasets import TokenMapper
from configs import BiLSTMConfig


class BaseModel(nn.Module):

    def serialize_model(self) -> dict:
        return self.state_dict()

    def deserialize_model(self, model_state: dict) -> None:
        self.load_state_dict(model_state)


class BiLSTM(BaseModel):

    def __init__(self, config: BiLSTMConfig, mapper: TokenMapper):
        super().__init__()
        self.config = config
        self.mapper = mapper
        embedding_dim = config.embedding_dim
        tokens_dim = mapper.get_tokens_dim()
        labels_dim = mapper.get_labels_dim()
        hidden_dim = config.hidden_dim
        padding_index = 0 if mapper.with_padding else None

        # layers
        self.embedding = nn.Embedding(tokens_dim, embedding_dim, padding_idx=padding_index)
        self.bi_lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(in_features=2 * hidden_dim, out_features=labels_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        embeddings = self.embedding(x)
        hidden, _ = self.bi_lstm(embeddings)  # hidden is of shape (batch, seq, 2 * hidden_dim)
        y_hat = self.linear(hidden)
        y_hat = y_hat.permute(0, 2, 1)  # transpose to Batch, Features, Sequence (needed for CE loss)

        return y_hat

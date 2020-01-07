import torch
import torch.nn as nn
from configs import ModelConfig, RNNConfig
from mappers import BaseMapper, BaseMapperWithPadding


class BaseModel(nn.Module):

    def __init__(self, config: ModelConfig, mapper: BaseMapper):
        super().__init__()
        self.config = config
        self.mapper = mapper

    def serialize_model(self) -> dict:
        model_name = self.__class__.__name__
        config_class_name = self.config.__class__.__name__
        mapper_class_name = self.mapper.__class__.__name__

        config_params = self.config.to_dict()
        model_state = self.state_dict()
        mapper_state = self.mapper.serialize()

        model_state = {
            "model": {"name": model_name, "state": model_state},
            "config": {"name": config_class_name, "state": config_params},
            "mapper": {"name": mapper_class_name, "state": mapper_state},
        }

        return model_state


class BasicBiLSTM(BaseModel):
    def __init__(self, config: RNNConfig, mapper: BaseMapperWithPadding):
        super().__init__(config, mapper)
        self.tokens_dim = mapper.get_tokens_dim()
        self.labels_dim = mapper.get_labels_dim()
        self.padding_idx = mapper.get_padding_index()
        self.embedding_dim = config["embedding_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.embedding = nn.Embedding(self.tokens_dim, self.embedding_dim, padding_idx=self.padding_idx)
        self.LSTM = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim,
                            num_layers=2, bidirectional=True, dropout=0.5, batch_first=True)
        self.linear = nn.Linear(in_features=self.hidden_dim * 2, out_features=self.labels_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.embedding(x)
        rnn_features, _ = self.LSTM(x)
        # RNN outputs has dimensions batch, sequence_length, features (features is num_directions * hidden dim)

        y_hat = self.linear(rnn_features)

        # Cross Entropy loss expects dimensions of type batch, features, sequence
        y_hat = y_hat.permute(0, 2, 1)
        return y_hat

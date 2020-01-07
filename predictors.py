from typing import Tuple

import torch

from mappers import BaseMapper, BaseMapperWithPadding


class BasePredictor(object):

    def __init__(self, mapper: BaseMapper):
        self.mapper = mapper

    def infer_model_outputs(self, model_outputs: torch.tensor) -> torch.tensor:
        raise NotImplementedError("A class deriving from BasePredictor must implement infer_model_outputs method")

    def infer_sample(self, model: torch.nn.Module, tokens_indices: torch.tensor):
        model_outputs = model(tokens_indices)
        raise self.infer_model_outputs(model_outputs)

    def infer_raw_sample(self, model: torch.nn.Module, raw_sample: list):
        sample_tokens = []
        for sample in raw_sample:
            token_indices = [self.mapper.get_token_idx(token) for token in sample]
            sample_tokens.append(token_indices)

        sample_tokens = torch.tensor(sample_tokens)
        return self.infer_sample(model, sample_tokens)

    def infer_model_outputs_with_gold_labels(self, model_outputs: torch.tensor, labels: torch.tensor) -> Tuple[int, int]:
        num_correct: int
        num_predictions: int

        num_predictions = len(labels)
        predictions: torch.tensor = self.infer_model_outputs(model_outputs)
        correct_predictions: torch.tensor = (predictions == labels).type(torch.int64)
        num_correct = torch.sum(correct_predictions).item()

        return num_correct, num_predictions


class GreedyLSTMPredictor(BasePredictor):
    def __init__(self, mapper: BaseMapperWithPadding):
        super().__init__(mapper)

    def infer_model_outputs(self, model_outputs: torch.tensor) -> torch.Tensor:
        # dimension of model outputs is batch_size, num_features, sequence_length
        # that is why we are using max on dimension 1 - the features dimension
        _, labels_tokens = torch.max(model_outputs, dim=1)
        return labels_tokens

    def infer_model_outputs_with_gold_labels(self, model_outputs: torch.tensor, labels: torch.tensor) -> Tuple[int, int]:
        num_correct: int
        num_predictions: int

        self.mapper: BaseMapperWithPadding
        padding_symbol = self.mapper.get_padding_symbol()
        label_padding_index = self.mapper.get_label_idx(padding_symbol)

        # create a mask to distinguish padding from real tokens
        padding_mask = (labels != label_padding_index).type(torch.int64)
        num_predictions = torch.sum(padding_mask).item()

        # compute prediction (greedy, argmax in each time sequence)
        predictions = self.infer_model_outputs(model_outputs)

        # compare between predictions and labels, masking out padding
        correct_predictions_raw = (predictions == labels).type(torch.int64)
        correct_prediction_no_padding = correct_predictions_raw * padding_mask
        num_correct = torch.sum(correct_prediction_no_padding).item()

        return num_correct, num_predictions

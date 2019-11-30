import torch
from torch.utils import data
from configs import InferenceConfig
from models import BiLSTM


class BaseInferer(object):

    def __init__(self, config: InferenceConfig):
        self.config = config

    def infer_text_sample(self, tokenized_text: list):
        return NotImplementedError("A class deriving from BaseInferer must implement this method")

    def infer_text_sample_with_gold_labels(self, tokenized_text: list, gold_labels: list):
        return NotImplementedError("A class deriving from BaseInferer must implement this method")

    def infer_dataset(self, dataset: data.Dataset):
        return NotImplementedError("A class deriving from BaseInferer must implement this method")

    def compute_accuracy(self, dataset: data.Dataset):
        return NotImplementedError("A class deriving from BaseInferer must implement this method")


class BiLSTMGreedyInferer(BaseInferer):

    def __init__(self, config: InferenceConfig, model: BiLSTM):
        super().__init__(config)
        self.model = model
        self.mapper = model.mapper

    def _infer_batch_output(self, model_output: torch.tensor, actual_lengths: torch.tensor) -> list:
        idx_to_label = self.mapper.idx_to_label
        batch_size = model_output.size()[0]
        model_predictions = []

        _, predictions = torch.max(model_output, dim=1)
        for i in range(batch_size):
            sample_size_without_padding = actual_lengths[i]
            prediction_tokens = predictions[i]

            # cut predictions belonging to padding and move to cpu
            prediction_tokens = prediction_tokens[:sample_size_without_padding]
            prediction_tokens = prediction_tokens.cpu().numpy()
            labels = [idx_to_label[label_token] for label_token in prediction_tokens]
            model_predictions.append(labels)

        return model_predictions

    def _get_samples_length_without_padding(self, x: torch.tensor) -> torch.tensor:
        padding_index: int = self.mapper.token_to_idx["PADD"]
        mask: torch.tensor = x != padding_index
        samples_length = torch.sum(mask, dim=1)

        return samples_length

    def infer_dataset(self, dataset: data.Dataset) -> list:
        model_predictions = []

        dataset_loader = data.DataLoader(dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers)
        device = torch.device(self.config.device)
        model = self.model.to(device)

        with torch.no_grad():
            model.eval()

            for sample in dataset_loader:
                x, _ = sample
                x = x.to(device)
                output = model(x)

                samples_length = self._get_samples_length_without_padding(x)
                y_pred = self._infer_batch_output(output, samples_length)
                model_predictions += y_pred

        return model_predictions

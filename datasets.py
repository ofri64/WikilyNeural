from typing import Tuple, List

import torch
import torch.utils.data as data

from mappers import BaseMapper, BaseMapperWithPadding


class BaseDataset(data.Dataset):
    def __init__(self, filepath: str, mapper: BaseMapper):
        super().__init__()
        self.filepath = filepath
        self.mapper = mapper
        self.samples = []
        self.labels = []

    def _init_dataset(self) -> None:
        raise NotImplementedError("A dataset class must implement a method to read the dataset to memory")

    def init_dataset_if_not_initiated(self) -> None:
        if len(self.samples) == 0:
            self._init_dataset()

    def __len__(self) -> int:
        self.init_dataset_if_not_initiated()
        return len(self.samples)


class BiLSTMDataset(BaseDataset):
    def __init__(self, filepath: str, mapper: BaseMapperWithPadding, sequence_length: int = 65):
        super().__init__(filepath, mapper)
        self.sequence_length = sequence_length

    def _init_dataset(self) -> None:
        with open(self.filepath, "r", encoding="utf8") as f:
            curr_sentence = []
            curr_labels = []
            for line in f:

                if line == "\n":  # empty line denotes end of a sentence

                    # now add padding
                    if len(curr_labels) > 0:
                        # append to list of labels
                        curr_labels = self._prune_or_pad_sample(curr_labels)
                        self.labels.append(curr_labels)
                        curr_labels = []

                    # anyway append to list of samples and continue to next sentence
                    curr_sentence = self._prune_or_pad_sample(curr_sentence)
                    self.samples.append(curr_sentence)
                    curr_sentence = []

                else:  # append word and label to current sentence
                    tokens = line[:-1].split(self.mapper.split_char)

                    # check that indeed we have a label and it is not a blind test set
                    if len(tokens) == 2:
                        label = tokens[1]
                        curr_labels.append(label)

                    # any way we will have words to predict
                    word = tokens[0]
                    curr_sentence.append(word)

    def __getitem__(self, item_idx: int) -> Tuple[torch.tensor, torch.tensor]:
        self.init_dataset_if_not_initiated()

        # check if we have labels or it is a blind test set
        if len(self.labels) > 0:
            labels = self.labels[item_idx]
            labels_indices = [self.mapper.get_label_idx(label) for label in labels]
            y = torch.tensor(labels_indices)
        else:
            y = torch.tensor([])

        # even for test set we anyway have samples to predict
        sample = self.samples[item_idx]
        sample_indices = [self.mapper.get_token_idx(word) for word in sample]
        x = torch.tensor(sample_indices)

        return x, y

    def _prune_or_pad_sample(self, sample: List[str]) -> List[str]:
        # padding or pruning
        self.mapper: BaseMapperWithPadding
        const_len_sample: List[str]
        sample_length = len(sample)

        if sample_length > self.sequence_length:
            const_len_sample = sample[:self.sequence_length]
        else:
            padding_length = self.sequence_length - sample_length
            const_len_sample = sample + [self.mapper.get_padding_symbol()] * padding_length

        return const_len_sample

    def get_dataset_max_sequence_length(self):
        max_sequence_length = 0
        with open(self.filepath, "r", encoding="utf8") as f:
            current_sequence_length = 0
            for line in f:
                if line == "\n":  # empty line denotes end of a sentence
                    max_sequence_length = max(max_sequence_length, current_sequence_length)
                    current_sequence_length = 0
                else:
                    current_sequence_length += 1

        return max_sequence_length

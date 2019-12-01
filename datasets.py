import torch
import torch.utils.data as data
from mappers import TokenMapper, PADD, UNK


class SupervisedDataset(data.Dataset):
    """
    Pytorch's Dataset derived class to create data sample from
    a path to a file
    """
    def __init__(self, filepath: str, mapper: TokenMapper, sequence_len: int):
        super().__init__()
        self.filepath = filepath
        self.mapper = mapper
        self.sequence_len = sequence_len
        self.samples = []
        self.labels = []

    def _load_file(self) -> None:
        curr_sent = []
        curr_labels = []
        with open(self.filepath, "r", encoding="utf8") as f:
            for line in f:
                if line.startswith("#"):  # comment rows
                    continue
                if line == "\n":  # marks end of sentence
                    self.samples.append(curr_sent)
                    self.labels.append(curr_labels)
                    # clear before reading next sentence
                    curr_sent = []
                    curr_labels = []

                else:  # line for word in a sentence
                    tokens = line.split("\t")
                    word, label = tokens[1], tokens[3]
                    curr_sent.append(word)
                    curr_labels.append(label)

    def __len__(self) -> int:
        # perform lazy evaluation of data loading
        if len(self.samples) == 0:
            self._load_file()

        return len(self.samples)

    def __getitem__(self, item_idx: int) -> tuple:
        # lazy evaluation of data loading
        if len(self.samples) == 0:
            self._load_file()

        # retrieve sample and transform from tokens to indices
        unknown_index = self.mapper.token_to_idx[UNK]
        sent_tokens, sent_labels = self.samples[item_idx], self.labels[item_idx]
        sent_indices = [self.mapper.token_to_idx.get(word, unknown_index) for word in sent_tokens]
        labels_indices = [self.mapper.label_to_idx[label] for label in sent_labels]

        # add pudding or prune in order to gain unified sequence length
        current_sent_length = len(sent_indices)
        if current_sent_length > self.sequence_len:
            sent_indices = sent_indices[:self.sequence_len]
            labels_indices = labels_indices[:self.sequence_len]
        else:
            num_padding = self.sequence_len - current_sent_length
            sent_indices = sent_indices + [self.mapper.token_to_idx[PADD]] * num_padding
            labels_indices = labels_indices + [self.mapper.label_to_idx[PADD]] * num_padding

        # create tensors
        x = torch.tensor(sent_indices)
        y = torch.tensor(labels_indices)
        return x, y

import torch
import torch.utils.data as data
from mappers import TokenMapper, PADD, UNK

PADD = "PADD"
UNK = "UNK"


class TokenMapper(object):
    """
    Class for mapping discrete tokens in a training set
    to indices and back
    """
    def __init__(self, min_frequency: int, with_padding: bool = True):
        self.with_padding = with_padding
        self.min_frequency = min_frequency
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.label_to_idx = {}
        self.idx_to_label = {}

    def get_tokens_dim(self) -> int:
        return len(self.token_to_idx)

    def get_labels_dim(self) -> int:
        return len(self.label_to_idx)

    def _init_mappings(self) -> None:
        if self.with_padding:
            self.token_to_idx[PADD] = 0
            self.token_to_idx[UNK] = 1
            self.idx_to_token[0] = PADD
            self.idx_to_token[1] = UNK

            self.label_to_idx[PADD] = 0
            self.idx_to_label[0] = PADD

        else:
            self.token_to_idx[UNK] = 0
            self.idx_to_token[0] = UNK

    def _remove_non_frequent(self, words_frequencies) -> set:
        # remove word below min_frequency
        words = set()
        for word, frequency in words_frequencies.items():
            if frequency >= self.min_frequency:
                words.add(word)

        return words

    def create_mapping(self, filepath: str) -> None:
        words_frequencies = {}
        labels = set()

        with open(filepath, "r", encoding="utf8") as f:
            for line in f:
                if line != "\n" and not line.startswith("#"):
                    line_tokens = line.split("\t")
                    word = line_tokens[1]
                    label = line_tokens[3]

                    words_frequencies[word] = words_frequencies.get(word, 0) + 1
                    labels.add(label)

        # remove word below min_frequency
        words = self._remove_non_frequent(words_frequencies)

        # init mappings with padding and unknown indices
        self._init_mappings()

        # start index will be different if index 0 marked already as padding
        word_start_index = len(self.token_to_idx)
        label_start_index = len(self.label_to_idx)

        # transform token to indices
        for index, word in enumerate(words, word_start_index):
            self.token_to_idx[word] = index
            self.idx_to_token[index] = word

        for index, label in enumerate(labels, label_start_index):
            self.label_to_idx[label] = index
            self.idx_to_label[index] = label


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

import torch
import torch.utils.data as data


class TokenMapper(object):
    """
    Class for mapping discrete tokens in a training set
    to indices and back
    """
    def __init__(self, with_padding: bool = True):
        self.with_padding = with_padding
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.label_to_idx = {}
        self.idx_to_label = {}

    def get_tokens_dim(self) -> int:
        return len(self.token_to_idx)

    def get_labels_dim(self) -> int:
        return len(self.label_to_idx)

    def create_mapping(self, filepath: str) -> None:
        words = set()
        labels = set()

        with open(filepath, "r", encoding="utf8") as f:
            for line in f:
                if line != "\n" and not line.startswith("#"):
                    line_tokens = line.split("\t")
                    word = line_tokens[1]
                    label = line_tokens[3]

                    words.add(word)
                    labels.add(label)

        # create mappings
        if self.with_padding:
            padding_str = "PADD"
            self.token_to_idx[padding_str] = 0
            self.idx_to_token[0] = padding_str
            self.label_to_idx[padding_str] = 0
            self.idx_to_label[0] = padding_str

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
    def __init__(self, filepath: str, mapper: TokenMapper):
        super().__init__()
        self.filepath = filepath
        self.mapper = mapper
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
        sent_tokens, sent_labels = self.samples[item_idx], self.labels[item_idx]
        sent_indices = [self.mapper.token_to_idx[word] for word in sent_tokens]
        labels_indices = [self.mapper.label_to_idx[label] for label in sent_labels]

        # create tensors
        x = torch.tensor(sent_indices)
        y = torch.tensor(labels_indices)
        return x, y

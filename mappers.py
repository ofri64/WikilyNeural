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

    def serialize(self) -> dict:
        return {
            "with_padding": self.with_padding,
            "min_frequency": self.min_frequency,
            "token_to_idx": self.token_to_idx,
            "label_to_idx": self.label_to_idx,
            "idx_to_token": self.idx_to_token,
            "idx_to_label": self.idx_to_label
        }

    def deserialize(self, serialized_mapper: dict) -> None:
        self.with_padding = serialized_mapper["with_padding"]
        self.min_frequency = serialized_mapper["min_frequency"]
        self.token_to_idx = serialized_mapper["token_to_idx"]
        self.label_to_idx = serialized_mapper["label_to_idx"]
        self.idx_to_token = serialized_mapper["idx_to_token"]
        self.idx_to_label = serialized_mapper["idx_to_label"]

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
